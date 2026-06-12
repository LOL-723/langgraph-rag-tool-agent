import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from core.config import settings
from llm.Agent.nodes import agent_loop_node, planner_node, select_next_step_node
from llm.Agent.prompt import FINAL_RESULT_SUMMARY_PROMPT
from llm.Agent.state import AgentState, MAX_PLAN_STEPS, MAX_REPLAN_COUNT, MAX_STEP_REPLAN_COUNT
from llm.tools import TOOL_ARGUMENTS, TOOL_DESCRIPTIONS, TOOL_REGISTRY


RouteName = Literal["agent", "chat", "tool"]
VerifierNext = Literal["end", "answer_node", "tool_selector_node", "router_node"]
EndStatus = Literal["finished", "failed"]
TOOL_ROUTE_TOOL_NAMES = ("get_current_time", "calculate_expression", "get_today_weather")
MAX_DIRECT_ANSWER_RETRIES = 1
MAX_TOOL_RETRIES = 2
MAX_CHAT_RETRIES = 1
MAX_ROUTER_RETRIES = 2
MAX_AGENT_NODE_ITERATIONS = MAX_PLAN_STEPS * (
    1 + MAX_REPLAN_COUNT + MAX_STEP_REPLAN_COUNT
) + 3


class LangGraphState(TypedDict, total=False):
    question: str
    system_prompt: str | None
    answer: str
    route: RouteName
    use_rag: bool
    file_info: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    agent_state: dict[str, Any]
    logs: list[dict[str, Any]]
    retry_count: int
    verification_count: int
    answer_retry_count: int
    tool_retry_count: int
    chat_retry_count: int
    router_retry_count: int
    verifier_next: VerifierNext
    end_status: EndStatus
    verifier_reason: str
    has_hallucination: bool


TOOL_ROUTER_PROMPT = (
    "You are a route classifier. Decide whether the user's message needs one of "
    "the available tools. Match by semantic meaning, not exact wording. Return "
    'one valid JSON object only with this shape: {"use_tool":true}. '
    'If no tool is needed, return {"use_tool":false}.'
)

AGENT_ROUTER_PROMPT = (
    "You are a route classifier. Decide whether the user's message needs the "
    "multi-step Agent. Use the Agent for complex questions that need planning, "
    "multi-step reasoning, investigation, diagnosis, comparison, synthesis, or "
    "uploaded-document/RAG retrieval. Do not use the Agent for simple chat, "
    "single arithmetic, current time, or simple weather requests. Return one "
    'valid JSON object only with this shape: {"use_agent":true}. '
    'If the Agent is not needed, return {"use_agent":false}.'
)

TOOL_SELECTOR_PROMPT = (
    "You are a tool selector. Decide which available tools should be called. "
    "Match by semantic meaning, not exact wording. For example, '现在几点?' means "
    "the same thing as '获取当前时间' and should call get_current_time. If the user "
    "asks for time in another country, city, or region, pass that place in "
    "arguments.location. If the user asks to calculate, solve, evaluate, or asks "
    "a basic arithmetic expression, call calculate_expression and pass only the "
    "math expression in arguments.expression, for example '(24+1)*4'. If the user "
    "asks about weather, today's weather, current weather, temperature, sunrise, "
    "or sunset, call get_today_weather. Pass arguments.location only when the "
    "user names a specific city, country, or region; otherwise omit it so the "
    "tool defaults to Shenyang. Return one "
    "valid JSON object only with this shape: "
    '{"tool_calls":[{"name":"tool_name","arguments":{"expression":"(24+1)*4"}}]}. '
    'If no tool is needed, return {"tool_calls":[]}.'
)

VERIFIER_PROMPT = (
    "You are a strict answer verifier. Decide whether the assistant answer is "
    "grounded in the provided context and actually answers the user's question. "
    "tool answers, tool results are authoritative; mark the answer as "
    "hallucinated if it contradicts or ignores them. For normal chat answers, "
    "judge relevance, internal consistency, and whether the answer appears to "
    "invent specific unsupported facts. Return one valid JSON object only with "
    'this shape: {"has_hallucination":false,"reason":"..."}'
)


def add_log(
    state: LangGraphState,
    node: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    logs = state.get("logs", [])
    log_item: dict[str, Any] = {
        "node": node,
        "message": message,
    }
    if extra:
        log_item.update(extra)
    return logs + [log_item]


def router_node(state: LangGraphState) -> LangGraphState:
    question = state["question"]

    if state.get("file_info") and state.get("use_rag", False):
        route: RouteName = "agent"
    else:
        route = _select_route(question)

    return {
        "route": route,
        "tool_calls": [],
        "logs": add_log(
            state=state,
            node="router_node",
            message="route selected",
            extra={"route": route},
        ),
    }


def route_decision(
    state: LangGraphState,
) -> Literal["agent_node", "tool_selector_node", "answer_node"]:
    route = state["route"]
    if route == "agent":
        return "agent_node"
    if route == "tool":
        return "tool_selector_node"
    return "answer_node"


def tool_selector_node(state: LangGraphState) -> LangGraphState:
    question = state["question"]
    tool_calls = _select_tool_calls(question)

    return {
        "tool_calls": tool_calls,
        "logs": add_log(
            state=state,
            node="tool_selector_node",
            message="tools selected",
            extra={"tool_calls": tool_calls},
        ),
    }


def tool_executor_node(state: LangGraphState) -> LangGraphState:
    tool_results: list[dict[str, Any]] = []

    for tool_call in state.get("tool_calls", []):
        name = tool_call.get("name")
        arguments = tool_call.get("arguments") or {}
        if not _is_tool_route_tool(name):
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        arguments = dict(arguments)

        result = TOOL_REGISTRY[name](**arguments)
        tool_results.append(
            {
                "name": name,
                "description": TOOL_DESCRIPTIONS.get(name, ""),
                "result": result,
            }
        )

    return {
        "tool_results": tool_results,
        "logs": add_log(
            state=state,
            node="tool_executor_node",
            message="tools executed",
            extra={"tool_count": len(tool_results)},
        ),
    }


def answer_node(state: LangGraphState) -> LangGraphState:
    answer = _chat_completion(
        user_message=_answer_user_message(state),
        system_prompt=_answer_system_prompt(state),
    )

    return {
        "answer": answer,
        "logs": add_log(
            state=state,
            node="answer_node",
            message="answered by model",
        ),
    }


def agent_node(state: LangGraphState) -> LangGraphState:
    agent_state: AgentState = {
        "question": state["question"],
        "document_id": _document_id_from_state(state),
        "logs": [],
        "failed_tools": [],
        "overthink_counts": {},
        "no_finding_counts": {},
        "subagent_results": [],
        "agent_depth": 0,
    }

    for _ in range(MAX_AGENT_NODE_ITERATIONS):
        if agent_state.get("planner_mode") in {"replan", "step_replan"} or "plan" not in agent_state:
            agent_state = _merge_agent_state(agent_state, planner_node(agent_state))
            if agent_state.get("agent_status") == "failed":
                return _agent_graph_failed(state, agent_state)
            continue

        agent_state = _merge_agent_state(agent_state, select_next_step_node(agent_state))
        if agent_state.get("agent_status") == "failed":
            return _agent_graph_failed(state, agent_state)
        if agent_state.get("should_continue_next") == "finish":
            answer = _summarize_agent_answer(agent_state)
            return {
                "route": "agent",
                "answer": answer,
                "end_status": "finished",
                "agent_state": dict(agent_state),
                "logs": add_log(
                    state=state,
                    node="agent_node",
                    message="agent completed",
                    extra={
                        "step_count": len(agent_state.get("step_results", [])),
                    },
                ),
            }

        agent_state = _merge_agent_state(agent_state, agent_loop_node(agent_state))
        if agent_state.get("agent_status") == "failed":
            return _agent_graph_failed(state, agent_state)

    agent_state["error"] = "agent exceeded graph iteration limit"
    return _agent_graph_failed(state, agent_state)


def verifier_node(state: LangGraphState) -> LangGraphState:
    verification = _verify_answer(state)
    has_hallucination = verification.get("has_hallucination", True)
    verification_count = int(state.get("verification_count", 0)) + 1
    reason = str(verification.get("reason", ""))

    update: LangGraphState = {
        "verification_count": verification_count,
        "has_hallucination": has_hallucination,
        "verifier_reason": reason,
        "logs": add_log(
            state=state,
            node="verifier_node",
            message="answer verification completed",
            extra={
                "has_hallucination": has_hallucination,
                "reason": reason,
                "verification_count": verification_count,
            },
        ),
    }

    if not has_hallucination:
        update["verifier_next"] = "end"
        update["end_status"] = "finished"
        return update

    next_node, retry_update = _next_verifier_step(state)
    update.update(retry_update)
    update["verifier_next"] = next_node

    if next_node == "end":
        update["end_status"] = "failed"
        update["answer"] = (
            "The answer still failed verification after all retry paths were "
            "exhausted, so no reliable final answer can be provided."
        )
        update["logs"] = add_log(
            state=_merge_state(state, update),
            node="verifier_node",
            message="verification retry budget exhausted",
            extra={
                "end_status": "failed",
                "verification_count": verification_count,
            },
        )

    return update


def build_graph():
    graph_builder = StateGraph(LangGraphState)

    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("agent_node", agent_node)
    graph_builder.add_node("tool_selector_node", tool_selector_node)
    graph_builder.add_node("tool_executor_node", tool_executor_node)
    graph_builder.add_node("answer_node", answer_node)
    graph_builder.add_node("verifier_node", verifier_node)

    graph_builder.add_edge(START, "router_node")
    graph_builder.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "agent_node": "agent_node",
            "tool_selector_node": "tool_selector_node",
            "answer_node": "answer_node",
        },
    )
    graph_builder.add_edge("agent_node", END)
    graph_builder.add_edge("tool_selector_node", "tool_executor_node")
    graph_builder.add_edge("tool_executor_node", "answer_node")
    graph_builder.add_edge("answer_node", "verifier_node")
    graph_builder.add_conditional_edges(
        "verifier_node",
        verifier_decision,
        {
            "end": END,
            "answer_node": "answer_node",
            "tool_selector_node": "tool_selector_node",
            "router_node": "router_node",
        },
    )

    return graph_builder.compile()


def verifier_decision(state: LangGraphState) -> VerifierNext:
    return state.get("verifier_next", "end")


def _next_verifier_step(state: LangGraphState) -> tuple[VerifierNext, LangGraphState]:
    answer_retry_count = int(state.get("answer_retry_count", 0))
    tool_retry_count = int(state.get("tool_retry_count", 0))
    chat_retry_count = int(state.get("chat_retry_count", 0))
    router_retry_count = int(state.get("router_retry_count", 0))

    if router_retry_count > 0:
        if router_retry_count < MAX_ROUTER_RETRIES:
            return "router_node", {"router_retry_count": router_retry_count + 1}
        return "end", {}

    if answer_retry_count < MAX_DIRECT_ANSWER_RETRIES:
        return "answer_node", {"answer_retry_count": answer_retry_count + 1}

    route = state.get("route", "chat")
    if route == "tool" and tool_retry_count < MAX_TOOL_RETRIES:
        return "tool_selector_node", {"tool_retry_count": tool_retry_count + 1}
    if route == "chat" and chat_retry_count < MAX_CHAT_RETRIES:
        return "answer_node", {"chat_retry_count": chat_retry_count + 1}

    if router_retry_count < MAX_ROUTER_RETRIES:
        return "router_node", {"router_retry_count": router_retry_count + 1}

    return "end", {}


def _verify_answer(state: LangGraphState) -> dict[str, Any]:
    answer = state.get("answer", "")
    if not answer or not answer.strip():
        return {
            "has_hallucination": True,
            "reason": "answer is empty",
        }

    payload = {
        "question": state["question"],
        "route": state.get("route", "chat"),
        "answer": answer,
        "context": _verification_context(state),
    }

    response = _openai_client().chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": VERIFIER_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=settings.LLM_TEMPERATURE,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {
            "has_hallucination": True,
            "reason": "verifier returned invalid JSON",
        }

    if not isinstance(data, dict):
        return {
            "has_hallucination": True,
            "reason": "verifier returned non-object JSON",
        }

    return {
        "has_hallucination": _json_bool(data.get("has_hallucination"), default=True),
        "reason": str(data.get("reason", "")),
    }


def _json_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default


def _verification_context(state: LangGraphState) -> dict[str, Any]:
    route = state.get("route", "chat")
    context: dict[str, Any] = {}

    if route == "tool":
        context["tool_calls"] = state.get("tool_calls", [])
        context["tool_results"] = state.get("tool_results", [])
    else:
        context["system_prompt"] = state.get("system_prompt")

    return context


def _merge_state(state: LangGraphState, update: LangGraphState) -> LangGraphState:
    merged = dict(state)
    merged.update(update)
    return merged


def _merge_agent_state(state: AgentState, update: AgentState) -> AgentState:
    merged = dict(state)
    merged.update(update)
    return merged


def _document_id_from_state(state: LangGraphState) -> str | None:
    file_info = state.get("file_info") or {}
    document_id = str(file_info.get("document_id", "")).strip()
    return document_id or None


def _summarize_agent_answer(agent_state: AgentState) -> str:
    payload = {
        "question": agent_state.get("question", ""),
        "plan": agent_state.get("plan", []),
        "step_results": agent_state.get("step_results", []),
        "failed_tools": agent_state.get("failed_tools", []),
        "document_id": agent_state.get("document_id"),
    }
    content = _chat_completion(
        user_message=json.dumps(payload, ensure_ascii=False),
        system_prompt=FINAL_RESULT_SUMMARY_PROMPT,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content
    if not isinstance(data, dict):
        return content
    final_answer = data.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        return final_answer
    return content


def _agent_graph_failed(
    state: LangGraphState,
    agent_state: AgentState,
) -> LangGraphState:
    error = str(agent_state.get("error") or "agent failed")
    return {
        "route": "agent",
        "answer": error,
        "end_status": "failed",
        "agent_state": dict(agent_state),
        "logs": add_log(
            state=state,
            node="agent_node",
            message="agent failed",
            extra={"error": error},
        ),
    }


def _answer_system_prompt(state: LangGraphState) -> str | None:
    route = state.get("route", "chat")
    user_system_prompt = state.get("system_prompt")

    if route == "tool":
        prompt = (
            "You are an assistant that answers from tool results. Tool results are authoritative. "
            "Answer the user based on the tool results. If the tool results are not usable, "
            "say that the tool results cannot answer the question. Do not invent facts."
        )
    else:
        return user_system_prompt

    if user_system_prompt and user_system_prompt.strip():
        return f"{user_system_prompt.strip()}\n\n{prompt}"
    return prompt


def _answer_user_message(state: LangGraphState) -> str:
    question = state["question"]
    route = state.get("route", "chat")

    if route == "tool":
        return (
            f"User question:\n{question}\n\n"
            "Tool results:\n"
            f"{json.dumps(state.get('tool_results', []), ensure_ascii=False)}\n\n"
            "Answer based on the tool results."
        )

    return question


def _select_route(user_message: str) -> RouteName:
    if _should_route_to_agent(user_message):
        return "agent"
    if _should_route_to_tool(user_message):
        return "tool"
    return "chat"


def _should_route_to_agent(user_message: str) -> bool:
    response = _openai_client().chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": AGENT_ROUTER_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=settings.LLM_TEMPERATURE,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or '{"use_agent":false}'
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return False

    if not isinstance(data, dict):
        return False

    return _json_bool(data.get("use_agent"), default=False)


def _should_route_to_tool(user_message: str) -> bool:
    available_tools = _available_tools()
    if not available_tools:
        return False

    response = _openai_client().chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": TOOL_ROUTER_PROMPT},
            {
                "role": "system",
                "content": (
                    "Available tools: "
                    f"{json.dumps(available_tools, ensure_ascii=False)}"
                ),
            },
            {"role": "user", "content": user_message},
        ],
        temperature=settings.LLM_TEMPERATURE,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or '{"use_tool":false}'
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return False

    if not isinstance(data, dict):
        return False

    return _json_bool(data.get("use_tool"), default=False)


def _available_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": TOOL_DESCRIPTIONS.get(name, ""),
            "arguments": TOOL_ARGUMENTS.get(name, {}),
        }
        for name in TOOL_ROUTE_TOOL_NAMES
        if name in TOOL_REGISTRY
    ]


def _is_tool_route_tool(name: object) -> bool:
    return isinstance(name, str) and name in TOOL_ROUTE_TOOL_NAMES and name in TOOL_REGISTRY


def _select_tool_calls(user_message: str) -> list[dict[str, Any]]:
    available_tools = _available_tools()
    if not available_tools:
        return []

    response = _openai_client().chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": TOOL_SELECTOR_PROMPT},
            {
                "role": "system",
                "content": (
                    "Available tools: "
                    f"{json.dumps(available_tools, ensure_ascii=False)}"
                ),
            },
            {"role": "user", "content": user_message},
        ],
        temperature=settings.LLM_TEMPERATURE,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or '{"tool_calls":[]}'
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, dict):
        return []

    tool_calls = data.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return []

    return [
        tool_call
        for tool_call in tool_calls
        if isinstance(tool_call, dict) and _is_tool_route_tool(tool_call.get("name"))
    ]


def _chat_completion(
    user_message: str,
    system_prompt: str | None = None,
    response_format: dict[str, str] | None = None,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    request: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "temperature": settings.LLM_TEMPERATURE,
    }
    if response_format is not None:
        request["response_format"] = response_format

    response = _openai_client().chat.completions.create(**request)
    return response.choices[0].message.content or ""


def _openai_client() -> OpenAI:
    return OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        timeout=settings.LLM_TIMEOUT,
    )
