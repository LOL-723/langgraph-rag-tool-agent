import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from llm.rag_service import rag_service

from core.config import settings
from llm.tools import TOOL_ARGUMENTS, TOOL_DESCRIPTIONS, TOOL_REGISTRY


RouteName = Literal["rag", "chat", "tool"]
MAX_VERIFIER_RETRIES = 3


class AgentState(TypedDict, total=False):
    question: str
    system_prompt: str | None
    answer: str
    route: RouteName
    use_rag: bool
    file_info: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    retrieved_docs: list[dict[str, Any]]
    rag_retrieval_mode: str
    logs: list[dict[str, Any]]
    retry_count: int


TOOL_ROUTER_PROMPT = (
    "You are a tool router. Decide whether the user's message needs one of the "
    "available tools. Match by semantic meaning, not exact wording. For example, "
    "'现在几点?' means the same thing as '获取当前时间' and should call "
    "get_current_time. If the user asks for time in another country, city, or "
    "region, pass that place in arguments.location. Return one valid JSON object "
    "only with this shape: "
    '{"tool_calls":[{"name":"tool_name","arguments":{"location":"place"}}]}. '
    'If no tool is needed, return {"tool_calls":[]}.'
)

VERIFIER_PROMPT = (
    "You are a strict answer verifier. Decide whether the assistant answer is "
    "grounded in the provided context and actually answers the user's question. "
    "For RAG answers, the uploaded document sources are authoritative; mark the "
    "answer as hallucinated if it adds facts not supported by the sources. For "
    "tool answers, tool results are authoritative; mark the answer as "
    "hallucinated if it contradicts or ignores them. For normal chat answers, "
    "judge relevance, internal consistency, and whether the answer appears to "
    "invent specific unsupported facts. Return one valid JSON object only with "
    'this shape: {"has_hallucination":false,"reason":"..."}'
)


def add_log(
    state: AgentState,
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


def router_node(state: AgentState) -> AgentState:
    question = state["question"]

    if state.get("file_info") and state.get("use_rag", False):
        route: RouteName = "rag"
        tool_calls: list[dict[str, Any]] = []
    else:
        tool_calls = _select_tool_calls(question)
        route = "tool" if tool_calls else "chat"

    return {
        "route": route,
        "tool_calls": tool_calls,
        "logs": add_log(
            state=state,
            node="router_node",
            message="route selected",
            extra={"route": route, "tool_calls": tool_calls},
        ),
    }


def route_decision(state: AgentState) -> Literal["rag_node", "tool_node", "answer_node"]:
    route = state["route"]
    if route == "rag":
        return "rag_node"
    if route == "tool":
        return "tool_node"
    return "answer_node"


def rag_node(state: AgentState) -> AgentState:

    question = state["question"]
    rag_result = rag_service.ask(question)
    sources = [source.model_dump() for source in rag_result.sources]

    return {
        "answer": rag_result.answer,
        "retrieved_docs": sources,
        "rag_retrieval_mode": rag_result.retrieval_mode,
        "logs": add_log(
            state=state,
            node="rag_node",
            message="answered from uploaded documents",
            extra={"source_count": len(sources), "retrieval_mode": rag_result.retrieval_mode},
        ),
    }


def tool_node(state: AgentState) -> AgentState:
    question = state["question"]
    tool_results: list[dict[str, Any]] = []

    for tool_call in state.get("tool_calls", []):
        name = tool_call.get("name")
        arguments = tool_call.get("arguments") or {}
        if not isinstance(name, str) or name not in TOOL_REGISTRY:
            continue
        if not isinstance(arguments, dict):
            arguments = {}

        result = TOOL_REGISTRY[name](**arguments)
        tool_results.append(
            {
                "name": name,
                "description": TOOL_DESCRIPTIONS.get(name, ""),
                "result": result,
            }
        )

    answer = _chat_completion(
        user_message=question,
        system_prompt=(
            "Tool results are authoritative. Answer the user's request using these "
            f"tool results: {json.dumps(tool_results, ensure_ascii=False)}"
        ),
    )

    return {
        "tool_results": tool_results,
        "answer": answer,
        "logs": add_log(
            state=state,
            node="tool_node",
            message="tools executed",
            extra={"tool_count": len(tool_results)},
        ),
    }


def answer_node(state: AgentState) -> AgentState:
    answer = _chat_completion(
        user_message=state["question"],
        system_prompt=state.get("system_prompt"),
    )

    return {
        "answer": answer,
        "logs": add_log(
            state=state,
            node="answer_node",
            message="answered by model",
        ),
    }


def verifier_node(state: AgentState) -> AgentState:
    current_state = dict(state)

    for _ in range(MAX_VERIFIER_RETRIES + 1):
        verification = _verify_answer(current_state)
        has_hallucination = verification.get("has_hallucination", True)
        retry_count = int(current_state.get("retry_count", 0))

        current_state["logs"] = add_log(
            state=current_state,
            node="verifier_node",
            message="answer verification completed",
            extra={
                "has_hallucination": has_hallucination,
                "reason": verification.get("reason", ""),
                "retry_count": retry_count,
            },
        )

        if not has_hallucination:
            return current_state

        if retry_count >= MAX_VERIFIER_RETRIES:
            fallback_answer = (
                "当前回复经过多次校验仍可能存在幻觉或偏题，无法可靠回答。"
                "请补充更明确的问题、相关文档或可调用工具结果后再试。"
            )
            current_state["answer"] = fallback_answer
            current_state["logs"] = add_log(
                state=current_state,
                node="verifier_node",
                message="max verification retries reached",
                extra={"max_retries": MAX_VERIFIER_RETRIES},
            )
            return current_state

        current_state = _rerun_from_router(current_state, retry_count + 1)

    return current_state


def build_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("rag_node", rag_node)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.add_node("answer_node", answer_node)
    graph_builder.add_node("verifier_node", verifier_node)

    graph_builder.add_edge(START, "router_node")
    graph_builder.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "rag_node": "rag_node",
            "tool_node": "tool_node",
            "answer_node": "answer_node",
        },
    )
    graph_builder.add_edge("rag_node", "verifier_node")
    graph_builder.add_edge("tool_node", "verifier_node")
    graph_builder.add_edge("answer_node", "verifier_node")
    graph_builder.add_edge("verifier_node", END)

    return graph_builder.compile()


def _verify_answer(state: AgentState) -> dict[str, Any]:
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


def _verification_context(state: AgentState) -> dict[str, Any]:
    route = state.get("route", "chat")
    context: dict[str, Any] = {}

    if route == "rag":
        context["retrieved_docs"] = state.get("retrieved_docs", [])
        context["rag_retrieval_mode"] = state.get("rag_retrieval_mode")
    elif route == "tool":
        context["tool_calls"] = state.get("tool_calls", [])
        context["tool_results"] = state.get("tool_results", [])
    else:
        context["system_prompt"] = state.get("system_prompt")

    return context


def _rerun_from_router(state: AgentState, retry_count: int) -> AgentState:
    retry_state: AgentState = {
        "question": state["question"],
        "system_prompt": state.get("system_prompt"),
        "use_rag": state.get("use_rag", False),
        "retry_count": retry_count,
        "logs": add_log(
            state=state,
            node="verifier_node",
            message="hallucination detected, rerunning from router",
            extra={"retry_count": retry_count},
        ),
    }
    if state.get("file_info"):
        retry_state["file_info"] = state["file_info"]

    retry_state = _merge_state(retry_state, router_node(retry_state))
    next_node = route_decision(retry_state)
    action_nodes = {
        "rag_node": rag_node,
        "tool_node": tool_node,
        "answer_node": answer_node,
    }
    return _merge_state(retry_state, action_nodes[next_node](retry_state))


def _merge_state(state: AgentState, update: AgentState) -> AgentState:
    merged = dict(state)
    merged.update(update)
    return merged


def _select_tool_calls(user_message: str) -> list[dict[str, Any]]:
    available_tools = [
        {
            "name": name,
            "description": description,
            "arguments": TOOL_ARGUMENTS.get(name, {}),
        }
        for name, description in TOOL_DESCRIPTIONS.items()
    ]
    if not available_tools:
        return []

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

    return [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]


def _chat_completion(user_message: str, system_prompt: str | None = None) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    response = _openai_client().chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def _openai_client() -> OpenAI:
    return OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        timeout=settings.LLM_TIMEOUT,
    )
