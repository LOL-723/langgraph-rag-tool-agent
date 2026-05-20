import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from core.config import settings
from llm.tools import TOOL_ARGUMENTS, TOOL_DESCRIPTIONS, TOOL_REGISTRY


RouteName = Literal["rag", "chat", "tool"]


class AgentState(TypedDict, total=False):
    question: str
    system_prompt: str | None
    answer: str
    route: RouteName
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

    if state.get("file_info") and _has_rag_intent(question):
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
    from llm.rag_service import rag_service

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


def build_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("rag_node", rag_node)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.add_node("answer_node", answer_node)

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
    graph_builder.add_edge("rag_node", END)
    graph_builder.add_edge("tool_node", END)
    graph_builder.add_edge("answer_node", END)

    return graph_builder.compile()


def _has_rag_intent(question: str) -> bool:
    if any(
        keyword in question
        for keyword in (
            "根据文档",
            "基于文档",
            "参考文档",
            "文档中",
            "资料中",
            "上传的文件",
            "上传文件",
            "RAG",
            "rag",
        )
    ):
        return True
    rag_keywords = ("根据文档", "基于文档", "参考文档", "文档中", "资料中", "RAG", "rag")
    return any(keyword in question for keyword in rag_keywords)


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
