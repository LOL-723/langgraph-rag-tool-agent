from typing import Any

from openai import OpenAI

from core.config import settings
from llm.Agent.state import AgentState
from llm.tools import TOOL_ARGUMENTS, TOOL_DESCRIPTIONS


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


def _available_tools(excluded_tools: list[str] | None = None) -> list[dict[str, Any]]:
    excluded_tool_names = set(excluded_tools or [])
    return [
        {
            "name": name,
            "description": description,
            "arguments": TOOL_ARGUMENTS.get(name, {}),
        }
        for name, description in TOOL_DESCRIPTIONS.items()
        if name not in excluded_tool_names
    ]


def _chat_completion(
    system_prompt: str,
    user_message: str,
    response_format: dict[str, str] | None = None,
) -> str:
    request: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
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
