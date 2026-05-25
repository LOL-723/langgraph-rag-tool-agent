import json
from typing import Any

from openai import OpenAI

from core.config import settings
from llm import langgraph_def


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
    ):
        self.model = model
        self.langgraph = langgraph_def.build_graph()

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
    ) -> str:
        if not user_message or not user_message.strip():
            raise ValueError("message cannot be empty")

        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def stream_chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
    ):
        if not user_message or not user_message.strip():
            raise ValueError("message cannot be empty")

        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if content:
                yield content

    def json_chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
        file: Any | None = None,
        use_rag: bool = False,
    ) -> dict[str, Any]:
        if not user_message or not user_message.strip():
            raise ValueError("message cannot be empty")

        file_info = self._upload_optional_file(file) if use_rag else None
        graph_result = self.run_langgraph(
            user_message=user_message,
            system_prompt=system_prompt,
            file_info=file_info,
            use_rag=use_rag,
        )
        return self._format_graph_json_result(graph_result)

    def run_langgraph(
        self,
        user_message: str,
        system_prompt: str | None = None,
        file_info: dict[str, Any] | None = None,
        use_rag: bool = False,
    ) -> langgraph_def.AgentState:
        initial_state: langgraph_def.AgentState = {
            "question": user_message,
            "system_prompt": system_prompt,
            "use_rag": use_rag,
            "retry_count": 0,
            "logs": [],
        }
        if file_info:
            initial_state["file_info"] = file_info

        return self.langgraph.invoke(initial_state)

    def _format_graph_json_result(
        self,
        graph_result: langgraph_def.AgentState,
    ) -> dict[str, Any]:
        route = graph_result.get("route", "chat")
        answer = graph_result.get("answer", "")
        use_local_retrieval = route == "rag" and graph_result.get("rag_retrieval_mode") == "local"

        try:
            parsed_answer = json.loads(answer)
        except json.JSONDecodeError:
            message = f"{answer}(本地检索)" if use_local_retrieval and answer else answer
            return {
                "route": route,
                "message": message,
            }

        if isinstance(parsed_answer, dict):
            if use_local_retrieval:
                message = str(parsed_answer.get("message", ""))
                parsed_answer["message"] = f"{message}(本地检索)"
            parsed_answer["route"] = route
            return parsed_answer

        result = {
            "route": route,
            "data": parsed_answer,
        }
        if use_local_retrieval:
            result["message"] = "(本地检索)"
        return result

    def _upload_optional_file(self, file: Any | None) -> dict[str, Any] | None:
        if file is None:
            return None

        import anyio
        from llm.rag_service import rag_service

        upload_result = anyio.run(rag_service.upload, file)
        return upload_result.model_dump()

    def _validate_json_semantics(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            self._validate_json_value(key, value)

    def _validate_json_value(self, key: str, value: Any) -> None:
        key_lower = key.lower()

        if isinstance(value, dict):
            self._validate_json_semantics(value)
            return

        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    self._validate_json_semantics(item)
            if self._key_contains(key_lower, ("skill", "tag", "item", "hobby")):
                invalid_items = [item for item in value if not isinstance(item, str)]
                if invalid_items:
                    raise ValueError(f"{key} must contain strings only")
            return

        if self._key_contains(key_lower, ("name", "title", "email", "phone", "address", "date")):
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")

        if self._key_contains(key_lower, ("age", "count", "quantity", "score", "number")):
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(f"{key} must be a number")

    @staticmethod
    def _key_contains(key: str, words: tuple[str, ...]) -> bool:
        return any(word in key for word in words)


llm_client = LLMClient(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    model=settings.LLM_MODEL,
    timeout=settings.LLM_TIMEOUT,
)
