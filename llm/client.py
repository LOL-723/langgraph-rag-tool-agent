import json
from typing import Any

from openai import OpenAI

from core.config import settings
from llm import langgraph


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        temperature: float = 0.1,
    ):
        self.model = model
        self.temperature = temperature
        self.langgraph = langgraph.build_graph()

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
            temperature=self.temperature,
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
            temperature=self.temperature,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if content:
                yield content

    def Agent_Ask(
        self,
        user_message: str,
        system_prompt: str | None = None,
        file: Any | None = None,
        use_rag: bool = False,
    ) -> dict[str, Any]:
        if not user_message or not user_message.strip():
            raise ValueError("message cannot be empty")
        if use_rag and file is None:
            raise ValueError("RAG requires an uploaded file")

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
    ) -> langgraph.LangGraphState:
        initial_state: langgraph.LangGraphState = {
            "question": user_message,
            "system_prompt": system_prompt,
            "use_rag": use_rag,
            "retry_count": 0,
            "verification_count": 0,
            "answer_retry_count": 0,
            "rag_retry_count": 0,
            "tool_retry_count": 0,
            "chat_retry_count": 0,
            "router_retry_count": 0,
            "logs": [],
        }
        if file_info:
            initial_state["file_info"] = file_info

        return self.langgraph.invoke(initial_state)

    def _format_graph_json_result(
        self,
        graph_result: langgraph.LangGraphState,
    ) -> dict[str, Any]:
        route = graph_result.get("route", "chat")
        answer = graph_result.get("answer", "")
        end_status = graph_result.get("end_status")
        use_local_retrieval = route == "rag" and graph_result.get("rag_retrieval_mode") == "local"
        rag_metadata = self._rag_response_metadata(graph_result)

        try:
            parsed_answer = json.loads(answer)
        except json.JSONDecodeError:
            message = f"{answer}(local retrieval)" if use_local_retrieval and answer else answer
            result = {
                "route": route,
                "end_status": end_status,
                "message": message,
            }
            result.update(rag_metadata)
            return result

        if isinstance(parsed_answer, dict):
            if use_local_retrieval:
                message = str(parsed_answer.get("message", ""))
                parsed_answer["message"] = f"{message}(local retrieval)"
            parsed_answer["route"] = route
            parsed_answer["end_status"] = end_status
            parsed_answer.update(rag_metadata)
            return parsed_answer

        result = {
            "route": route,
            "end_status": end_status,
            "data": parsed_answer,
        }
        if use_local_retrieval:
            result["message"] = "(local retrieval)"
        result.update(rag_metadata)
        return result

    @staticmethod
    def _rag_response_metadata(graph_result: langgraph.LangGraphState) -> dict[str, Any]:
        if graph_result.get("route") != "rag":
            return {}

        file_info = graph_result.get("file_info") or {}
        return {
            "rag_document": {
                "document_id": file_info.get("document_id"),
                "filename": file_info.get("filename"),
                "chunk_count": file_info.get("chunk_count"),
            },
            "rag_sources": graph_result.get("retrieved_docs", []),
            "rag_retrieval_mode": graph_result.get("rag_retrieval_mode"),
            "rag_query_str": graph_result.get("rag_query_str"),
        }

    def _upload_optional_file(self, file: Any | None) -> dict[str, Any] | None:
        if file is None:
            return None

        import anyio
        from llm.rag_service import rag_service

        upload_result = anyio.run(rag_service.upload, file)
        return upload_result.model_dump()

llm_client = LLMClient(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    model=settings.LLM_MODEL,
    timeout=settings.LLM_TIMEOUT,
    temperature=settings.LLM_TEMPERATURE,
)
