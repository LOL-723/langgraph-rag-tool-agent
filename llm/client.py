import json
from typing import Any

from openai import OpenAI
from core.config import settings


class LLMClient:
    JSON_OUTPUT_PROMPT = (
        "You must preserve and follow any previous system message from the user. "
        "The following rules only constrain the output format. "
        "Return one valid JSON object only. Choose the JSON property names yourself "
        "according to the user's request and the extracted information. "
        "Do not include markdown, comments, code fences, or extra text. "
        "Use conventional JSON value types: names, titles, emails, phone numbers, "
        "addresses, descriptions, summaries, and dates should be strings; ages, counts, "
        "scores, and quantities should be numbers; lists of skills, tags, or items should "
        "be arrays. Never put a number in a name field."
    )

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
    ):
        self.model = model

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
    ) -> dict[str, Any]:
        if not user_message or not user_message.strip():
            raise ValueError("message cannot be empty")

        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        messages.append(
            {
                "role": "system",
                "content": self.JSON_OUTPUT_PROMPT,
            }
        )
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("LLM response must be a JSON object")

        self._validate_json_semantics(data)
        return data

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
    timeout=settings.LLM_TIMEOUT
)
