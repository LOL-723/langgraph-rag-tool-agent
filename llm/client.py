import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel
from core.config import settings


class LLMClient:
    JSON_OUTPUT_PROMPT = (
        "You must preserve and follow any previous system message from the user. "
        "The following rules only constrain the output format. "
        "Return one valid JSON object only. "
        "Do not include markdown, comments, code fences, or extra text. "
        "Every value must match the provided JSON schema exactly. "
        "Do not use a string when the schema requires a number, and do not use a number "
        "when the schema requires a string."
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
        output_schema: type[BaseModel],
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        if not user_message or not user_message.strip():
            raise ValueError("message cannot be empty")

        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        schema = output_schema.model_json_schema()
        messages.append(
            {
                "role": "system",
                "content": (
                    f"{self.JSON_OUTPUT_PROMPT}\n"
                    f"JSON schema: {json.dumps(schema, ensure_ascii=False)}"
                ),
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
        return output_schema.model_validate(data).model_dump()


llm_client = LLMClient(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    model=settings.LLM_MODEL,
    timeout=settings.LLM_TIMEOUT
)
