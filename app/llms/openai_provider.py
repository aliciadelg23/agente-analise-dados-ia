"""OpenAI adapter for the LLM abstraction."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from app.core.exceptions import MissingCredentialsError
from app.llms.base import LLMProvider, LLMResponse, Message


class OpenAIProvider(LLMProvider):
    """Adapter over the OpenAI Chat Completions API."""

    name = "openai"

    def __init__(self, api_key: str | None, default_model: str = "gpt-4o-mini") -> None:
        if not api_key:
            raise MissingCredentialsError("OPENAI_API_KEY is not configured.")
        self._client = OpenAI(api_key=api_key)
        self.default_model = default_model

    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        chosen_model = model or self.default_model
        payload = [{"role": message.role, "content": message.content} for message in messages]
        response = self._client.chat.completions.create(
            model=chosen_model,
            messages=payload,
            **kwargs,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage.model_dump() if response.usage is not None else None
        return LLMResponse(
            content=content,
            model=chosen_model,
            provider=self.name,
            usage=usage,
        )
