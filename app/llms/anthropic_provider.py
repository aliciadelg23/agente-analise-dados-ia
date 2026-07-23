"""Anthropic (Claude) adapter for the LLM abstraction."""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from app.core.exceptions import MissingCredentialsError
from app.llms.base import LLMProvider, LLMResponse, Message

_DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(LLMProvider):
    """Adapter over the Anthropic Messages API.

    Anthropic separates the system prompt from the message list, so
    we lift any ``system`` role message out of ``messages`` and pass
    it as the top-level ``system`` argument.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        default_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        if not api_key:
            raise MissingCredentialsError("ANTHROPIC_API_KEY is not configured.")
        self._client = Anthropic(api_key=api_key)
        self.default_model = default_model

    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        chosen_model = model or self.default_model
        system_prompt = next(
            (message.content for message in messages if message.role == "system"),
            None,
        )
        remaining = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role != "system"
        ]
        max_tokens = kwargs.pop("max_tokens", _DEFAULT_MAX_TOKENS)
        response = self._client.messages.create(
            model=chosen_model,
            system=system_prompt,
            messages=remaining,
            max_tokens=max_tokens,
            **kwargs,
        )
        text = "".join(getattr(block, "text", "") for block in response.content)
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return LLMResponse(
            content=text,
            model=chosen_model,
            provider=self.name,
            usage=usage,
        )
