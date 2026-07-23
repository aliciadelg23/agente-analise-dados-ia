"""Google Gemini adapter for the LLM abstraction."""

from __future__ import annotations

from typing import Any

import google.generativeai as genai

from app.core.exceptions import MissingCredentialsError
from app.llms.base import LLMProvider, LLMResponse, Message


class GeminiProvider(LLMProvider):
    """Adapter over the google-generativeai SDK.

    Gemini has no dedicated system role; we prepend any system
    message to the first user message so the instruction still
    reaches the model.
    """

    name = "gemini"

    def __init__(self, api_key: str | None, default_model: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise MissingCredentialsError("GEMINI_API_KEY is not configured.")
        genai.configure(api_key=api_key)
        self.default_model = default_model

    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        chosen_model = model or self.default_model
        system_prompts = [message.content for message in messages if message.role == "system"]
        remaining = [message for message in messages if message.role != "system"]

        if not remaining:
            raise ValueError("At least one non-system message is required.")

        history: list[dict[str, Any]] = []
        for message in remaining[:-1]:
            history.append(
                {
                    "role": "user" if message.role == "user" else "model",
                    "parts": [message.content],
                }
            )

        latest = remaining[-1]
        prompt_prefix = "\n\n".join(system_prompts)
        latest_content = (
            f"{prompt_prefix}\n\n{latest.content}" if prompt_prefix else latest.content
        )

        client = genai.GenerativeModel(chosen_model)
        chat_session = client.start_chat(history=history)
        response = chat_session.send_message(latest_content, **kwargs)

        return LLMResponse(
            content=getattr(response, "text", "") or "",
            model=chosen_model,
            provider=self.name,
        )
