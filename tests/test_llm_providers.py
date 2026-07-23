"""Unit tests for the LLM providers.

Every provider is exercised with its SDK client fully mocked so
tests never touch the network or need real credentials.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import MissingCredentialsError
from app.llms.anthropic_provider import AnthropicProvider
from app.llms.base import Message
from app.llms.gemini_provider import GeminiProvider
from app.llms.openai_provider import OpenAIProvider


class TestOpenAIProvider:
    def test_requires_api_key(self) -> None:
        with pytest.raises(MissingCredentialsError):
            OpenAIProvider(api_key=None)

    def test_translates_messages_and_returns_llm_response(self) -> None:
        with patch("app.llms.openai_provider.OpenAI") as mocked_openai:
            usage = SimpleNamespace(model_dump=lambda: {"total_tokens": 12})
            response = SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="Hello, world!")),
                ],
                usage=usage,
            )
            client = MagicMock()
            client.chat.completions.create.return_value = response
            mocked_openai.return_value = client

            provider = OpenAIProvider(api_key="sk-test", default_model="gpt-4o-mini")
            result = provider.chat([Message(role="user", content="Hi")], model="gpt-4o-mini")

            client.chat.completions.create.assert_called_once()
            call_kwargs = client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o-mini"
            assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]
            assert result.content == "Hello, world!"
            assert result.provider == "openai"
            assert result.model == "gpt-4o-mini"
            assert result.usage == {"total_tokens": 12}


class TestAnthropicProvider:
    def test_requires_api_key(self) -> None:
        with pytest.raises(MissingCredentialsError):
            AnthropicProvider(api_key=None)

    def test_extracts_system_prompt_and_returns_llm_response(self) -> None:
        with patch("app.llms.anthropic_provider.Anthropic") as mocked_anthropic:
            response = SimpleNamespace(
                content=[SimpleNamespace(text="Sure thing.")],
                usage=SimpleNamespace(input_tokens=5, output_tokens=3),
            )
            client = MagicMock()
            client.messages.create.return_value = response
            mocked_anthropic.return_value = client

            provider = AnthropicProvider(
                api_key="secret",
                default_model="claude-haiku-4-5-20251001",
            )
            result = provider.chat(
                [
                    Message(role="system", content="Be terse."),
                    Message(role="user", content="Ping"),
                ]
            )

            client.messages.create.assert_called_once()
            call_kwargs = client.messages.create.call_args.kwargs
            assert call_kwargs["system"] == "Be terse."
            assert call_kwargs["messages"] == [{"role": "user", "content": "Ping"}]
            assert call_kwargs["max_tokens"] == 1024
            assert result.content == "Sure thing."
            assert result.provider == "anthropic"
            assert result.usage == {"input_tokens": 5, "output_tokens": 3}


class TestGeminiProvider:
    def test_requires_api_key(self) -> None:
        with pytest.raises(MissingCredentialsError):
            GeminiProvider(api_key=None)

    def test_prepends_system_prompt_into_last_message(self) -> None:
        with (
            patch("app.llms.gemini_provider.genai.configure"),
            patch("app.llms.gemini_provider.genai.GenerativeModel") as mocked_model,
        ):
            chat_session = MagicMock()
            chat_session.send_message.return_value = SimpleNamespace(text="answer")
            model_instance = MagicMock()
            model_instance.start_chat.return_value = chat_session
            mocked_model.return_value = model_instance

            provider = GeminiProvider(api_key="key", default_model="gemini-2.5-flash")
            result = provider.chat(
                [
                    Message(role="system", content="Reply in English."),
                    Message(role="user", content="Bonjour"),
                ]
            )

            model_instance.start_chat.assert_called_once()
            chat_session.send_message.assert_called_once()
            sent_content = chat_session.send_message.call_args.args[0]
            assert "Reply in English." in sent_content
            assert "Bonjour" in sent_content
            assert result.content == "answer"
            assert result.provider == "gemini"
            assert result.model == "gemini-2.5-flash"
