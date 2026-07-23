"""Tests for the LangChain agent factory."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from app.agents.langchain_agent import build_agent, run_agent
from app.core.exceptions import MissingCredentialsError, UnknownProviderError


class _StubTool(BaseTool):
    name: str = "stub"
    description: str = "A tool used only in tests."

    def _run(self, *args: object, **kwargs: object) -> str:
        return "ok"


def test_build_agent_requires_api_key() -> None:
    with pytest.raises(MissingCredentialsError):
        build_agent([_StubTool()], provider="openai", api_key=None, model="gpt-4o-mini")


def test_build_agent_rejects_unknown_provider() -> None:
    with pytest.raises(UnknownProviderError):
        build_agent([_StubTool()], provider="cohere", api_key="secret", model="whatever")


def test_build_agent_creates_openai_chat_model() -> None:
    with (
        patch("langchain_openai.ChatOpenAI") as mocked_chat,
        patch("app.agents.langchain_agent.create_agent") as mocked_create,
    ):
        mocked_chat.return_value = MagicMock()
        mocked_create.return_value = MagicMock()
        tools = [_StubTool()]

        build_agent(tools, provider="openai", api_key="sk-test", model="gpt-4o-mini")

        mocked_chat.assert_called_once_with(api_key="sk-test", model="gpt-4o-mini", temperature=0.0)
        mocked_create.assert_called_once()
        kwargs = mocked_create.call_args.kwargs
        assert kwargs["tools"] is tools
        assert "system_prompt" in kwargs


def test_build_agent_creates_gemini_chat_model() -> None:
    with (
        patch("langchain_google_genai.ChatGoogleGenerativeAI") as mocked_chat,
        patch("app.agents.langchain_agent.create_agent") as mocked_create,
    ):
        mocked_chat.return_value = MagicMock()
        mocked_create.return_value = MagicMock()

        build_agent(
            [_StubTool()],
            provider="gemini",
            api_key="google-key",
            model="gemini-2.0-flash-exp",
        )

        mocked_chat.assert_called_once_with(
            google_api_key="google-key",
            model="gemini-2.0-flash-exp",
            temperature=0.0,
        )
        mocked_create.assert_called_once()


def test_build_agent_creates_anthropic_chat_model() -> None:
    with (
        patch("langchain_anthropic.ChatAnthropic") as mocked_chat,
        patch("app.agents.langchain_agent.create_agent") as mocked_create,
    ):
        mocked_chat.return_value = MagicMock()
        mocked_create.return_value = MagicMock()

        build_agent(
            [_StubTool()],
            provider="anthropic",
            api_key="anthropic-key",
            model="claude-haiku-4-5-20251001",
        )

        mocked_chat.assert_called_once_with(
            api_key="anthropic-key",
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
        )
        mocked_create.assert_called_once()


def test_run_agent_returns_final_message_content() -> None:
    final_message = SimpleNamespace(content="Here is the answer.")
    agent = MagicMock()
    agent.invoke.return_value = {"messages": [final_message]}

    result = run_agent(agent, "hi")

    agent.invoke.assert_called_once_with({"messages": [{"role": "user", "content": "hi"}]})
    assert result == "Here is the answer."


def test_run_agent_handles_list_content() -> None:
    final_message = SimpleNamespace(content=[{"text": "part 1"}, {"text": "part 2"}])
    agent = MagicMock()
    agent.invoke.return_value = {"messages": [final_message]}

    result = run_agent(agent, "hi")

    assert result == "part 1part 2"


def test_run_agent_returns_empty_string_when_no_messages() -> None:
    agent = MagicMock()
    agent.invoke.return_value = {"messages": []}

    result = run_agent(agent, "hi")

    assert result == ""
