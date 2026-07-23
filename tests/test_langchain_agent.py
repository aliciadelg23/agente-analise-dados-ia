"""Tests for the LangChain agent factory."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from app.agents.langchain_agent import build_agent, run_agent
from app.core.exceptions import MissingCredentialsError


class _StubTool(BaseTool):
    name: str = "stub"
    description: str = "A tool used only in tests."

    def _run(self, *args: object, **kwargs: object) -> str:
        return "ok"


def test_build_agent_requires_api_key() -> None:
    with pytest.raises(MissingCredentialsError):
        build_agent([_StubTool()], api_key=None, model="gpt-4o-mini")


def test_build_agent_creates_agent_with_expected_wiring() -> None:
    with (
        patch("app.agents.langchain_agent.ChatOpenAI") as mocked_chat,
        patch("app.agents.langchain_agent.create_agent") as mocked_create,
    ):
        llm_instance = MagicMock()
        mocked_chat.return_value = llm_instance
        mocked_create.return_value = MagicMock()
        tools = [_StubTool()]

        build_agent(tools, api_key="sk-test", model="gpt-4o-mini")

        mocked_chat.assert_called_once_with(api_key="sk-test", model="gpt-4o-mini", temperature=0.0)
        mocked_create.assert_called_once()
        kwargs = mocked_create.call_args.kwargs
        assert kwargs["model"] is llm_instance
        assert kwargs["tools"] is tools
        assert "system_prompt" in kwargs


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
