"""LangChain agent that orchestrates the app's tools.

LangChain 1.x switched to ``create_agent`` returning a compiled
LangGraph state machine. This module wraps that surface with a
small factory the API layer can call.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.core.exceptions import MissingCredentialsError
from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a data-science assistant with access to tools that can "
    "inspect stored datasets, run exploratory analyses, train ML "
    "models, and produce charts. Use the tools to gather concrete "
    "evidence before making claims. Respond concisely and cite the "
    "tools you used."
)


def build_agent(
    tools: list[BaseTool],
    *,
    api_key: str | None,
    model: str,
    temperature: float = 0.0,
) -> Any:
    """Build a LangChain agent wired to ``tools`` and ChatOpenAI.

    Raises ``MissingCredentialsError`` when the OpenAI API key is
    not configured, matching the stage 8 LLM abstraction contract.
    """
    if not api_key:
        raise MissingCredentialsError("OPENAI_API_KEY is required to run the LangChain agent.")
    llm = ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
    return create_agent(model=llm, tools=tools, system_prompt=_SYSTEM_PROMPT)


def run_agent(agent: Any, query: str) -> str:
    """Invoke ``agent`` with ``query`` and return the final answer text."""
    logger.info("Agent query: %s", query[:200])
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = result.get("messages", [])
    if not messages:
        return ""
    final = messages[-1]
    content = getattr(final, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(parts)
    return str(content) if content is not None else ""
