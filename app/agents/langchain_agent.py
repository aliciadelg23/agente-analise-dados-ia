"""LangChain agent that orchestrates the app's tools.

LangChain 1.x switched to ``create_agent`` returning a compiled
LangGraph state machine. This module wraps that surface and picks
the underlying chat model according to the requested provider so
callers can talk to OpenAI, Google Gemini or Anthropic through the
same agent.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.core.exceptions import MissingCredentialsError, UnknownProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a data-science assistant with access to tools that can "
    "inspect stored datasets, run exploratory analyses, train ML "
    "models, and produce charts. Use the tools to gather concrete "
    "evidence before making claims. Respond concisely and cite the "
    "tools you used."
)


def _build_chat_model(
    provider: str,
    api_key: str | None,
    model: str,
    temperature: float,
) -> BaseChatModel:
    """Return a LangChain chat model for ``provider``.

    Raises MissingCredentialsError when the API key for the chosen
    provider is not set, and UnknownProviderError for unrecognised
    provider names. Imports the adapter package lazily so unrelated
    providers do not pay the import cost on module load.
    """
    resolved_provider = provider.lower()
    if not api_key:
        raise MissingCredentialsError(
            f"{resolved_provider.upper()}_API_KEY is required to run the LangChain agent."
        )

    if resolved_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)

    if resolved_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=api_key, model=model, temperature=temperature
        )

    if resolved_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(api_key=api_key, model=model, temperature=temperature)

    raise UnknownProviderError(
        f"Unknown LLM provider '{provider}'. Allowed: openai, gemini, anthropic."
    )


def build_agent(
    tools: list[BaseTool],
    *,
    provider: str,
    api_key: str | None,
    model: str,
    temperature: float = 0.0,
) -> Any:
    """Build a LangChain agent wired to ``tools`` and the chosen LLM."""
    llm = _build_chat_model(provider, api_key, model, temperature)
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
