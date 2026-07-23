"""LangChain agent that orchestrates the app's tools."""

from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
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


def build_agent_executor(
    tools: list[BaseTool],
    *,
    api_key: str | None,
    model: str,
    temperature: float = 0.0,
    max_iterations: int = 6,
    verbose: bool = False,
) -> AgentExecutor:
    """Build an ``AgentExecutor`` wired to ``tools`` and ChatOpenAI.

    Raises ``MissingCredentialsError`` at construction time when the
    OpenAI API key is not configured, matching the behavior of the
    LLM abstraction from stage 8.
    """
    if not api_key:
        raise MissingCredentialsError(
            "OPENAI_API_KEY is required to run the LangChain agent."
        )

    llm = ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        return_intermediate_steps=False,
    )


def run_agent(executor: AgentExecutor, query: str) -> dict[str, Any]:
    """Invoke ``executor`` with ``query`` and return the raw result dict."""
    logger.info("Agent query: %s", query[:200])
    return executor.invoke({"input": query})
