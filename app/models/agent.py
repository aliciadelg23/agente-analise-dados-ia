"""Pydantic schemas for the LangChain agent endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    """Body accepted by POST /agent/chat."""

    query: str = Field(..., min_length=1, description="Natural-language question for the agent.")
    model: str | None = Field(
        default=None,
        description="Optional model override; defaults to OPENAI_MODEL from settings.",
    )


class AgentChatResponse(BaseModel):
    """Response returned by POST /agent/chat."""

    output: str = Field(..., description="Final answer produced by the agent.")
    model: str = Field(..., description="Model identifier that produced the answer.")
