"""Pydantic schemas for the AI insights endpoint."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class InsightsRequest(BaseModel):
    """Body accepted by POST /datasets/{id}/insights."""

    provider: str | None = Field(
        default=None,
        description="Optional provider override (openai, anthropic, gemini).",
    )
    model: str | None = Field(
        default=None,
        description="Optional model override for the chosen provider.",
    )


class DatasetInsightsResponse(BaseModel):
    """Structured AI-generated analysis for a dataset."""

    dataset_id: UUID = Field(..., description="Source dataset identifier.")
    provider: str = Field(..., description="LLM provider that produced the analysis.")
    model: str = Field(..., description="Model identifier that produced the analysis.")
    executive_summary: str = Field(
        ..., description="Short natural-language overview of the dataset."
    )
    insights: list[str] = Field(
        default_factory=list, description="Notable findings about the data."
    )
    anomalies: list[str] = Field(
        default_factory=list, description="Suspicious rows, columns, or values."
    )
    suggestions: list[str] = Field(default_factory=list, description="Actionable next steps.")
    risks: list[str] = Field(
        default_factory=list, description="Data-quality or modeling risks flagged by the model."
    )
    raw_llm_response: str | None = Field(
        default=None,
        description="Raw LLM output; populated only when the JSON payload could not be parsed.",
    )
