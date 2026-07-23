"""Pydantic schemas for the analysis workflow endpoint."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class WorkflowAnalyzeRequest(BaseModel):
    """Body accepted by POST /workflows/analyze."""

    dataset_id: UUID = Field(..., description="Source dataset identifier.")
    user_query: str | None = Field(
        default=None,
        description="Optional natural-language description of what the caller wants.",
    )
    target_column: str | None = Field(
        default=None,
        description="Target column when the workflow should train an ML model.",
    )
    problem_type: str | None = Field(
        default=None,
        description="'classification' or 'regression' when the workflow should train.",
    )
    run_ml: bool = Field(
        default=True,
        description="Whether to run the ML step; combined with target_column to decide.",
    )
    llm_provider: str | None = Field(
        default=None,
        description="Optional override for the LLM provider used by Insight and Report nodes.",
    )


class WorkflowAnalyzeResponse(BaseModel):
    """Response returned by POST /workflows/analyze."""

    dataset_id: UUID
    cleaned_dataset_id: UUID | None = None
    plan: dict[str, Any] = Field(default_factory=dict)
    eda_summary: dict[str, Any] = Field(default_factory=dict)
    cleaning_report: dict[str, Any] = Field(default_factory=dict)
    ml_result: dict[str, Any] | None = None
    insights: dict[str, Any] = Field(default_factory=dict)
    final_report: str = ""
