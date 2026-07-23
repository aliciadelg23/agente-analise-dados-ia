"""Shared state passed between LangGraph nodes.

Each node receives the current ``WorkflowState`` and returns a
partial dict with just the fields it wants to update. LangGraph
merges the update into the state before invoking the next node.
"""

from __future__ import annotations

from typing import Any, TypedDict
from uuid import UUID


class WorkflowPlan(TypedDict, total=False):
    """Plan produced by the planner agent."""

    should_train: bool
    target_column: str | None
    problem_type: str | None
    notes: str


class WorkflowState(TypedDict, total=False):
    """State carried through every node of the analysis workflow.

    ``total=False`` means fields are populated progressively as the
    graph runs; nodes read what they need and produce what they own.
    """

    dataset_id: UUID
    user_query: str | None
    plan: WorkflowPlan
    eda_summary: dict[str, Any]
    cleaned_dataset_id: UUID
    cleaning_report: dict[str, Any]
    ml_result: dict[str, Any] | None
    insights: dict[str, Any]
    final_report: str
