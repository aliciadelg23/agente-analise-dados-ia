"""Planner node: turns the incoming request into a WorkflowPlan.

Kept deterministic (no LLM) so the graph starts from a predictable
state and tests do not require mocking. Consumers can supply target
column and problem type explicitly; when they are missing, the node
defaults to skipping the ML step rather than guessing.
"""

from __future__ import annotations

from app.agents.graph.state import WorkflowPlan, WorkflowState
from app.core.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_PROBLEM_TYPES = {"classification", "regression"}


class PlannerAgent:
    """Produce a WorkflowPlan from the request-carrying state."""

    def __call__(self, state: WorkflowState) -> WorkflowState:
        overrides = state.get("plan") or {}
        target = overrides.get("target_column")
        problem_type = overrides.get("problem_type")
        should_train = bool(overrides.get("should_train", True))

        notes_parts: list[str] = []
        if problem_type is not None and problem_type not in _SUPPORTED_PROBLEM_TYPES:
            notes_parts.append(
                f"Unsupported problem_type '{problem_type}'; skipping ML."
            )
            problem_type = None
            should_train = False

        if should_train and not target:
            notes_parts.append("No target_column provided; skipping ML step.")
            should_train = False

        if should_train and not problem_type:
            notes_parts.append(
                "No problem_type provided; defaulting to 'classification'."
            )
            problem_type = "classification"

        plan: WorkflowPlan = {
            "should_train": should_train,
            "target_column": target,
            "problem_type": problem_type,
            "notes": " ".join(notes_parts) if notes_parts else "Plan ready.",
        }
        logger.info(
            "Planner produced plan: should_train=%s target=%s type=%s",
            should_train,
            target,
            problem_type,
        )
        return {"plan": plan}
