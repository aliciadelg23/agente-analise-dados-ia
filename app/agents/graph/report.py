"""Report node: synthesizes the full workflow into a final report."""

from __future__ import annotations

import json

from app.agents.graph.state import WorkflowState
from app.core.logging import get_logger
from app.llms.base import LLMProvider, Message
from app.services.vector_index_service import VectorIndexService

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior data analyst preparing a written report. You will "
    "receive a JSON payload with EDA, cleaning, ML and insight results "
    "for a single dataset. Return a concise markdown report with the "
    "following sections in order: Overview, EDA highlights, Cleaning, "
    "ML results, Insights, Recommendations. Do not invent numbers; "
    "cite the values from the payload."
)


class ReportNode:
    """Ask the LLM to compose a markdown report from the collected state."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        vector_index: VectorIndexService | None = None,
    ) -> None:
        self._llm = llm_provider
        self._vector_index = vector_index

    def __call__(self, state: WorkflowState) -> WorkflowState:
        payload = {
            "dataset_id": str(state["dataset_id"]),
            "cleaned_dataset_id": str(state.get("cleaned_dataset_id") or ""),
            "plan": state.get("plan"),
            "eda_summary": state.get("eda_summary"),
            "cleaning_report": state.get("cleaning_report"),
            "ml_result": state.get("ml_result"),
            "insights": state.get("insights"),
            "user_query": state.get("user_query"),
        }
        content = json.dumps(payload, default=str, ensure_ascii=False)
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=f"WORKFLOW_STATE_JSON:\n{content}"),
        ]
        response = self._llm.chat(messages)
        logger.info(
            "Report node received %d chars from provider=%s",
            len(response.content),
            response.provider,
        )
        if self._vector_index is not None:
            self._vector_index.index_report(state["dataset_id"], response.content)
        return {"final_report": response.content}
