"""EDA node: delegates to EDAService and stores the summary in state."""

from __future__ import annotations

from app.agents.graph.state import WorkflowState
from app.core.logging import get_logger
from app.services.eda_service import EDAService

logger = get_logger(__name__)


class EDANode:
    """Produce the descriptive summary of the source dataset."""

    def __init__(self, service: EDAService) -> None:
        self._service = service

    def __call__(self, state: WorkflowState) -> WorkflowState:
        dataset_id = state["dataset_id"]
        summary = self._service.summarize(dataset_id)
        logger.info(
            "EDA node summarized dataset %s (rows=%d, columns=%d)",
            dataset_id,
            summary.rows,
            summary.columns,
        )
        return {"eda_summary": summary.model_dump(mode="json")}
