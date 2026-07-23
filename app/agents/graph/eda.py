"""EDA node: delegates to EDAService and stores the summary in state."""

from __future__ import annotations

from app.agents.graph.state import WorkflowState
from app.core.logging import get_logger
from app.services.eda_service import EDAService
from app.services.vector_index_service import VectorIndexService

logger = get_logger(__name__)


class EDANode:
    """Produce the descriptive summary of the source dataset."""

    def __init__(
        self,
        service: EDAService,
        vector_index: VectorIndexService | None = None,
    ) -> None:
        self._service = service
        self._vector_index = vector_index

    def __call__(self, state: WorkflowState) -> WorkflowState:
        dataset_id = state["dataset_id"]
        summary = self._service.summarize(dataset_id)
        payload = summary.model_dump(mode="json")
        logger.info(
            "EDA node summarized dataset %s (rows=%d, columns=%d)",
            dataset_id,
            summary.rows,
            summary.columns,
        )
        if self._vector_index is not None:
            self._vector_index.index_eda(dataset_id, payload)
        return {"eda_summary": payload}
