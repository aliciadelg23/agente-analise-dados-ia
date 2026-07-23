"""Insight node: produces AI-authored insights via the LLM layer."""

from __future__ import annotations

from app.agents.graph.state import WorkflowState
from app.core.logging import get_logger
from app.llms.base import LLMProvider
from app.services.ai_insight_service import AIInsightService
from app.services.vector_index_service import VectorIndexService

logger = get_logger(__name__)


class InsightNode:
    """Ask the LLM for structured insights about the cleaned dataset."""

    def __init__(
        self,
        service: AIInsightService,
        llm_provider: LLMProvider,
        vector_index: VectorIndexService | None = None,
    ) -> None:
        self._service = service
        self._llm = llm_provider
        self._vector_index = vector_index

    def __call__(self, state: WorkflowState) -> WorkflowState:
        dataset_id = state.get("cleaned_dataset_id") or state["dataset_id"]
        response = self._service.analyze(dataset_id, self._llm)
        payload = response.model_dump(mode="json")
        logger.info(
            "Insight node produced %d insights via provider=%s",
            len(response.insights),
            response.provider,
        )
        if self._vector_index is not None:
            self._vector_index.index_insights(dataset_id, payload)
        return {"insights": payload}
