"""Insight node: produces AI-authored insights via the LLM layer."""

from __future__ import annotations

from app.agents.graph.state import WorkflowState
from app.core.logging import get_logger
from app.llms.base import LLMProvider
from app.services.ai_insight_service import AIInsightService

logger = get_logger(__name__)


class InsightNode:
    """Ask the LLM for structured insights about the cleaned dataset."""

    def __init__(self, service: AIInsightService, llm_provider: LLMProvider) -> None:
        self._service = service
        self._llm = llm_provider

    def __call__(self, state: WorkflowState) -> WorkflowState:
        dataset_id = state.get("cleaned_dataset_id") or state["dataset_id"]
        response = self._service.analyze(dataset_id, self._llm)
        logger.info(
            "Insight node produced %d insights via provider=%s",
            len(response.insights),
            response.provider,
        )
        return {"insights": response.model_dump(mode="json")}
