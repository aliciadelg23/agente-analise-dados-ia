"""Cleaning node: applies the default cleaning pipeline to the dataset."""

from __future__ import annotations

from app.agents.graph.state import WorkflowState
from app.core.logging import get_logger
from app.models.cleaning import CleaningOptions
from app.services.cleaning_service import CleaningService

logger = get_logger(__name__)


class CleaningNode:
    """Clean the dataset and publish the cleaned id + report."""

    def __init__(self, service: CleaningService) -> None:
        self._service = service

    def __call__(self, state: WorkflowState) -> WorkflowState:
        dataset_id = state["dataset_id"]
        response = self._service.clean(dataset_id, CleaningOptions())
        logger.info(
            "Cleaning node produced dataset %s -> %s",
            dataset_id,
            response.cleaned_dataset_id,
        )
        return {
            "cleaned_dataset_id": response.cleaned_dataset_id,
            "cleaning_report": response.report.model_dump(mode="json"),
        }
