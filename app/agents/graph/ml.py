"""ML node: trains a model when the plan asks for it, otherwise skips."""

from __future__ import annotations

from app.agents.graph.state import WorkflowState
from app.core.logging import get_logger
from app.models.ml import ProblemType, TrainRequest
from app.services.ml_pipeline_service import MLPipelineService

logger = get_logger(__name__)


class MLNode:
    """Train ML candidates on the cleaned dataset when the plan requests it."""

    def __init__(self, service: MLPipelineService) -> None:
        self._service = service

    def __call__(self, state: WorkflowState) -> WorkflowState:
        plan = state.get("plan") or {}
        if not plan.get("should_train"):
            logger.info("ML node skipped: plan.should_train is false")
            return {"ml_result": None}

        target = plan.get("target_column")
        problem_type = plan.get("problem_type")
        if not target or not problem_type:
            logger.info("ML node skipped: missing target_column or problem_type")
            return {"ml_result": None}

        dataset_id = state.get("cleaned_dataset_id") or state["dataset_id"]
        request = TrainRequest(
            target_column=target, problem_type=ProblemType(problem_type)
        )
        response = self._service.train(dataset_id, request)
        logger.info(
            "ML node trained on %s (chosen=%s)", dataset_id, response.chosen_algorithm
        )
        return {"ml_result": response.model_dump(mode="json")}
