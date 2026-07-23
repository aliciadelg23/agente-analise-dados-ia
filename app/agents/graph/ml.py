"""ML node: trains a model when the plan asks for it, otherwise skips."""

from __future__ import annotations

from app.agents.graph.state import WorkflowState
from app.core.logging import get_logger
from app.models.ml import ProblemType, TrainRequest
from app.services.ml_pipeline_service import MLPipelineService
from app.services.vector_index_service import VectorIndexService

logger = get_logger(__name__)


class MLNode:
    """Train ML candidates on the cleaned dataset when the plan requests it."""

    def __init__(
        self,
        service: MLPipelineService,
        vector_index: VectorIndexService | None = None,
    ) -> None:
        self._service = service
        self._vector_index = vector_index

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
        request = TrainRequest(target_column=target, problem_type=ProblemType(problem_type))
        response = self._service.train(dataset_id, request)
        payload = response.model_dump(mode="json")
        logger.info("ML node trained on %s (chosen=%s)", dataset_id, response.chosen_algorithm)
        if self._vector_index is not None:
            manifest = {
                "model_id": str(response.model_id),
                "dataset_id": str(response.dataset_id),
                "problem_type": response.problem_type.value,
                "target_column": response.target_column,
                "features": response.features,
                "chosen_algorithm": response.chosen_algorithm,
            }
            self._vector_index.index_model(response.model_id, manifest)
        return {"ml_result": payload}
