"""Unit tests for ExplainabilityService."""

from __future__ import annotations

import random
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import DatasetNotFoundError, ModelNotFoundError
from app.models.ml import ProblemType, TrainRequest
from app.repositories.dataset_repository import DatasetRepository
from app.services.explainability_service import ExplainabilityService
from app.services.ml_pipeline_service import MLPipelineService


def _synthetic_csv(seed: int = 42, rows: int = 200) -> bytes:
    rng = random.Random(seed)
    lines = ["age,salary,city,churn"]
    for _ in range(rows):
        age = rng.randint(18, 70)
        salary = rng.randint(1000, 10000)
        city = rng.choice(["Lisbon", "Porto", "Braga"])
        churn = 1 if (age > 50 or salary < 2000) else 0
        lines.append(f"{age},{salary},{city},{churn}")
    return ("\n".join(lines) + "\n").encode()


def _store(repository: DatasetRepository) -> UUID:
    dataset_id = uuid4()
    repository.save(dataset_id, "sample.csv", _synthetic_csv())
    return dataset_id


def test_explain_returns_full_payload_for_trained_model(
    dataset_repository: DatasetRepository,
    ml_pipeline_service: MLPipelineService,
    explainability_service: ExplainabilityService,
) -> None:
    dataset_id = _store(dataset_repository)
    trained = ml_pipeline_service.train(
        dataset_id, TrainRequest(target_column="churn", problem_type=ProblemType.CLASSIFICATION)
    )

    result = explainability_service.explain(trained.model_id)

    assert result.model_id == trained.model_id
    assert result.dataset_id == dataset_id
    assert result.target_column == "churn"
    assert result.algorithm == trained.chosen_algorithm
    assert result.problem_type == "classification"
    assert len(result.feature_importance) > 0
    assert len(result.shap.mean_abs_values) > 0
    assert result.shap.chart_url.endswith("shap_summary.png")
    assert result.top_features
    assert result.narrative.startswith("The model")


def test_explain_persists_summary_plot(
    dataset_repository: DatasetRepository,
    ml_pipeline_service: MLPipelineService,
    explainability_service: ExplainabilityService,
    temp_storage: Path,
) -> None:
    dataset_id = _store(dataset_repository)
    trained = ml_pipeline_service.train(
        dataset_id, TrainRequest(target_column="churn", problem_type=ProblemType.CLASSIFICATION)
    )

    explainability_service.explain(trained.model_id)

    expected_path = temp_storage / "charts" / "models" / str(trained.model_id) / "shap_summary.png"
    assert expected_path.exists()
    assert expected_path.stat().st_size > 0


def test_explain_supports_regression_models(
    dataset_repository: DatasetRepository,
    ml_pipeline_service: MLPipelineService,
    explainability_service: ExplainabilityService,
) -> None:
    rng = random.Random(7)
    lines = ["x1,x2,category,y"]
    for _ in range(150):
        x1 = rng.uniform(0, 10)
        x2 = rng.uniform(0, 5)
        cat = rng.choice(["a", "b", "c"])
        y = 3 * x1 + 2 * x2 + rng.gauss(0, 0.5)
        lines.append(f"{x1:.3f},{x2:.3f},{cat},{y:.3f}")
    dataset_id = uuid4()
    dataset_repository.save(dataset_id, "reg.csv", ("\n".join(lines) + "\n").encode())

    trained = ml_pipeline_service.train(
        dataset_id, TrainRequest(target_column="y", problem_type=ProblemType.REGRESSION)
    )

    result = explainability_service.explain(trained.model_id)

    assert result.problem_type == "regression"
    assert len(result.feature_importance) > 0
    assert len(result.shap.mean_abs_values) > 0


def test_explain_raises_when_model_missing(
    explainability_service: ExplainabilityService,
) -> None:
    with pytest.raises(ModelNotFoundError):
        explainability_service.explain(uuid4())


def test_explain_raises_when_dataset_missing(
    dataset_repository: DatasetRepository,
    ml_pipeline_service: MLPipelineService,
    explainability_service: ExplainabilityService,
) -> None:
    dataset_id = _store(dataset_repository)
    trained = ml_pipeline_service.train(
        dataset_id, TrainRequest(target_column="churn", problem_type=ProblemType.CLASSIFICATION)
    )

    original = dataset_repository.find(dataset_id)
    assert original is not None
    original.rename(original.parent / "moved_away.txt")

    with pytest.raises(DatasetNotFoundError):
        explainability_service.explain(trained.model_id)
