"""Unit tests for MLPipelineService."""

from __future__ import annotations

import random
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import (
    DatasetNotFoundError,
    InsufficientDataError,
    InvalidTargetColumnError,
)
from app.models.ml import ProblemType, TrainRequest
from app.repositories.dataset_repository import DatasetRepository
from app.services.ml_pipeline_service import MLPipelineService


def _store(repository: DatasetRepository, content: str) -> UUID:
    dataset_id = uuid4()
    repository.save(dataset_id, "sample.csv", content.encode("utf-8"))
    return dataset_id


def _synthetic_classification_csv(rows: int = 200, seed: int = 42) -> str:
    rng = random.Random(seed)
    lines = ["age,salary,city,churn"]
    for _ in range(rows):
        age = rng.randint(18, 70)
        salary = rng.randint(1000, 10000)
        city = rng.choice(["Lisbon", "Porto", "Braga"])
        churn = 1 if (age > 50 or salary < 2000) else 0
        lines.append(f"{age},{salary},{city},{churn}")
    return "\n".join(lines) + "\n"


def _synthetic_regression_csv(rows: int = 200, seed: int = 42) -> str:
    rng = random.Random(seed)
    lines = ["x1,x2,category,y"]
    for _ in range(rows):
        x1 = rng.uniform(0, 10)
        x2 = rng.uniform(0, 5)
        cat = rng.choice(["a", "b", "c"])
        y = 3 * x1 + 2 * x2 + rng.gauss(0, 0.5)
        lines.append(f"{x1:.3f},{x2:.3f},{cat},{y:.3f}")
    return "\n".join(lines) + "\n"


def test_train_classification_picks_best_of_three(
    dataset_repository: DatasetRepository, ml_pipeline_service: MLPipelineService
) -> None:
    dataset_id = _store(dataset_repository, _synthetic_classification_csv())

    response = ml_pipeline_service.train(
        dataset_id,
        TrainRequest(target_column="churn", problem_type=ProblemType.CLASSIFICATION),
    )

    assert response.problem_type == ProblemType.CLASSIFICATION
    assert response.target_column == "churn"
    assert set(response.features) == {"age", "salary", "city"}
    assert len(response.candidates) == 3
    algorithms = {c.algorithm for c in response.candidates}
    assert algorithms == {"logistic_regression", "decision_tree", "random_forest"}
    assert response.chosen_algorithm in algorithms
    assert response.best_metrics.accuracy is not None
    assert response.best_metrics.f1 is not None
    assert response.best_metrics.roc_auc is not None  # binary target with predict_proba


def test_train_regression_returns_regression_metrics(
    dataset_repository: DatasetRepository, ml_pipeline_service: MLPipelineService
) -> None:
    dataset_id = _store(dataset_repository, _synthetic_regression_csv())

    response = ml_pipeline_service.train(
        dataset_id,
        TrainRequest(target_column="y", problem_type=ProblemType.REGRESSION),
    )

    assert response.problem_type == ProblemType.REGRESSION
    algorithms = {c.algorithm for c in response.candidates}
    assert algorithms == {"linear_regression", "random_forest"}
    assert response.best_metrics.r2 is not None
    assert response.best_metrics.mae is not None
    assert response.best_metrics.rmse is not None
    assert response.best_metrics.accuracy is None


def test_train_persists_model_artifact_and_manifest(
    dataset_repository: DatasetRepository,
    ml_pipeline_service: MLPipelineService,
    temp_storage: Path,
) -> None:
    dataset_id = _store(dataset_repository, _synthetic_classification_csv())

    response = ml_pipeline_service.train(
        dataset_id,
        TrainRequest(target_column="churn", problem_type=ProblemType.CLASSIFICATION),
    )

    models_dir = temp_storage / "models"
    assert (models_dir / f"{response.model_id}.joblib").exists()
    assert (models_dir / f"{response.model_id}.json").exists()


def test_train_raises_for_missing_target_column(
    dataset_repository: DatasetRepository, ml_pipeline_service: MLPipelineService
) -> None:
    dataset_id = _store(dataset_repository, _synthetic_classification_csv())

    with pytest.raises(InvalidTargetColumnError):
        ml_pipeline_service.train(
            dataset_id,
            TrainRequest(target_column="not_a_column", problem_type=ProblemType.CLASSIFICATION),
        )


def test_train_raises_for_single_class_target(
    dataset_repository: DatasetRepository, ml_pipeline_service: MLPipelineService
) -> None:
    csv = "x,y\n" + "\n".join(f"{i},A" for i in range(30)) + "\n"
    dataset_id = _store(dataset_repository, csv)

    with pytest.raises(InsufficientDataError):
        ml_pipeline_service.train(
            dataset_id,
            TrainRequest(target_column="y", problem_type=ProblemType.CLASSIFICATION),
        )


def test_train_raises_for_insufficient_rows(
    dataset_repository: DatasetRepository, ml_pipeline_service: MLPipelineService
) -> None:
    dataset_id = _store(dataset_repository, "x,y\n1,A\n2,B\n3,A\n")

    with pytest.raises(InsufficientDataError):
        ml_pipeline_service.train(
            dataset_id,
            TrainRequest(target_column="y", problem_type=ProblemType.CLASSIFICATION),
        )


def test_train_raises_for_unknown_dataset(ml_pipeline_service: MLPipelineService) -> None:
    with pytest.raises(DatasetNotFoundError):
        ml_pipeline_service.train(
            uuid4(),
            TrainRequest(target_column="y", problem_type=ProblemType.CLASSIFICATION),
        )
