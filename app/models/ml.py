"""Pydantic schemas for the ML training endpoint."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class ProblemType(StrEnum):
    """Supported supervised learning problem types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TrainRequest(BaseModel):
    """Body accepted by POST /datasets/{id}/train."""

    target_column: str = Field(..., min_length=1, description="Name of the target column.")
    problem_type: ProblemType = Field(..., description="'classification' or 'regression'.")
    test_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Proportion of rows held out for testing.",
    )
    cv_folds: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of folds for cross-validation.",
    )


class ModelMetrics(BaseModel):
    """Metrics for a trained model.

    Fields not applicable to the problem type are left null so the
    same schema can be reused for classification and regression.
    """

    accuracy: float | None = Field(default=None, description="Classification accuracy.")
    precision: float | None = Field(default=None, description="Weighted precision.")
    recall: float | None = Field(default=None, description="Weighted recall.")
    f1: float | None = Field(default=None, description="Weighted f1-score.")
    roc_auc: float | None = Field(
        default=None,
        description="Area under the ROC curve (binary classification with predict_proba).",
    )
    r2: float | None = Field(default=None, description="Coefficient of determination.")
    mae: float | None = Field(default=None, description="Mean absolute error.")
    rmse: float | None = Field(default=None, description="Root mean squared error.")


class CandidateResult(BaseModel):
    """Result for a single candidate algorithm."""

    algorithm: str = Field(..., description="Algorithm identifier (snake_case).")
    cv_score_mean: float = Field(..., description="Mean cross-validation score.")
    cv_score_std: float = Field(..., description="Standard deviation of CV scores.")
    test_metrics: ModelMetrics = Field(..., description="Metrics on the held-out test set.")


class TrainResponse(BaseModel):
    """Response returned after a successful training run."""

    dataset_id: UUID = Field(..., description="Source dataset identifier.")
    model_id: UUID = Field(..., description="Server-generated model identifier.")
    problem_type: ProblemType = Field(..., description="Type of problem solved.")
    target_column: str = Field(..., description="Column used as the target.")
    features: list[str] = Field(..., description="Feature columns fed to the pipeline.")
    chosen_algorithm: str = Field(..., description="Winning algorithm by CV mean score.")
    n_samples_train: int = Field(..., ge=0, description="Rows used for training.")
    n_samples_test: int = Field(..., ge=0, description="Rows held out for testing.")
    candidates: list[CandidateResult] = Field(
        ..., description="Every candidate evaluated, in evaluation order."
    )
    best_metrics: ModelMetrics = Field(..., description="Test metrics of the chosen algorithm.")
    model_uri: str = Field(
        ..., description="Filesystem location (relative to storage) of the serialized model."
    )
