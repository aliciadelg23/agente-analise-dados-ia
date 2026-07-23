"""ML training pipeline service.

Orchestrates dataset loading, preprocessing, cross-validated model
selection, and persistence of the winning pipeline. Kept independent
of FastAPI so it can be reused from CLIs or workers.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.core.exceptions import (
    DatasetNotFoundError,
    InsufficientDataError,
    InvalidTargetColumnError,
)
from app.core.logging import get_logger
from app.models.ml import (
    CandidateResult,
    ModelMetrics,
    ProblemType,
    TrainRequest,
    TrainResponse,
)
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.model_repository import ModelRepository
from app.utils.csv_inspector import detect_encoding, detect_separator

logger = get_logger(__name__)

_MIN_ROWS = 10
_MAX_CATEGORICAL_UNIQUE = 50
_MAX_NULL_RATIO = 0.9


_CLASSIFIERS: dict[str, Any] = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

_REGRESSORS: dict[str, Any] = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}


class MLPipelineService:
    """Train candidate models on a stored dataset and persist the winner."""

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        model_repository: ModelRepository,
    ) -> None:
        self._datasets = dataset_repository
        self._models = model_repository

    def train(self, dataset_id: UUID, request: TrainRequest) -> TrainResponse:
        """Run the full training pipeline and return the result."""
        path = self._datasets.find(dataset_id)
        if path is None:
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' was not found.")

        frame = self._load_dataframe(path)

        if request.target_column not in frame.columns:
            raise InvalidTargetColumnError(
                f"Target column '{request.target_column}' does not exist in the dataset."
            )

        y = frame[request.target_column]
        x_raw = frame.drop(columns=[request.target_column])
        x, feature_columns = self._select_features(x_raw)

        y = y.dropna()
        x = x.loc[y.index]

        if len(x) < _MIN_ROWS:
            raise InsufficientDataError(
                f"Dataset has {len(x)} usable rows; at least {_MIN_ROWS} are required."
            )

        if request.problem_type == ProblemType.CLASSIFICATION and y.nunique() < 2:
            raise InsufficientDataError("Classification requires at least 2 distinct classes.")

        numeric_cols = x.select_dtypes(include="number").columns.tolist()
        categorical_cols = [c for c in x.columns if c not in numeric_cols]
        preprocessor = self._build_preprocessor(numeric_cols, categorical_cols)

        stratify = y if request.problem_type == ProblemType.CLASSIFICATION else None
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=request.test_size, random_state=42, stratify=stratify
            )
        except ValueError:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=request.test_size, random_state=42
            )

        candidates_registry = (
            _CLASSIFIERS if request.problem_type == ProblemType.CLASSIFICATION else _REGRESSORS
        )
        scoring = "f1_weighted" if request.problem_type == ProblemType.CLASSIFICATION else "r2"

        candidates: list[CandidateResult] = []
        best: tuple[str, Pipeline, CandidateResult] | None = None

        for name, estimator in candidates_registry.items():
            pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    pipeline, x_train, y_train, cv=request.cv_folds, scoring=scoring, n_jobs=-1
                )
                pipeline.fit(x_train, y_train)

            test_metrics = self._compute_metrics(pipeline, x_test, y_test, request.problem_type)
            candidate = CandidateResult(
                algorithm=name,
                cv_score_mean=float(np.mean(cv_scores)),
                cv_score_std=float(np.std(cv_scores)),
                test_metrics=test_metrics,
            )
            candidates.append(candidate)

            if best is None or candidate.cv_score_mean > best[2].cv_score_mean:
                best = (name, pipeline, candidate)

        assert best is not None
        best_name, best_pipeline, best_result = best
        model_id = uuid4()
        manifest = {
            "model_id": str(model_id),
            "dataset_id": str(dataset_id),
            "problem_type": request.problem_type.value,
            "target_column": request.target_column,
            "features": feature_columns,
            "chosen_algorithm": best_name,
            "cv_score_mean": best_result.cv_score_mean,
        }
        artifact_path = self._models.save(model_id, best_pipeline, manifest)
        logger.info(
            "Trained %s on dataset %s (winner=%s, cv=%.3f)",
            request.problem_type.value,
            dataset_id,
            best_name,
            best_result.cv_score_mean,
        )

        return TrainResponse(
            dataset_id=dataset_id,
            model_id=model_id,
            problem_type=request.problem_type,
            target_column=request.target_column,
            features=feature_columns,
            chosen_algorithm=best_name,
            n_samples_train=int(len(x_train)),
            n_samples_test=int(len(x_test)),
            candidates=candidates,
            best_metrics=best_result.test_metrics,
            model_uri=str(artifact_path),
        )

    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        raw = path.read_bytes()
        encoding = detect_encoding(raw)
        try:
            sample = raw[:8192].decode(encoding, errors="replace")
        except LookupError:
            encoding = "utf-8"
            sample = raw[:8192].decode(encoding, errors="replace")
        separator = detect_separator(sample)
        return pd.read_csv(path, sep=separator, encoding=encoding)

    def _select_features(self, x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Drop columns that are too sparse or too high-cardinality."""
        total = len(x)
        selected: list[str] = []
        for column in x.columns:
            series = x[column]
            null_ratio = float(series.isna().sum()) / total if total else 1.0
            if null_ratio > _MAX_NULL_RATIO:
                continue
            if series.dtype.name in ("object", "str", "string", "category", "bool"):
                if series.nunique(dropna=True) > _MAX_CATEGORICAL_UNIQUE:
                    continue
            selected.append(str(column))
        return x[selected].copy(), selected

    def _build_preprocessor(
        self, numeric_cols: list[str], categorical_cols: list[str]
    ) -> ColumnTransformer:
        transformers = []
        if numeric_cols:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                )
            )
        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                )
            )
        if not transformers:
            transformers.append(("passthrough", "passthrough", []))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _compute_metrics(
        self,
        pipeline: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: ProblemType,
    ) -> ModelMetrics:
        predictions = pipeline.predict(x_test)

        if problem_type == ProblemType.CLASSIFICATION:
            metrics = ModelMetrics(
                accuracy=float(accuracy_score(y_test, predictions)),
                precision=float(
                    precision_score(y_test, predictions, average="weighted", zero_division=0)
                ),
                recall=float(
                    recall_score(y_test, predictions, average="weighted", zero_division=0)
                ),
                f1=float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            )
            if y_test.nunique() == 2 and hasattr(pipeline.named_steps["model"], "predict_proba"):
                try:
                    proba = pipeline.predict_proba(x_test)[:, 1]
                    metrics.roc_auc = float(roc_auc_score(y_test, proba))
                except (ValueError, IndexError):
                    metrics.roc_auc = None
            return metrics

        mse = float(mean_squared_error(y_test, predictions))
        return ModelMetrics(
            r2=float(r2_score(y_test, predictions)),
            mae=float(mean_absolute_error(y_test, predictions)),
            rmse=float(np.sqrt(mse)),
        )
