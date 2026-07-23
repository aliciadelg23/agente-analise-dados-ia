"""Model explainability service.

Given a trained model id, load the pipeline and the source dataset,
compute feature importance and SHAP values, render a summary plot,
and return the aggregated view expected by the API layer.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from uuid import UUID

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from app.core.exceptions import (
    DatasetNotFoundError,
    ExplainabilityError,
    ModelNotFoundError,
)
from app.core.logging import get_logger
from app.models.explainability import (
    ExplainabilityResponse,
    FeatureImportanceItem,
    ShapSummary,
    ShapValueItem,
    TopFeature,
)
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.model_repository import ModelRepository
from app.utils.csv_inspector import detect_encoding, detect_separator

logger = get_logger(__name__)

_MAX_BACKGROUND_ROWS = 100
_MAX_SHAP_ROWS = 300
_TOP_N = 10


class ExplainabilityService:
    """Compute feature importance, SHAP values, and a summary plot."""

    def __init__(
        self,
        dataset_repository: DatasetRepository,
        model_repository: ModelRepository,
        charts_dir: Path,
        url_prefix: str,
    ) -> None:
        self._datasets = dataset_repository
        self._models = model_repository
        self._charts_dir = charts_dir
        self._url_prefix = url_prefix.rstrip("/")

    def explain(self, model_id: UUID) -> ExplainabilityResponse:
        """Return the full explainability payload for ``model_id``."""
        try:
            manifest = self._models.read_manifest(model_id)
            pipeline = self._models.load(model_id)
        except FileNotFoundError as exc:
            raise ModelNotFoundError(f"Model '{model_id}' was not found.") from exc

        dataset_id = UUID(manifest["dataset_id"])
        dataset_path = self._datasets.find(dataset_id)
        if dataset_path is None:
            raise DatasetNotFoundError(
                f"Source dataset '{dataset_id}' for model '{model_id}' was not found."
            )

        assert isinstance(pipeline, Pipeline), "Expected an sklearn Pipeline artifact."
        estimator = pipeline.named_steps["model"]

        frame = self._load_dataframe(dataset_path)
        features: list[str] = list(manifest.get("features", []))
        target = str(manifest.get("target_column", ""))
        x_df = frame.reindex(columns=features)

        y_series = frame[target] if target in frame.columns else None
        if y_series is not None:
            valid_mask = y_series.notna()
            x_df = x_df.loc[valid_mask]

        try:
            transformed_names = self._transformed_feature_names(pipeline, features)
            x_transformed = self._transform(pipeline, x_df)
            importance = self._feature_importance(estimator, transformed_names)
            shap_values = self._compute_shap(
                estimator, x_transformed, transformed_names, manifest.get("problem_type", "")
            )
            chart_url = self._render_summary_plot(
                model_id, x_transformed, shap_values, transformed_names
            )
        except (ValueError, RuntimeError, TypeError) as exc:
            logger.exception("Failed to compute explainability for model %s", model_id)
            raise ExplainabilityError(f"Failed to compute explainability: {exc}") from exc

        mean_abs = self._mean_abs_shap(shap_values)
        shap_summary = ShapSummary(
            mean_abs_values=[
                ShapValueItem(feature=f, value=v)
                for f, v in self._sorted(transformed_names, mean_abs)
            ],
            chart_url=chart_url,
        )
        top_features = self._top_features(transformed_names, importance, mean_abs)

        return ExplainabilityResponse(
            model_id=model_id,
            dataset_id=dataset_id,
            algorithm=str(manifest.get("chosen_algorithm", "")),
            problem_type=str(manifest.get("problem_type", "")),
            target_column=target,
            feature_importance=[
                FeatureImportanceItem(feature=f, importance=v)
                for f, v in self._sorted(transformed_names, importance)
            ],
            shap=shap_summary,
            top_features=top_features,
            narrative=self._narrative(top_features),
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

    def _transform(self, pipeline: Pipeline, x_df: pd.DataFrame) -> np.ndarray:
        preprocessor = pipeline.named_steps.get("prep")
        if preprocessor is None:
            return x_df.to_numpy()
        transformed = preprocessor.transform(x_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=float)

    def _transformed_feature_names(self, pipeline: Pipeline, raw_features: list[str]) -> list[str]:
        preprocessor = pipeline.named_steps.get("prep")
        if preprocessor is None:
            return list(raw_features)
        try:
            return [str(name) for name in preprocessor.get_feature_names_out()]
        except AttributeError:
            return list(raw_features)

    def _feature_importance(self, estimator: object, names: list[str]) -> np.ndarray:
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=float)
            coef = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
            values = coef
        else:
            values = np.zeros(len(names), dtype=float)
        if values.shape[0] != len(names):
            values = np.resize(values, len(names))
        return values

    def _compute_shap(
        self,
        estimator: object,
        x_transformed: np.ndarray,
        names: list[str],
        problem_type: str,
    ) -> np.ndarray:
        sample = self._sample(x_transformed, _MAX_SHAP_ROWS)
        background = self._sample(x_transformed, _MAX_BACKGROUND_ROWS)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(estimator, "feature_importances_"):
                explainer = shap.TreeExplainer(estimator)
                values = explainer.shap_values(sample)
            elif hasattr(estimator, "coef_"):
                explainer = shap.LinearExplainer(estimator, background)
                values = explainer.shap_values(sample)
            else:
                predict = getattr(estimator, "predict_proba", None) or estimator.predict
                explainer = shap.KernelExplainer(predict, background)
                values = explainer.shap_values(sample, nsamples=100, silent=True)

        return self._coerce_shap_shape(values, sample.shape, problem_type)

    def _coerce_shap_shape(
        self, values: object, sample_shape: tuple[int, int], problem_type: str
    ) -> np.ndarray:
        if isinstance(values, list):
            stack = np.stack([np.asarray(v, dtype=float) for v in values])
            if problem_type == "classification" and stack.shape[0] == 2:
                return stack[1]
            return np.mean(np.abs(stack), axis=0)

        array = np.asarray(values, dtype=float)
        if array.ndim == 3:
            if problem_type == "classification" and array.shape[-1] == 2:
                return array[..., 1]
            return np.mean(np.abs(array), axis=-1)
        if array.shape != sample_shape:
            array = array.reshape(sample_shape)
        return array

    def _mean_abs_shap(self, shap_values: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(shap_values), axis=0)

    def _sample(self, matrix: np.ndarray, max_rows: int) -> np.ndarray:
        if matrix.shape[0] <= max_rows:
            return matrix
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(matrix.shape[0], size=max_rows, replace=False)
        return matrix[idx]

    def _render_summary_plot(
        self,
        model_id: UUID,
        x_transformed: np.ndarray,
        shap_values: np.ndarray,
        names: list[str],
    ) -> str:
        target_dir = self._charts_dir / "models" / str(model_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        output = target_dir / "shap_summary.png"

        fig = plt.figure(figsize=(8, max(4, min(20, len(names) * 0.35))))
        try:
            shap.summary_plot(
                shap_values,
                x_transformed,
                feature_names=names,
                show=False,
                plot_size=None,
            )
            plt.tight_layout()
            plt.savefig(output, dpi=100, bbox_inches="tight")
        finally:
            plt.close(fig)
            plt.close("all")
        return f"{self._url_prefix}/models/{model_id}/shap_summary.png"

    def _sorted(self, names: list[str], values: np.ndarray) -> list[tuple[str, float]]:
        pairs = list(zip(names, values.tolist(), strict=False))
        pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        return [(str(f), float(v)) for f, v in pairs]

    def _top_features(
        self, names: list[str], importance: np.ndarray, shap_mean_abs: np.ndarray
    ) -> list[TopFeature]:
        ranking = {}
        for feature, value in self._sorted(names, shap_mean_abs)[:_TOP_N]:
            ranking[feature] = TopFeature(feature=feature, mean_abs_shap=value)
        for feature, value in self._sorted(names, importance)[:_TOP_N]:
            existing = ranking.get(feature)
            if existing is not None:
                existing.importance = value
            else:
                ranking[feature] = TopFeature(feature=feature, importance=value)
        return list(ranking.values())[:_TOP_N]

    def _narrative(self, top_features: list[TopFeature]) -> str:
        if not top_features:
            return "No features contributed detectably to this model's predictions."

        top_names = [feature.feature for feature in top_features[:3]]
        joined = ", ".join(f"'{name}'" for name in top_names)
        return (
            f"The model relies primarily on {joined}. "
            "See feature_importance and shap.mean_abs_values for the full ranking."
        )
