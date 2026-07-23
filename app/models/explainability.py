"""Pydantic schemas for the model explainability endpoint."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class FeatureImportanceItem(BaseModel):
    """Feature importance derived from the estimator itself."""

    feature: str = Field(..., description="Feature name after preprocessing.")
    importance: float = Field(
        ..., description="Importance value (feature_importances_ for trees, |coef_| for linear)."
    )


class ShapValueItem(BaseModel):
    """Aggregate SHAP magnitude for a single feature."""

    feature: str = Field(..., description="Feature name after preprocessing.")
    value: float = Field(..., description="Mean absolute SHAP value across the sample.")


class ShapSummary(BaseModel):
    """SHAP aggregates plus the summary plot URL."""

    mean_abs_values: list[ShapValueItem] = Field(
        ..., description="Features ordered by descending mean absolute SHAP magnitude."
    )
    chart_url: str = Field(..., description="URL of the summary plot (PNG).")


class TopFeature(BaseModel):
    """Consolidated view for the top-impact features."""

    feature: str = Field(..., description="Feature name after preprocessing.")
    importance: float | None = Field(
        default=None, description="Model-level importance (may be null for kernel explainers)."
    )
    mean_abs_shap: float | None = Field(
        default=None, description="Mean absolute SHAP magnitude."
    )


class ExplainabilityResponse(BaseModel):
    """Response returned by GET /models/{id}/explain."""

    model_id: UUID = Field(..., description="Model identifier.")
    dataset_id: UUID = Field(..., description="Source dataset identifier.")
    algorithm: str = Field(..., description="Algorithm chosen at training time.")
    problem_type: str = Field(..., description="'classification' or 'regression'.")
    target_column: str = Field(..., description="Column the model was trained to predict.")
    feature_importance: list[FeatureImportanceItem] = Field(
        ..., description="Per-feature importance, ordered by descending value."
    )
    shap: ShapSummary = Field(..., description="Aggregated SHAP output and chart URL.")
    top_features: list[TopFeature] = Field(
        ..., description="Consolidated top-N view combining importance and SHAP."
    )
    narrative: str = Field(
        ..., description="Short human-readable explanation of the model's dominant features."
    )
