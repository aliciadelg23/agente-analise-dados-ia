"""Model-scoped endpoints (explainability, ...)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Path

from app.api.dependencies import get_explainability_service
from app.models.explainability import ExplainabilityResponse
from app.services.explainability_service import ExplainabilityService

router = APIRouter(prefix="/models", tags=["models"])


@router.get(
    "/{model_id}/explain",
    response_model=ExplainabilityResponse,
    summary="Feature importance, SHAP values, and summary plot for a trained model",
)
async def explain_model(
    model_id: UUID = Path(..., description="Server-generated model identifier."),
    service: ExplainabilityService = Depends(get_explainability_service),
) -> ExplainabilityResponse:
    """Compute feature importance, SHAP values, and render the summary plot."""
    return service.explain(model_id)
