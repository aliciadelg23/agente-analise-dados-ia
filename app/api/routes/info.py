"""Root endpoint exposing service metadata."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.config.settings import Settings, get_settings
from app.models.api import ApiInfoResponse

router = APIRouter(tags=["info"])


@router.get("/", response_model=ApiInfoResponse, summary="Service metadata")
async def api_info(settings: Settings = Depends(get_settings)) -> ApiInfoResponse:
    """Return high-level information about the running service."""
    return ApiInfoResponse(
        name="Agente de Analise de Dados com IA",
        version=__version__,
        description="Plataforma de analise de dados orquestrada por agentes de IA.",
        docs_url="/docs",
        environment=settings.app_env,
    )
