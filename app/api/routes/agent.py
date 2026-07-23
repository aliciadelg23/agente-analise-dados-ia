"""Agent-scoped endpoints (chat, ...)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from app.agents.langchain_agent import build_agent, run_agent
from app.agents.tools import build_tools
from app.api.dependencies import (
    get_chart_service,
    get_eda_service,
    get_ml_pipeline_service,
)
from app.config.settings import Settings, get_settings
from app.models.agent import AgentChatRequest, AgentChatResponse
from app.repositories.dataset_repository import DatasetRepository
from app.services.chart_service import ChartService
from app.services.eda_service import EDAService
from app.services.ml_pipeline_service import MLPipelineService

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    summary="Ask the LangChain agent a natural-language question",
)
async def chat(
    request: AgentChatRequest,
    settings: Settings = Depends(get_settings),
    eda_service: EDAService = Depends(get_eda_service),
    ml_pipeline_service: MLPipelineService = Depends(get_ml_pipeline_service),
    chart_service: ChartService = Depends(get_chart_service),
) -> AgentChatResponse:
    """Run the LangChain agent with the app's tools and return its final answer."""
    dataset_repository = DatasetRepository(Path(settings.storage_dir))
    dataset_repository.ensure_ready()
    tools = build_tools(
        dataset_repository=dataset_repository,
        eda_service=eda_service,
        ml_pipeline_service=ml_pipeline_service,
        chart_service=chart_service,
    )

    provider = (settings.default_llm_provider or "openai").lower()
    provider_config = {
        "openai": (settings.openai_api_key, settings.openai_model),
        "gemini": (settings.gemini_api_key, settings.gemini_model),
        "anthropic": (settings.anthropic_api_key, settings.anthropic_model),
    }
    api_key, default_model = provider_config.get(provider, (None, ""))
    model = request.model or default_model

    agent = build_agent(
        tools,
        provider=provider,
        api_key=api_key,
        model=model,
    )
    output = run_agent(agent, request.query)
    return AgentChatResponse(output=output, model=model)
