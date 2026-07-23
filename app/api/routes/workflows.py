"""Workflow-scoped endpoints (multi-agent analysis, ...)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agents.graph.cleaning import CleaningNode
from app.agents.graph.eda import EDANode
from app.agents.graph.insights import InsightNode
from app.agents.graph.ml import MLNode
from app.agents.graph.planner import PlannerAgent
from app.agents.graph.report import ReportNode
from app.agents.graph.state import WorkflowPlan
from app.agents.graph.workflow import build_workflow_graph, run_workflow
from app.api.dependencies import (
    get_ai_insight_service,
    get_cleaning_service,
    get_eda_service,
    get_ml_pipeline_service,
    get_vector_index_service,
)
from app.llms.factory import get_llm_provider
from app.models.workflow import WorkflowAnalyzeRequest, WorkflowAnalyzeResponse
from app.services.ai_insight_service import AIInsightService
from app.services.cleaning_service import CleaningService
from app.services.eda_service import EDAService
from app.services.ml_pipeline_service import MLPipelineService
from app.services.vector_index_service import VectorIndexService

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.post(
    "/analyze",
    response_model=WorkflowAnalyzeResponse,
    summary="Run the full multi-agent analysis workflow",
)
async def analyze(
    request: WorkflowAnalyzeRequest,
    eda_service: EDAService = Depends(get_eda_service),
    cleaning_service: CleaningService = Depends(get_cleaning_service),
    ml_service: MLPipelineService = Depends(get_ml_pipeline_service),
    insight_service: AIInsightService = Depends(get_ai_insight_service),
    vector_service: VectorIndexService = Depends(get_vector_index_service),
) -> WorkflowAnalyzeResponse:
    """Execute planner -> eda -> cleaning -> ml -> insights -> report."""
    llm_provider = get_llm_provider(request.llm_provider)

    planner = PlannerAgent()
    eda_node = EDANode(service=eda_service, vector_index=vector_service)
    cleaning_node = CleaningNode(service=cleaning_service)
    ml_node = MLNode(service=ml_service, vector_index=vector_service)
    insight_node = InsightNode(
        service=insight_service, llm_provider=llm_provider, vector_index=vector_service
    )
    report_node = ReportNode(llm_provider=llm_provider, vector_index=vector_service)

    graph = build_workflow_graph(
        planner=planner,
        eda=eda_node,
        cleaning=cleaning_node,
        ml=ml_node,
        insights=insight_node,
        report=report_node,
    )

    plan_overrides: WorkflowPlan = {
        "should_train": request.run_ml,
        "target_column": request.target_column,
        "problem_type": request.problem_type,
    }
    final_state = run_workflow(
        graph,
        dataset_id=request.dataset_id,
        user_query=request.user_query,
        plan_overrides=plan_overrides,
    )

    return WorkflowAnalyzeResponse(
        dataset_id=request.dataset_id,
        cleaned_dataset_id=final_state.get("cleaned_dataset_id"),
        plan=dict(final_state.get("plan") or {}),
        eda_summary=final_state.get("eda_summary") or {},
        cleaning_report=final_state.get("cleaning_report") or {},
        ml_result=final_state.get("ml_result"),
        insights=final_state.get("insights") or {},
        final_report=final_state.get("final_report") or "",
    )
