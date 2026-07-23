"""Assemble the analysis workflow as a LangGraph state machine.

The graph is strictly linear:

    START -> planner -> eda -> cleaning -> ml -> insights -> report -> END

Nodes are injected at construction time so the same builder works
with the real services in production and with mocks in tests.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from app.agents.graph.cleaning import CleaningNode
from app.agents.graph.eda import EDANode
from app.agents.graph.insights import InsightNode
from app.agents.graph.ml import MLNode
from app.agents.graph.planner import PlannerAgent
from app.agents.graph.report import ReportNode
from app.agents.graph.state import WorkflowPlan, WorkflowState


def build_workflow_graph(
    planner: PlannerAgent,
    eda: EDANode,
    cleaning: CleaningNode,
    ml: MLNode,
    insights: InsightNode,
    report: ReportNode,
) -> Any:
    """Compile and return the linear analysis workflow."""
    graph = StateGraph(WorkflowState)
    graph.add_node("planner", planner)
    graph.add_node("eda", eda)
    graph.add_node("cleaning", cleaning)
    graph.add_node("ml", ml)
    graph.add_node("insights", insights)
    graph.add_node("report", report)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "eda")
    graph.add_edge("eda", "cleaning")
    graph.add_edge("cleaning", "ml")
    graph.add_edge("ml", "insights")
    graph.add_edge("insights", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_workflow(
    compiled_graph: Any,
    *,
    dataset_id: UUID,
    user_query: str | None = None,
    plan_overrides: WorkflowPlan | None = None,
) -> WorkflowState:
    """Invoke the compiled graph with the given inputs."""
    initial_state: WorkflowState = {"dataset_id": dataset_id, "user_query": user_query}
    if plan_overrides:
        initial_state["plan"] = plan_overrides
    return compiled_graph.invoke(initial_state)
