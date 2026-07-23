"""Tests for the LangGraph workflow builder."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from app.agents.graph.workflow import build_workflow_graph, run_workflow


def _stub_node(returned_update: dict) -> MagicMock:
    node = MagicMock()
    node.return_value = returned_update
    return node


def test_run_workflow_visits_all_nodes_in_order() -> None:
    calls: list[str] = []

    def _record(name: str, update: dict):
        def _inner(_state):
            calls.append(name)
            return update

        return _inner

    dataset_id = uuid4()
    cleaned_id = uuid4()
    graph = build_workflow_graph(
        planner=_record("planner", {"plan": {"should_train": False}}),
        eda=_record("eda", {"eda_summary": {"rows": 5}}),
        cleaning=_record(
            "cleaning",
            {"cleaned_dataset_id": cleaned_id, "cleaning_report": {"rows_before": 5}},
        ),
        ml=_record("ml", {"ml_result": None}),
        insights=_record("insights", {"insights": {"insights": []}}),
        report=_record("report", {"final_report": "done"}),
    )

    final_state = run_workflow(graph, dataset_id=dataset_id)

    assert calls == ["planner", "eda", "cleaning", "ml", "insights", "report"]
    assert final_state["dataset_id"] == dataset_id
    assert final_state["cleaned_dataset_id"] == cleaned_id
    assert final_state["final_report"] == "done"


def test_run_workflow_forwards_plan_overrides() -> None:
    seen_state: dict = {}

    def _planner(state):
        seen_state.update(state)
        return {"plan": state.get("plan") or {}}

    graph = build_workflow_graph(
        planner=_planner,
        eda=_stub_node({"eda_summary": {}}),
        cleaning=_stub_node({"cleaned_dataset_id": uuid4(), "cleaning_report": {}}),
        ml=_stub_node({"ml_result": None}),
        insights=_stub_node({"insights": {}}),
        report=_stub_node({"final_report": ""}),
    )

    run_workflow(
        graph,
        dataset_id=uuid4(),
        user_query="analyze",
        plan_overrides={"should_train": True, "target_column": "churn"},
    )

    assert seen_state["user_query"] == "analyze"
    assert seen_state["plan"]["target_column"] == "churn"
