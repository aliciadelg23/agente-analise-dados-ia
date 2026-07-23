"""Unit tests for the LangGraph workflow nodes."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from app.agents.graph.cleaning import CleaningNode
from app.agents.graph.eda import EDANode
from app.agents.graph.insights import InsightNode
from app.agents.graph.ml import MLNode
from app.agents.graph.planner import PlannerAgent
from app.agents.graph.report import ReportNode
from app.llms.base import LLMResponse


class TestPlannerAgent:
    def test_uses_explicit_overrides(self) -> None:
        planner = PlannerAgent()

        update = planner(
            {
                "dataset_id": uuid4(),
                "plan": {
                    "should_train": True,
                    "target_column": "churn",
                    "problem_type": "classification",
                },
            }
        )

        plan = update["plan"]
        assert plan["should_train"] is True
        assert plan["target_column"] == "churn"
        assert plan["problem_type"] == "classification"

    def test_skips_ml_when_target_missing(self) -> None:
        planner = PlannerAgent()

        update = planner({"dataset_id": uuid4(), "plan": {"should_train": True}})

        plan = update["plan"]
        assert plan["should_train"] is False
        assert "No target_column" in plan["notes"]

    def test_defaults_problem_type_to_classification(self) -> None:
        planner = PlannerAgent()

        update = planner(
            {
                "dataset_id": uuid4(),
                "plan": {"should_train": True, "target_column": "churn"},
            }
        )

        plan = update["plan"]
        assert plan["problem_type"] == "classification"

    def test_rejects_invalid_problem_type(self) -> None:
        planner = PlannerAgent()

        update = planner(
            {
                "dataset_id": uuid4(),
                "plan": {
                    "should_train": True,
                    "target_column": "churn",
                    "problem_type": "clustering",
                },
            }
        )

        plan = update["plan"]
        assert plan["should_train"] is False
        assert plan["problem_type"] is None


class TestEDANode:
    def test_returns_summary_dict(self) -> None:
        summary = MagicMock()
        summary.rows = 10
        summary.columns = 3
        summary.model_dump.return_value = {"rows": 10, "columns": 3}
        service = MagicMock()
        service.summarize.return_value = summary
        node = EDANode(service=service)

        dataset_id = uuid4()
        update = node({"dataset_id": dataset_id})

        service.summarize.assert_called_once_with(dataset_id)
        assert update["eda_summary"] == {"rows": 10, "columns": 3}


class TestCleaningNode:
    def test_publishes_cleaned_id_and_report(self) -> None:
        response = MagicMock()
        cleaned_id = uuid4()
        response.cleaned_dataset_id = cleaned_id
        response.report.model_dump.return_value = {"rows_before": 10, "rows_after": 8}
        service = MagicMock()
        service.clean.return_value = response
        node = CleaningNode(service=service)

        update = node({"dataset_id": uuid4()})

        assert update["cleaned_dataset_id"] == cleaned_id
        assert update["cleaning_report"]["rows_before"] == 10


class TestMLNode:
    def test_returns_none_when_plan_skips_training(self) -> None:
        service = MagicMock()
        node = MLNode(service=service)

        update = node(
            {
                "dataset_id": uuid4(),
                "cleaned_dataset_id": uuid4(),
                "plan": {"should_train": False},
            }
        )

        assert update == {"ml_result": None}
        service.train.assert_not_called()

    def test_returns_none_when_target_or_type_missing(self) -> None:
        service = MagicMock()
        node = MLNode(service=service)

        update = node(
            {
                "dataset_id": uuid4(),
                "cleaned_dataset_id": uuid4(),
                "plan": {"should_train": True},
            }
        )

        assert update == {"ml_result": None}
        service.train.assert_not_called()

    def test_trains_using_cleaned_dataset_id(self) -> None:
        response = MagicMock()
        response.chosen_algorithm = "random_forest"
        response.model_dump.return_value = {"chosen_algorithm": "random_forest"}
        service = MagicMock()
        service.train.return_value = response
        node = MLNode(service=service)

        cleaned_id = uuid4()
        node(
            {
                "dataset_id": uuid4(),
                "cleaned_dataset_id": cleaned_id,
                "plan": {
                    "should_train": True,
                    "target_column": "churn",
                    "problem_type": "classification",
                },
            }
        )

        called_id, request = service.train.call_args.args
        assert called_id == cleaned_id
        assert request.target_column == "churn"


class TestInsightNode:
    def test_returns_insights_dict(self) -> None:
        analysis = MagicMock()
        analysis.insights = ["insight-1", "insight-2"]
        analysis.provider = "openai"
        analysis.model_dump.return_value = {"insights": ["insight-1", "insight-2"]}
        service = MagicMock()
        service.analyze.return_value = analysis
        provider = MagicMock()
        node = InsightNode(service=service, llm_provider=provider)

        dataset_id = uuid4()
        cleaned_id = uuid4()
        update = node({"dataset_id": dataset_id, "cleaned_dataset_id": cleaned_id})

        called_id, called_provider = service.analyze.call_args.args
        assert called_id == cleaned_id
        assert called_provider is provider
        assert update["insights"]["insights"] == ["insight-1", "insight-2"]


class TestReportNode:
    def test_sends_state_json_and_stores_output(self) -> None:
        provider = MagicMock()
        provider.chat.return_value = LLMResponse(
            content="# Report",
            model="gpt-4o-mini",
            provider="openai",
        )
        node = ReportNode(llm_provider=provider)

        dataset_id = uuid4()
        update = node(
            {
                "dataset_id": dataset_id,
                "plan": {"should_train": False},
                "eda_summary": {"rows": 10},
                "cleaning_report": {},
                "ml_result": None,
                "insights": {"executive_summary": "ok"},
                "user_query": "summarize",
            }
        )

        provider.chat.assert_called_once()
        messages = provider.chat.call_args.args[0]
        assert messages[0].role == "system"
        assert "WORKFLOW_STATE_JSON" in messages[1].content
        assert update["final_report"] == "# Report"
