"""Unit tests for the LangChain tools."""

from __future__ import annotations

import json
import random
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from app.agents.tools import (
    build_chart_tool,
    build_dataset_tool,
    build_eda_tool,
    build_ml_tool,
    build_statistics_tool,
    build_tools,
)
from app.repositories.dataset_repository import DatasetRepository
from app.services.chart_service import ChartService
from app.services.eda_service import EDAService
from app.services.ml_pipeline_service import MLPipelineService


def _store_csv(repository: DatasetRepository) -> UUID:
    rng = random.Random(42)
    rows = ["age,salary,city,churn"]
    for _ in range(150):
        rows.append(
            f"{rng.randint(18, 70)},{rng.randint(1000, 10000)},"
            f"{rng.choice(['Lisbon', 'Porto', 'Braga'])},"
            f"{1 if rng.random() > 0.7 else 0}"
        )
    dataset_id = uuid4()
    repository.save(dataset_id, "sample.csv", ("\n".join(rows) + "\n").encode("utf-8"))
    return dataset_id


class TestDatasetTool:
    def test_returns_metadata_for_known_dataset(
        self, dataset_repository: DatasetRepository
    ) -> None:
        dataset_id = _store_csv(dataset_repository)
        tool = build_dataset_tool(dataset_repository)

        payload = json.loads(tool.invoke({"dataset_id": str(dataset_id)}))

        assert payload["dataset_id"] == str(dataset_id)
        assert payload["filename"].startswith(str(dataset_id))
        assert payload["size_bytes"] > 0

    def test_returns_error_for_unknown_dataset(self, dataset_repository: DatasetRepository) -> None:
        tool = build_dataset_tool(dataset_repository)

        payload = json.loads(tool.invoke({"dataset_id": str(uuid4())}))

        assert payload["error"]["code"] == "dataset_not_found"

    def test_returns_error_for_invalid_uuid(self, dataset_repository: DatasetRepository) -> None:
        tool = build_dataset_tool(dataset_repository)

        payload = json.loads(tool.invoke({"dataset_id": "not-a-uuid"}))

        assert payload["error"]["code"] == "invalid_uuid"


class TestEdaTool:
    def test_returns_full_summary(
        self, dataset_repository: DatasetRepository, eda_service: EDAService
    ) -> None:
        dataset_id = _store_csv(dataset_repository)
        tool = build_eda_tool(eda_service)

        payload = json.loads(tool.invoke({"dataset_id": str(dataset_id)}))

        assert payload["dataset_id"] == str(dataset_id)
        assert payload["rows"] == 150
        assert "numeric_stats" in payload


class TestStatisticsTool:
    def test_returns_numeric_stats_for_numeric_column(
        self, dataset_repository: DatasetRepository, eda_service: EDAService
    ) -> None:
        dataset_id = _store_csv(dataset_repository)
        tool = build_statistics_tool(eda_service)

        payload = json.loads(tool.invoke({"dataset_id": str(dataset_id), "column": "age"}))

        assert payload["column"] == "age"
        assert payload["kind"] == "numeric"
        assert "mean" in payload["stats"]

    def test_returns_categorical_stats_for_categorical_column(
        self, dataset_repository: DatasetRepository, eda_service: EDAService
    ) -> None:
        dataset_id = _store_csv(dataset_repository)
        tool = build_statistics_tool(eda_service)

        payload = json.loads(tool.invoke({"dataset_id": str(dataset_id), "column": "city"}))

        assert payload["column"] == "city"
        assert payload["kind"] == "categorical"
        assert payload["stats"]["unique_count"] >= 1

    def test_returns_error_for_unknown_column(
        self, dataset_repository: DatasetRepository, eda_service: EDAService
    ) -> None:
        dataset_id = _store_csv(dataset_repository)
        tool = build_statistics_tool(eda_service)

        payload = json.loads(tool.invoke({"dataset_id": str(dataset_id), "column": "not_a_column"}))

        assert payload["error"]["code"] == "unknown_column"


class TestMlTool:
    def test_forwards_to_ml_service(self, ml_pipeline_service: MLPipelineService) -> None:
        wrapped = MagicMock(spec=MLPipelineService)
        response = MagicMock()
        response.model_dump_json.return_value = '{"chosen_algorithm": "random_forest"}'
        wrapped.train.return_value = response

        tool = build_ml_tool(wrapped)
        payload = json.loads(
            tool.invoke(
                {
                    "dataset_id": str(uuid4()),
                    "target_column": "churn",
                    "problem_type": "classification",
                }
            )
        )

        assert payload["chosen_algorithm"] == "random_forest"
        wrapped.train.assert_called_once()

    def test_returns_error_for_invalid_problem_type(
        self, ml_pipeline_service: MLPipelineService
    ) -> None:
        tool = build_ml_tool(ml_pipeline_service)

        payload = json.loads(
            tool.invoke(
                {
                    "dataset_id": str(uuid4()),
                    "target_column": "churn",
                    "problem_type": "clustering",
                }
            )
        )

        assert payload["error"]["code"] == "invalid_problem_type"


class TestChartTool:
    def test_forwards_to_chart_service(self, chart_service: ChartService) -> None:
        wrapped = MagicMock(spec=ChartService)
        response = MagicMock()
        response.model_dump_json.return_value = '{"dataset_id": "abc", "charts": {}}'
        wrapped.generate.return_value = response

        tool = build_chart_tool(wrapped)
        payload = json.loads(tool.invoke({"dataset_id": str(uuid4())}))

        assert "dataset_id" in payload
        wrapped.generate.assert_called_once()


class TestBuildTools:
    def test_returns_five_tools_with_expected_names(
        self,
        dataset_repository: DatasetRepository,
        eda_service: EDAService,
        ml_pipeline_service: MLPipelineService,
        chart_service: ChartService,
    ) -> None:
        tools = build_tools(
            dataset_repository=dataset_repository,
            eda_service=eda_service,
            ml_pipeline_service=ml_pipeline_service,
            chart_service=chart_service,
        )

        names = [tool.name for tool in tools]
        assert names == [
            "dataset_info",
            "dataset_summary",
            "column_statistics",
            "train_model",
            "generate_charts",
        ]
