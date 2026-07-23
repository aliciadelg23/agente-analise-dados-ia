"""LangChain tools exposing the app's services.

Each tool is a factory that captures the concrete service in a
closure so the returned ``BaseTool`` can be safely handed to
LangChain agents. Tool inputs are validated by Pydantic schemas
and outputs are JSON strings, matching the format LangChain
agents expect.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.models.ml import ProblemType, TrainRequest
from app.repositories.dataset_repository import DatasetRepository
from app.services.chart_service import ChartService
from app.services.eda_service import EDAService
from app.services.ml_pipeline_service import MLPipelineService


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _error(code: str, message: str) -> str:
    return _dumps({"error": {"code": code, "message": message}})


class _DatasetIdInput(BaseModel):
    dataset_id: str = Field(..., description="Server-generated dataset identifier (UUID).")


class _ColumnStatsInput(BaseModel):
    dataset_id: str = Field(..., description="Server-generated dataset identifier (UUID).")
    column: str = Field(..., description="Column name to inspect.")


class _TrainInput(BaseModel):
    dataset_id: str = Field(..., description="Server-generated dataset identifier (UUID).")
    target_column: str = Field(..., description="Name of the target column.")
    problem_type: str = Field(
        ..., description="Problem type: 'classification' or 'regression'."
    )


def _parse_uuid(value: str) -> UUID | None:
    try:
        return UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None


def build_dataset_tool(repository: DatasetRepository) -> BaseTool:
    """Tool that reports basic filesystem info for a dataset id."""

    @tool("dataset_info", args_schema=_DatasetIdInput)
    def dataset_info(dataset_id: str) -> str:
        """Return whether the dataset id maps to a stored file and where."""
        parsed = _parse_uuid(dataset_id)
        if parsed is None:
            return _error("invalid_uuid", "dataset_id must be a UUID string.")
        path = repository.find(parsed)
        if path is None:
            return _error("dataset_not_found", f"Dataset '{parsed}' was not found.")
        return _dumps(
            {
                "dataset_id": str(parsed),
                "path": str(path),
                "filename": path.name,
                "size_bytes": path.stat().st_size,
            }
        )

    return dataset_info


def build_eda_tool(service: EDAService) -> BaseTool:
    """Tool that returns the full EDA summary for a dataset id."""

    @tool("dataset_summary", args_schema=_DatasetIdInput)
    def dataset_summary(dataset_id: str) -> str:
        """Return rows, columns, memory, dtypes, null counts, and per-column stats."""
        parsed = _parse_uuid(dataset_id)
        if parsed is None:
            return _error("invalid_uuid", "dataset_id must be a UUID string.")
        summary = service.summarize(parsed)
        return summary.model_dump_json()

    return dataset_summary


def build_statistics_tool(service: EDAService) -> BaseTool:
    """Tool that returns focused statistics for a single column."""

    @tool("column_statistics", args_schema=_ColumnStatsInput)
    def column_statistics(dataset_id: str, column: str) -> str:
        """Return numeric or categorical stats for a single column."""
        parsed = _parse_uuid(dataset_id)
        if parsed is None:
            return _error("invalid_uuid", "dataset_id must be a UUID string.")
        summary = service.summarize(parsed)
        if column in summary.numeric_stats:
            stats = summary.numeric_stats[column].model_dump()
            return _dumps({"column": column, "kind": "numeric", "stats": stats})
        if column in summary.categorical_stats:
            stats = summary.categorical_stats[column].model_dump()
            return _dumps({"column": column, "kind": "categorical", "stats": stats})
        return _error(
            "unknown_column",
            f"Column '{column}' is not part of the dataset's numeric or categorical stats.",
        )

    return column_statistics


def build_ml_tool(service: MLPipelineService) -> BaseTool:
    """Tool that trains ML candidates on a dataset."""

    @tool("train_model", args_schema=_TrainInput)
    def train_model(dataset_id: str, target_column: str, problem_type: str) -> str:
        """Train candidate models and return metrics plus the winning algorithm."""
        parsed = _parse_uuid(dataset_id)
        if parsed is None:
            return _error("invalid_uuid", "dataset_id must be a UUID string.")
        try:
            resolved_type = ProblemType(problem_type)
        except ValueError:
            return _error(
                "invalid_problem_type",
                "problem_type must be 'classification' or 'regression'.",
            )
        request = TrainRequest(target_column=target_column, problem_type=resolved_type)
        response = service.train(parsed, request)
        return response.model_dump_json()

    return train_model


def build_chart_tool(service: ChartService) -> BaseTool:
    """Tool that renders the standard chart set for a dataset."""

    @tool("generate_charts", args_schema=_DatasetIdInput)
    def generate_charts(dataset_id: str) -> str:
        """Generate histograms, boxplots, heatmap, bar charts and category distributions."""
        parsed = _parse_uuid(dataset_id)
        if parsed is None:
            return _error("invalid_uuid", "dataset_id must be a UUID string.")
        response = service.generate(parsed)
        return response.model_dump_json()

    return generate_charts


def build_tools(
    dataset_repository: DatasetRepository,
    eda_service: EDAService,
    ml_pipeline_service: MLPipelineService,
    chart_service: ChartService,
) -> list[BaseTool]:
    """Return the full tool list wired to concrete services."""
    return [
        build_dataset_tool(dataset_repository),
        build_eda_tool(eda_service),
        build_statistics_tool(eda_service),
        build_ml_tool(ml_pipeline_service),
        build_chart_tool(chart_service),
    ]
