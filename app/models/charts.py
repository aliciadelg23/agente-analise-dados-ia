"""Pydantic schemas for the chart-generation endpoint."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class ChartArtifact(BaseModel):
    """URLs for a single chart in both output formats."""

    png_url: str = Field(..., description="URL of the static PNG rendered with matplotlib.")
    html_url: str = Field(..., description="URL of the interactive HTML rendered with plotly.")


class ColumnChart(ChartArtifact):
    """A chart tied to a specific column."""

    column: str = Field(..., description="Column name the chart was generated for.")


class DatasetChartsResponse(BaseModel):
    """Full set of charts generated for a dataset."""

    dataset_id: UUID = Field(..., description="Source dataset identifier.")
    charts: DatasetCharts = Field(..., description="Charts grouped by type.")


class DatasetCharts(BaseModel):
    """Group of charts produced for a dataset, one field per chart type."""

    histograms: list[ColumnChart] = Field(
        default_factory=list,
        description="One histogram per numeric column.",
    )
    boxplots: list[ColumnChart] = Field(
        default_factory=list,
        description="One boxplot per numeric column.",
    )
    correlation_heatmap: ChartArtifact | None = Field(
        default=None,
        description="Correlation heatmap across numeric columns; null when fewer than two numeric columns exist.",
    )
    bar_charts: list[ColumnChart] = Field(
        default_factory=list,
        description="Top-N value counts per categorical column.",
    )
    category_distributions: ChartArtifact | None = Field(
        default=None,
        description="Aggregate view of unique-value counts across categorical columns.",
    )


DatasetChartsResponse.model_rebuild()
