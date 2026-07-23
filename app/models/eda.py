"""Pydantic schemas for the exploratory data analysis endpoints."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class NumericColumnStats(BaseModel):
    """Descriptive statistics for a numeric column."""

    mean: float = Field(..., description="Arithmetic mean.")
    median: float = Field(..., description="50th percentile.")
    min: float = Field(..., description="Minimum observed value.")
    max: float = Field(..., description="Maximum observed value.")
    std: float = Field(..., description="Sample standard deviation.")
    q25: float = Field(..., description="First quartile (25th percentile).")
    q50: float = Field(..., description="Second quartile (50th percentile, median).")
    q75: float = Field(..., description="Third quartile (75th percentile).")


class TopValue(BaseModel):
    """Frequency of a single value inside a categorical column."""

    value: str = Field(..., description="String representation of the value.")
    count: int = Field(..., ge=0, description="Number of occurrences.")


class CategoricalColumnStats(BaseModel):
    """Descriptive statistics for a categorical column."""

    unique_count: int = Field(..., ge=0, description="Number of distinct categories.")
    top_values: list[TopValue] = Field(
        default_factory=list,
        description="Most frequent values, ordered by descending frequency.",
    )


class DatasetSummaryResponse(BaseModel):
    """Full exploratory summary for a stored dataset."""

    dataset_id: UUID = Field(..., description="Server-generated dataset identifier.")
    rows: int = Field(..., ge=0, description="Number of data rows.")
    columns: int = Field(..., ge=0, description="Number of columns.")
    memory: str = Field(..., description="Human-readable memory footprint in RAM.")
    duplicates: int = Field(..., ge=0, description="Number of duplicated rows.")
    null_counts: dict[str, int] = Field(..., description="Absolute null count per column.")
    null_percentages: dict[str, float] = Field(
        ..., description="Null count as a percentage of total rows (0-100)."
    )
    dtypes: dict[str, str] = Field(..., description="Pandas dtype string per column.")
    numeric_columns: list[str] = Field(..., description="Names of numeric columns.")
    categorical_columns: list[str] = Field(
        ..., description="Names of categorical columns (object, category, bool)."
    )
    numeric_stats: dict[str, NumericColumnStats] = Field(
        ..., description="Descriptive statistics per numeric column."
    )
    categorical_stats: dict[str, CategoricalColumnStats] = Field(
        ..., description="Descriptive statistics per categorical column."
    )
