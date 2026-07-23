"""Pydantic schemas for the data-cleaning endpoint."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class CleaningOptions(BaseModel):
    """Flags controlling which cleaning steps to apply.

    Every flag defaults to True so an empty request body applies the
    full pipeline. Clients can opt out of individual steps.
    """

    remove_duplicates: bool = Field(default=True, description="Drop duplicated rows.")
    remove_empty_rows: bool = Field(
        default=True, description="Drop rows where every value is null after stripping."
    )
    fill_nulls: bool = Field(
        default=True,
        description="Fill nulls (numeric with median, categorical with mode).",
    )
    strip_whitespace: bool = Field(
        default=True,
        description="Trim surrounding whitespace on textual columns; empty strings become nulls.",
    )
    standardize_column_names: bool = Field(
        default=True,
        description="Normalize column names to snake_case ASCII.",
    )
    convert_types: bool = Field(
        default=True,
        description="Attempt automatic type conversion (numeric, datetime).",
    )


class TypeConversion(BaseModel):
    """Represents a dtype change applied to a single column."""

    before: str = Field(..., description="Original pandas dtype.")
    after: str = Field(..., description="New pandas dtype after conversion.")


class CleaningReport(BaseModel):
    """Summary of what changed during a cleaning run."""

    rows_before: int = Field(..., ge=0, description="Row count before cleaning.")
    rows_after: int = Field(..., ge=0, description="Row count after cleaning.")
    rows_removed: int = Field(..., ge=0, description="Rows removed (duplicates + empty).")
    duplicates_removed: int = Field(..., ge=0, description="Rows removed by deduplication.")
    empty_rows_removed: int = Field(..., ge=0, description="Rows removed for being fully empty.")
    nulls_filled: dict[str, int] = Field(
        default_factory=dict,
        description="Per-column count of null values filled.",
    )
    whitespace_stripped_columns: list[str] = Field(
        default_factory=list,
        description="Columns whose values had surrounding whitespace stripped.",
    )
    columns_renamed: dict[str, str] = Field(
        default_factory=dict,
        description="Map of original column name to standardized name.",
    )
    types_converted: dict[str, TypeConversion] = Field(
        default_factory=dict,
        description="Per-column dtype conversions performed.",
    )
    operations_applied: list[str] = Field(
        default_factory=list,
        description="Names of the cleaning steps that ran.",
    )


class DatasetCleanResponse(BaseModel):
    """Response returned after a successful cleaning run."""

    original_dataset_id: UUID = Field(..., description="The source dataset (unchanged).")
    cleaned_dataset_id: UUID = Field(..., description="Identifier of the new cleaned dataset.")
    report: CleaningReport = Field(..., description="Detailed change report.")
