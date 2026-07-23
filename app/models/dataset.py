"""Schemas describing dataset payloads exchanged with the API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ColumnInfo:
    """Internal representation of a single column discovered in a CSV."""

    name: str
    dtype: str


@dataclass(frozen=True)
class DatasetMetadata:
    """Internal metadata produced by inspecting an uploaded CSV.

    Kept separate from the API response so the inspection layer does
    not depend on Pydantic and can evolve independently.
    """

    rows: int
    columns: tuple[ColumnInfo, ...]
    encoding: str
    separator: str

    @property
    def column_count(self) -> int:
        return len(self.columns)


class DatasetUploadResponse(BaseModel):
    """Response returned after a successful CSV upload."""

    dataset_id: UUID = Field(..., description="Server-generated identifier.")
    filename: str = Field(..., description="Original filename provided by the client.")
    rows: int = Field(..., ge=0, description="Number of data rows detected.")
    columns: int = Field(..., ge=0, description="Number of columns detected.")
    size: str = Field(..., description="Human-readable file size, for example '1.2 MB'.")
    uploaded_at: datetime = Field(..., description="UTC timestamp when the upload was accepted.")
    encoding: str = Field(..., description="Detected file encoding.")
    separator: str = Field(..., description="Detected column separator.")
