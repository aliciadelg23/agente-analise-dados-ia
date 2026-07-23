"""Pydantic schemas for the vector-database endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class VectorQueryRequest(BaseModel):
    """Body accepted by POST /vector/query."""

    query: str = Field(..., min_length=1, description="Natural-language query.")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum matches to return.")
    type_filter: str | None = Field(
        default=None,
        description="Restrict search to a single type: eda, insights, model, or report.",
    )


class VectorMatch(BaseModel):
    """Single match returned by the vector store."""

    collection: str = Field(..., description="Name of the collection the match came from.")
    item_id: str = Field(..., description="Identifier of the indexed item.")
    document: str = Field(..., description="Text document stored in the vector store.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Structured metadata.")
    distance: float = Field(..., description="Cosine distance; lower is more similar.")


class VectorQueryResponse(BaseModel):
    """Response returned by POST /vector/query."""

    matches: list[VectorMatch] = Field(default_factory=list)


class VectorIndexResponse(BaseModel):
    """Response returned by POST /vector/index/{dataset_id}."""

    dataset_id: UUID
    indexed: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-type flag indicating whether that artifact was indexed.",
    )
