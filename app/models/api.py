"""Schemas describing high-level API responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ApiInfoResponse(BaseModel):
    """Payload returned by the root endpoint."""

    name: str = Field(..., description="Human-readable service name.")
    version: str = Field(..., description="Package version.")
    description: str = Field(..., description="Short service description.")
    docs_url: str = Field(..., description="URL where interactive docs are exposed.")
    environment: str = Field(..., description="Runtime environment (development, production, ...).")
