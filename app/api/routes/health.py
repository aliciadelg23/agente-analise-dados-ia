"""Liveness endpoint used by CI and orchestration platforms."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service liveness probe")
async def health() -> dict[str, str]:
    """Return a static payload indicating the service is up."""
    return {"status": "ok", "version": __version__}
