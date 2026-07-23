"""Smoke test for the /health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import __version__
from app.main import app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
