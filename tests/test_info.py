"""Tests for the root info endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_returns_service_info(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Agente de Analise de Dados com IA"
    assert body["docs_url"] == "/docs"
    assert "version" in body
    assert "environment" in body
