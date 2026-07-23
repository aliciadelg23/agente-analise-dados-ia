"""Endpoint tests for POST /datasets/{dataset_id}/insights."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.llms.base import LLMResponse

_CSV = b"age,salary,city\n30,3000,Lisbon\n25,2500,Porto\n40,5000,Braga\n35,4200,Porto\n"


def _upload(client: TestClient) -> str:
    response = client.post(
        "/datasets/upload",
        files={"file": ("data.csv", io.BytesIO(_CSV), "text/csv")},
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def _canonical_json() -> str:
    return json.dumps(
        {
            "executive_summary": "Small dataset with mixed columns.",
            "insights": ["Salary correlates with age."],
            "anomalies": [],
            "suggestions": ["Add more rows before training."],
            "risks": ["Dataset is too small for reliable ML."],
        }
    )


def _mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.chat.return_value = LLMResponse(
        content=_canonical_json(),
        model="gpt-4o-mini",
        provider="openai",
        usage={"total_tokens": 21},
    )
    return provider


def test_insights_endpoint_returns_structured_analysis(client: TestClient) -> None:
    dataset_id = _upload(client)
    provider = _mock_provider()

    with patch("app.api.routes.datasets.get_llm_provider", return_value=provider):
        response = client.post(f"/datasets/{dataset_id}/insights", json={})

    assert response.status_code == 201
    body = response.json()
    assert body["dataset_id"] == dataset_id
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"
    assert body["executive_summary"].startswith("Small")
    assert body["insights"] == ["Salary correlates with age."]
    assert body["suggestions"] == ["Add more rows before training."]
    assert body["risks"] == ["Dataset is too small for reliable ML."]
    assert body["raw_llm_response"] is None


def test_insights_endpoint_forwards_provider_override(client: TestClient) -> None:
    dataset_id = _upload(client)
    provider = _mock_provider()

    with patch("app.api.routes.datasets.get_llm_provider", return_value=provider) as factory_call:
        response = client.post(
            f"/datasets/{dataset_id}/insights",
            json={"provider": "anthropic", "model": "claude-opus-4-8"},
        )

    assert response.status_code == 201
    factory_call.assert_called_once_with("anthropic")
    provider.chat.assert_called_once()
    assert provider.chat.call_args.kwargs["model"] == "claude-opus-4-8"


def test_insights_endpoint_returns_404_for_unknown_dataset(client: TestClient) -> None:
    provider = _mock_provider()

    with patch("app.api.routes.datasets.get_llm_provider", return_value=provider):
        response = client.post(f"/datasets/{uuid4()}/insights", json={})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"
