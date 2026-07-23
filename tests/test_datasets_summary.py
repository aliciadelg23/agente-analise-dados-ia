"""Endpoint tests for GET /datasets/{dataset_id}/summary."""

from __future__ import annotations

import io
from uuid import uuid4

from fastapi.testclient import TestClient

_VALID_CSV = b"name,age,city\nAlice,30,Lisbon\nBob,25,Porto\nCarol,40,Lisbon\nDan,,Braga\n"


def _upload(client: TestClient) -> str:
    response = client.post(
        "/datasets/upload",
        files={"file": ("people.csv", io.BytesIO(_VALID_CSV), "text/csv")},
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def test_summary_returns_200_for_known_dataset(client: TestClient) -> None:
    dataset_id = _upload(client)

    response = client.get(f"/datasets/{dataset_id}/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == dataset_id
    assert body["rows"] == 4
    assert body["columns"] == 3
    assert body["numeric_columns"] == ["age"]
    assert set(body["categorical_columns"]) == {"name", "city"}
    assert body["null_counts"]["age"] == 1
    assert body["null_percentages"]["age"] == 25.0
    assert "age" in body["numeric_stats"]
    assert body["numeric_stats"]["age"]["min"] == 25.0
    assert body["numeric_stats"]["age"]["max"] == 40.0


def test_summary_returns_404_for_unknown_dataset(client: TestClient) -> None:
    response = client.get(f"/datasets/{uuid4()}/summary")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"


def test_summary_rejects_invalid_uuid(client: TestClient) -> None:
    response = client.get("/datasets/not-a-uuid/summary")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
