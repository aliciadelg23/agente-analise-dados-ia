"""Endpoint tests for POST /datasets/{dataset_id}/clean."""

from __future__ import annotations

import io
from uuid import uuid4

from fastapi.testclient import TestClient

_MESSY_CSV = (
    b"Nome do Cliente,Idade (anos),City\n Alice ,30,Lisbon\n Alice ,30,Lisbon\nBob,,Porto\n,,\n"
)


def _upload(client: TestClient) -> str:
    response = client.post(
        "/datasets/upload",
        files={"file": ("messy.csv", io.BytesIO(_MESSY_CSV), "text/csv")},
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def test_clean_endpoint_applies_full_pipeline_by_default(client: TestClient) -> None:
    dataset_id = _upload(client)

    response = client.post(f"/datasets/{dataset_id}/clean", json={})

    assert response.status_code == 201
    body = response.json()
    assert body["original_dataset_id"] == dataset_id
    assert body["cleaned_dataset_id"] != dataset_id
    report = body["report"]
    assert report["duplicates_removed"] == 1
    assert report["empty_rows_removed"] == 1
    assert set(report["columns_renamed"].keys()) == {"Nome do Cliente", "Idade (anos)", "City"}
    assert set(report["operations_applied"]) >= {
        "remove_duplicates",
        "remove_empty_rows",
        "standardize_column_names",
        "strip_whitespace",
    }


def test_clean_endpoint_respects_disabled_options(client: TestClient) -> None:
    dataset_id = _upload(client)

    response = client.post(
        f"/datasets/{dataset_id}/clean",
        json={
            "remove_duplicates": False,
            "remove_empty_rows": False,
            "fill_nulls": False,
            "strip_whitespace": False,
            "standardize_column_names": False,
            "convert_types": False,
        },
    )

    assert response.status_code == 201
    report = response.json()["report"]
    assert report["duplicates_removed"] == 0
    assert report["empty_rows_removed"] == 0
    assert report["operations_applied"] == []


def test_clean_endpoint_returns_404_for_unknown_dataset(client: TestClient) -> None:
    response = client.post(f"/datasets/{uuid4()}/clean", json={})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"


def test_clean_endpoint_result_is_queryable_by_summary(client: TestClient) -> None:
    dataset_id = _upload(client)

    clean_response = client.post(f"/datasets/{dataset_id}/clean", json={})
    cleaned_id = clean_response.json()["cleaned_dataset_id"]

    summary = client.get(f"/datasets/{cleaned_id}/summary")

    assert summary.status_code == 200
    assert summary.json()["dataset_id"] == cleaned_id
