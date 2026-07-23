"""Endpoint tests for GET /datasets/{dataset_id}/charts."""

from __future__ import annotations

import io
from uuid import uuid4

from fastapi.testclient import TestClient

_CSV = b"age,salary,city\n30,3000,Lisbon\n25,2500,Porto\n40,5000,Lisbon\n35,4200,Braga\n"


def _upload(client: TestClient) -> str:
    response = client.post(
        "/datasets/upload",
        files={"file": ("data.csv", io.BytesIO(_CSV), "text/csv")},
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def test_charts_endpoint_returns_all_groups(client: TestClient) -> None:
    dataset_id = _upload(client)

    response = client.get(f"/datasets/{dataset_id}/charts")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == dataset_id
    charts = body["charts"]
    assert len(charts["histograms"]) == 2
    assert len(charts["boxplots"]) == 2
    assert charts["correlation_heatmap"] is not None
    assert len(charts["bar_charts"]) == 1
    assert charts["category_distributions"] is not None


def test_charts_endpoint_returns_404_for_unknown_dataset(client: TestClient) -> None:
    response = client.get(f"/datasets/{uuid4()}/charts")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"


def test_charts_endpoint_returns_urls_with_expected_prefix(client: TestClient) -> None:
    dataset_id = _upload(client)

    response = client.get(f"/datasets/{dataset_id}/charts")

    charts = response.json()["charts"]
    prefix = f"/static/charts/{dataset_id}/"
    assert charts["histograms"][0]["png_url"].startswith(prefix)
    assert charts["histograms"][0]["html_url"].endswith(".html")
    assert charts["correlation_heatmap"]["png_url"].startswith(prefix)
