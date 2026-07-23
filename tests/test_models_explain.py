"""Endpoint tests for GET /models/{model_id}/explain."""

from __future__ import annotations

import io
import random
from uuid import uuid4

from fastapi.testclient import TestClient


def _synthetic_csv() -> bytes:
    rng = random.Random(42)
    lines = ["age,salary,city,churn"]
    for _ in range(150):
        age = rng.randint(18, 70)
        salary = rng.randint(1000, 10000)
        city = rng.choice(["Lisbon", "Porto", "Braga"])
        churn = 1 if (age > 50 or salary < 2000) else 0
        lines.append(f"{age},{salary},{city},{churn}")
    return ("\n".join(lines) + "\n").encode()


def _train_model(client: TestClient) -> str:
    upload = client.post(
        "/datasets/upload",
        files={"file": ("d.csv", io.BytesIO(_synthetic_csv()), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]
    train = client.post(
        f"/datasets/{dataset_id}/train",
        json={"target_column": "churn", "problem_type": "classification"},
    )
    return train.json()["model_id"]


def test_explain_endpoint_returns_200_with_full_payload(client: TestClient) -> None:
    model_id = _train_model(client)

    response = client.get(f"/models/{model_id}/explain")

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == model_id
    assert body["target_column"] == "churn"
    assert body["problem_type"] == "classification"
    assert len(body["feature_importance"]) > 0
    assert len(body["shap"]["mean_abs_values"]) > 0
    assert body["shap"]["chart_url"].endswith("shap_summary.png")
    assert body["top_features"]
    assert body["narrative"]


def test_explain_endpoint_returns_404_for_unknown_model(client: TestClient) -> None:
    response = client.get(f"/models/{uuid4()}/explain")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
