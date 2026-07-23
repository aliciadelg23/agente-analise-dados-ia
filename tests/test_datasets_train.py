"""Endpoint tests for POST /datasets/{dataset_id}/train."""

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


def _upload(client: TestClient) -> str:
    response = client.post(
        "/datasets/upload",
        files={"file": ("data.csv", io.BytesIO(_synthetic_csv()), "text/csv")},
    )
    assert response.status_code == 201
    return response.json()["dataset_id"]


def test_train_endpoint_returns_201_with_full_result(client: TestClient) -> None:
    dataset_id = _upload(client)

    response = client.post(
        f"/datasets/{dataset_id}/train",
        json={"target_column": "churn", "problem_type": "classification"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["dataset_id"] == dataset_id
    assert body["problem_type"] == "classification"
    assert body["target_column"] == "churn"
    assert len(body["candidates"]) == 3
    assert body["chosen_algorithm"] in {"logistic_regression", "decision_tree", "random_forest"}
    assert body["best_metrics"]["accuracy"] is not None


def test_train_endpoint_returns_400_for_missing_target(client: TestClient) -> None:
    dataset_id = _upload(client)

    response = client.post(
        f"/datasets/{dataset_id}/train",
        json={"target_column": "not_a_column", "problem_type": "classification"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_target_column"


def test_train_endpoint_returns_422_for_invalid_problem_type(client: TestClient) -> None:
    dataset_id = _upload(client)

    response = client.post(
        f"/datasets/{dataset_id}/train",
        json={"target_column": "churn", "problem_type": "clustering"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_train_endpoint_returns_404_for_unknown_dataset(client: TestClient) -> None:
    response = client.post(
        f"/datasets/{uuid4()}/train",
        json={"target_column": "churn", "problem_type": "classification"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"
