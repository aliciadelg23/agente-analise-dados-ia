"""Unit tests for the vector index service."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.repositories.vector_repository import VectorRepository
from app.services.vector_index_service import VectorIndexService


@pytest.fixture
def vector_service(tmp_path: Path) -> VectorIndexService:
    repository = VectorRepository(tmp_path / "chroma")
    return VectorIndexService(repository=repository)


def test_index_eda_stores_document_with_metadata(vector_service: VectorIndexService) -> None:
    dataset_id = uuid4()
    summary = {
        "dataset_id": str(dataset_id),
        "rows": 200,
        "columns": 4,
        "memory": "1.2 KB",
        "duplicates": 0,
        "numeric_columns": ["age", "salary"],
        "categorical_columns": ["city"],
        "null_percentages": {"email": 25.0, "age": 0.0},
    }

    vector_service.index_eda(dataset_id, summary)

    results = vector_service.query("many rows", top_k=1, type_filter="eda")
    assert len(results) == 1
    assert results[0]["item_id"] == str(dataset_id)
    assert results[0]["metadata"]["rows"] == 200
    assert "email" in results[0]["document"]


def test_index_insights_indexes_categories(vector_service: VectorIndexService) -> None:
    dataset_id = uuid4()
    payload = {
        "provider": "openai",
        "executive_summary": "Small dataset, mostly clean.",
        "insights": ["High correlation between age and salary."],
        "anomalies": ["12% nulls in email."],
        "suggestions": ["Drop the internal_id column."],
        "risks": ["Class imbalance in churn."],
    }

    vector_service.index_insights(dataset_id, payload)

    results = vector_service.query("class imbalance risk", top_k=1, type_filter="insights")
    assert len(results) == 1
    assert results[0]["metadata"]["provider"] == "openai"
    assert "imbalance" in results[0]["document"].lower()


def test_index_model_stores_manifest(vector_service: VectorIndexService) -> None:
    model_id = uuid4()
    manifest = {
        "model_id": str(model_id),
        "dataset_id": str(uuid4()),
        "problem_type": "classification",
        "target_column": "churn",
        "features": ["age", "salary", "city"],
        "chosen_algorithm": "random_forest",
        "cv_score_mean": 0.87,
    }

    vector_service.index_model(model_id, manifest)

    results = vector_service.query("random forest classifier", top_k=1, type_filter="model")
    assert len(results) == 1
    assert results[0]["metadata"]["algorithm"] == "random_forest"


def test_index_report_and_query_finds_it(vector_service: VectorIndexService) -> None:
    dataset_id = uuid4()

    vector_service.index_report(
        dataset_id,
        "# Analysis Report\n\nThe dataset has heavy null values in the target column.",
    )

    results = vector_service.query("heavy nulls in the target", top_k=1, type_filter="report")
    assert len(results) == 1
    assert results[0]["metadata"]["type"] == "report"


def test_query_across_all_collections_returns_matches_from_multiple_types(
    vector_service: VectorIndexService,
) -> None:
    dataset_id = uuid4()
    vector_service.index_eda(
        dataset_id,
        {
            "dataset_id": str(dataset_id),
            "rows": 100,
            "columns": 3,
            "memory": "500 B",
            "duplicates": 0,
            "numeric_columns": ["age"],
            "categorical_columns": ["city"],
            "null_percentages": {},
        },
    )
    vector_service.index_report(dataset_id, "The dataset is well balanced.")

    results = vector_service.query("dataset balance", top_k=5)
    collections = {item["collection"] for item in results}
    assert len(results) >= 1
    assert collections & {"dataset_eda", "dataset_reports"}


def test_query_raises_for_unknown_type_filter(vector_service: VectorIndexService) -> None:
    with pytest.raises(ValueError):
        vector_service.query("hello", type_filter="cluster")
