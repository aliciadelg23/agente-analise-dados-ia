"""Unit tests for EDAService."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.core.exceptions import DatasetNotFoundError
from app.repositories.dataset_repository import DatasetRepository
from app.services.eda_service import EDAService

_CSV_WITH_NULLS_AND_DUPES = (
    "name,age,city\nAlice,30,Lisbon\nBob,25,Porto\nCarol,,Braga\nAlice,30,Lisbon\nDan,40,\n"
)


def _store_csv(repository: DatasetRepository, content: str) -> UUID:
    dataset_id = uuid4()
    repository.save(dataset_id, "sample.csv", content.encode("utf-8"))
    return dataset_id


def test_summarize_computes_row_column_and_duplicate_counts(
    dataset_repository: DatasetRepository, eda_service: EDAService
) -> None:
    dataset_id = _store_csv(dataset_repository, _CSV_WITH_NULLS_AND_DUPES)

    summary = eda_service.summarize(dataset_id)

    assert summary.rows == 5
    assert summary.columns == 3
    assert summary.duplicates == 1


def test_summarize_reports_nulls_and_percentages(
    dataset_repository: DatasetRepository, eda_service: EDAService
) -> None:
    dataset_id = _store_csv(dataset_repository, _CSV_WITH_NULLS_AND_DUPES)

    summary = eda_service.summarize(dataset_id)

    assert summary.null_counts["age"] == 1
    assert summary.null_counts["city"] == 1
    assert summary.null_counts["name"] == 0
    assert summary.null_percentages["age"] == pytest.approx(20.0)
    assert summary.null_percentages["name"] == pytest.approx(0.0)


def test_summarize_classifies_numeric_and_categorical_columns(
    dataset_repository: DatasetRepository, eda_service: EDAService
) -> None:
    dataset_id = _store_csv(dataset_repository, _CSV_WITH_NULLS_AND_DUPES)

    summary = eda_service.summarize(dataset_id)

    assert summary.numeric_columns == ["age"]
    assert set(summary.categorical_columns) == {"name", "city"}


def test_summarize_returns_expected_numeric_statistics(
    dataset_repository: DatasetRepository, eda_service: EDAService
) -> None:
    dataset_id = _store_csv(dataset_repository, _CSV_WITH_NULLS_AND_DUPES)

    summary = eda_service.summarize(dataset_id)

    age = summary.numeric_stats["age"]
    assert age.min == pytest.approx(25.0)
    assert age.max == pytest.approx(40.0)
    assert age.median == pytest.approx(30.0)
    assert age.mean == pytest.approx(31.25)
    assert age.q25 <= age.median <= age.q75


def test_summarize_returns_top_categorical_values(
    dataset_repository: DatasetRepository, eda_service: EDAService
) -> None:
    dataset_id = _store_csv(dataset_repository, _CSV_WITH_NULLS_AND_DUPES)

    summary = eda_service.summarize(dataset_id)

    name_stats = summary.categorical_stats["name"]
    assert name_stats.unique_count == 4
    top = {item.value: item.count for item in name_stats.top_values}
    assert top["Alice"] == 2


def test_summarize_raises_when_dataset_not_found(eda_service: EDAService) -> None:
    with pytest.raises(DatasetNotFoundError):
        eda_service.summarize(uuid4())
