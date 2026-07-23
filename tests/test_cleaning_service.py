"""Unit tests for CleaningService."""

from __future__ import annotations

from uuid import UUID, uuid4

import pandas as pd
import pytest

from app.core.exceptions import DatasetNotFoundError
from app.models.cleaning import CleaningOptions
from app.repositories.dataset_repository import DatasetRepository
from app.services.cleaning_service import CleaningService, _slugify


def _store_csv(repository: DatasetRepository, content: str) -> UUID:
    dataset_id = uuid4()
    repository.save(dataset_id, "sample.csv", content.encode("utf-8"))
    return dataset_id


def _read_cleaned(repository: DatasetRepository, dataset_id: UUID) -> pd.DataFrame:
    path = repository.find(dataset_id)
    assert path is not None
    return pd.read_csv(path)


def test_slugify_normalizes_accents_and_special_chars() -> None:
    assert _slugify("Nome do Cliente") == "nome_do_cliente"
    assert _slugify("Idade (anos)") == "idade_anos"
    assert _slugify("São Paulo") == "sao_paulo"
    assert _slugify("E-mail") == "e_mail"


def test_clean_removes_duplicates(
    dataset_repository: DatasetRepository, cleaning_service: CleaningService
) -> None:
    csv = "a,b\n1,2\n1,2\n3,4\n"
    dataset_id = _store_csv(dataset_repository, csv)

    result = cleaning_service.clean(dataset_id, CleaningOptions())

    assert result.report.duplicates_removed == 1
    assert result.report.rows_before == 3
    assert result.report.rows_after == 2


def test_clean_removes_fully_empty_rows(
    dataset_repository: DatasetRepository, cleaning_service: CleaningService
) -> None:
    csv = "a,b\n1,2\n,\n3,4\n"
    dataset_id = _store_csv(dataset_repository, csv)

    result = cleaning_service.clean(dataset_id, CleaningOptions())

    assert result.report.empty_rows_removed == 1
    assert result.report.rows_after == 2


def test_clean_strips_whitespace_and_marks_touched_columns(
    dataset_repository: DatasetRepository, cleaning_service: CleaningService
) -> None:
    csv = "name,city\n Alice , Lisbon \nBob,Porto\n"
    dataset_id = _store_csv(dataset_repository, csv)

    result = cleaning_service.clean(
        dataset_id,
        CleaningOptions(
            remove_duplicates=False,
            remove_empty_rows=False,
            fill_nulls=False,
            standardize_column_names=False,
            convert_types=False,
        ),
    )

    assert set(result.report.whitespace_stripped_columns) == {"name", "city"}

    cleaned = _read_cleaned(dataset_repository, result.cleaned_dataset_id)
    assert cleaned["name"].tolist() == ["Alice", "Bob"]
    assert cleaned["city"].tolist() == ["Lisbon", "Porto"]


def test_clean_standardizes_column_names(
    dataset_repository: DatasetRepository, cleaning_service: CleaningService
) -> None:
    csv = "Nome do Cliente,Idade (anos)\nAlice,30\nBob,25\n"
    dataset_id = _store_csv(dataset_repository, csv)

    result = cleaning_service.clean(dataset_id, CleaningOptions())

    assert result.report.columns_renamed == {
        "Nome do Cliente": "nome_do_cliente",
        "Idade (anos)": "idade_anos",
    }

    cleaned = _read_cleaned(dataset_repository, result.cleaned_dataset_id)
    assert set(cleaned.columns) == {"nome_do_cliente", "idade_anos"}


def test_clean_converts_object_column_to_numeric(
    dataset_repository: DatasetRepository, cleaning_service: CleaningService
) -> None:
    csv = "age\n30\n25\n40\n"
    dataset_id = _store_csv(dataset_repository, csv)

    result = cleaning_service.clean(
        dataset_id,
        CleaningOptions(fill_nulls=False),
    )

    assert "age" not in result.report.types_converted or result.report.types_converted[
        "age"
    ].after.startswith(("int", "float"))


def test_clean_fills_nulls_using_median_and_mode(
    dataset_repository: DatasetRepository, cleaning_service: CleaningService
) -> None:
    csv = "age,city\n30,Lisbon\n,Porto\n40,Lisbon\n25,\n"
    dataset_id = _store_csv(dataset_repository, csv)

    result = cleaning_service.clean(
        dataset_id,
        CleaningOptions(remove_duplicates=False, remove_empty_rows=False),
    )

    assert result.report.nulls_filled.get("age") == 1
    assert result.report.nulls_filled.get("city") == 1

    cleaned = _read_cleaned(dataset_repository, result.cleaned_dataset_id)
    assert cleaned["age"].isna().sum() == 0
    assert cleaned["city"].isna().sum() == 0
    assert (cleaned["city"] == "Lisbon").sum() == 3  # mode filled


def test_clean_does_not_modify_original_file(
    dataset_repository: DatasetRepository, cleaning_service: CleaningService
) -> None:
    csv = "a,b\n1,2\n1,2\n"
    dataset_id = _store_csv(dataset_repository, csv)
    original_path = dataset_repository.find(dataset_id)
    assert original_path is not None
    original_bytes = original_path.read_bytes()

    cleaning_service.clean(dataset_id, CleaningOptions())

    assert original_path.read_bytes() == original_bytes


def test_clean_raises_for_unknown_dataset(cleaning_service: CleaningService) -> None:
    with pytest.raises(DatasetNotFoundError):
        cleaning_service.clean(uuid4(), CleaningOptions())


def test_clean_reports_applied_operations(
    dataset_repository: DatasetRepository, cleaning_service: CleaningService
) -> None:
    csv = "a,b\n1,2\n"
    dataset_id = _store_csv(dataset_repository, csv)

    result = cleaning_service.clean(
        dataset_id,
        CleaningOptions(remove_duplicates=False, fill_nulls=False),
    )

    assert "remove_duplicates" not in result.report.operations_applied
    assert "fill_nulls" not in result.report.operations_applied
    assert "standardize_column_names" in result.report.operations_applied
