"""Tests for the /datasets/upload endpoint and DatasetService."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    InvalidFileExtensionError,
)
from app.services.dataset_service import DatasetService

_VALID_CSV = b"name,age,city\nAlice,30,Lisbon\nBob,25,Porto\nCarol,40,Braga\n"


def test_upload_valid_csv_returns_metadata(client: TestClient) -> None:
    response = client.post(
        "/datasets/upload",
        files={"file": ("people.csv", io.BytesIO(_VALID_CSV), "text/csv")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "people.csv"
    assert body["rows"] == 3
    assert body["columns"] == 3
    assert body["separator"] == ","
    assert body["size"].endswith(("B", "KB", "MB", "GB", "TB"))
    assert "dataset_id" in body
    assert "uploaded_at" in body


def test_upload_rejects_non_csv_extension(client: TestClient) -> None:
    response = client.post(
        "/datasets/upload",
        files={"file": ("data.txt", io.BytesIO(b"a,b\n1,2\n"), "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_file_extension"


def test_upload_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/datasets/upload",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"


def test_upload_rejects_garbage_content_with_csv_extension(client: TestClient) -> None:
    response = client.post(
        "/datasets/upload",
        files={"file": ("bad.csv", io.BytesIO(b"\x00\x01\x02\x03"), "text/csv")},
    )

    assert response.status_code in {400, 422}
    assert response.json()["error"]["code"] in {"empty_file", "invalid_csv"}


def test_service_rejects_file_over_size_limit(dataset_service: DatasetService) -> None:
    dataset_service._max_size_bytes = 10

    with pytest.raises(FileTooLargeError):
        dataset_service.upload("big.csv", b"a,b,c\n" + b"x" * 100)


def test_service_rejects_invalid_extension(dataset_service: DatasetService) -> None:
    with pytest.raises(InvalidFileExtensionError):
        dataset_service.upload("data.txt", b"a,b\n1,2\n")


def test_service_rejects_empty_content(dataset_service: DatasetService) -> None:
    with pytest.raises(EmptyFileError):
        dataset_service.upload("empty.csv", b"")
