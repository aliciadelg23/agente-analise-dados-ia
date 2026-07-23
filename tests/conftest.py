"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import _dataset_repository, get_dataset_service
from app.config.settings import get_settings
from app.main import app
from app.repositories.dataset_repository import DatasetRepository
from app.services.dataset_service import DatasetService


@pytest.fixture
def temp_storage(tmp_path: Path) -> Path:
    """Return an isolated storage directory for a single test."""
    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    return storage


@pytest.fixture
def dataset_service(temp_storage: Path) -> DatasetService:
    """Build a DatasetService pointing at an isolated storage."""
    repository = DatasetRepository(temp_storage)
    repository.ensure_ready()
    return DatasetService(repository=repository, max_size_bytes=5 * 1024 * 1024)


@pytest.fixture
def client(temp_storage: Path) -> Iterator[TestClient]:
    """FastAPI test client with dataset storage swapped to a temp dir."""
    get_settings.cache_clear()
    _dataset_repository.cache_clear()

    def _override() -> DatasetService:
        repository = DatasetRepository(temp_storage)
        repository.ensure_ready()
        return DatasetService(
            repository=repository,
            max_size_bytes=5 * 1024 * 1024,
        )

    app.dependency_overrides[get_dataset_service] = _override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_dataset_service, None)
        get_settings.cache_clear()
        _dataset_repository.cache_clear()
