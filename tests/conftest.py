"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    _dataset_repository,
    get_cleaning_service,
    get_dataset_service,
    get_eda_service,
)
from app.config.settings import get_settings
from app.main import app
from app.repositories.dataset_repository import DatasetRepository
from app.services.cleaning_service import CleaningService
from app.services.dataset_service import DatasetService
from app.services.eda_service import EDAService


@pytest.fixture
def temp_storage(tmp_path: Path) -> Path:
    """Return an isolated storage directory for a single test."""
    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    return storage


@pytest.fixture
def dataset_repository(temp_storage: Path) -> DatasetRepository:
    """Filesystem repository backed by the per-test storage directory."""
    repository = DatasetRepository(temp_storage)
    repository.ensure_ready()
    return repository


@pytest.fixture
def dataset_service(dataset_repository: DatasetRepository) -> DatasetService:
    """Build a DatasetService pointing at an isolated storage."""
    return DatasetService(repository=dataset_repository, max_size_bytes=5 * 1024 * 1024)


@pytest.fixture
def eda_service(dataset_repository: DatasetRepository) -> EDAService:
    """Build an EDAService pointing at an isolated storage."""
    return EDAService(repository=dataset_repository)


@pytest.fixture
def cleaning_service(dataset_repository: DatasetRepository) -> CleaningService:
    """Build a CleaningService pointing at an isolated storage."""
    return CleaningService(repository=dataset_repository)


@pytest.fixture
def client(temp_storage: Path) -> Iterator[TestClient]:
    """FastAPI test client with dataset storage swapped to a temp dir."""
    get_settings.cache_clear()
    _dataset_repository.cache_clear()

    def _override_dataset() -> DatasetService:
        repository = DatasetRepository(temp_storage)
        repository.ensure_ready()
        return DatasetService(
            repository=repository,
            max_size_bytes=5 * 1024 * 1024,
        )

    def _override_eda() -> EDAService:
        repository = DatasetRepository(temp_storage)
        repository.ensure_ready()
        return EDAService(repository=repository)

    def _override_cleaning() -> CleaningService:
        repository = DatasetRepository(temp_storage)
        repository.ensure_ready()
        return CleaningService(repository=repository)

    app.dependency_overrides[get_dataset_service] = _override_dataset
    app.dependency_overrides[get_eda_service] = _override_eda
    app.dependency_overrides[get_cleaning_service] = _override_cleaning
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_dataset_service, None)
        app.dependency_overrides.pop(get_eda_service, None)
        app.dependency_overrides.pop(get_cleaning_service, None)
        get_settings.cache_clear()
        _dataset_repository.cache_clear()
