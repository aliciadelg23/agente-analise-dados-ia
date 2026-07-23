"""Reusable FastAPI dependencies.

Keeps wiring in one place so routes can request a ready-to-use
service without knowing which repository or settings back it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Depends

from app.config.settings import Settings, get_settings
from app.repositories.dataset_repository import DatasetRepository
from app.services.cleaning_service import CleaningService
from app.services.dataset_service import DatasetService
from app.services.eda_service import EDAService


@lru_cache(maxsize=1)
def _dataset_repository(base_dir: str) -> DatasetRepository:
    repo = DatasetRepository(Path(base_dir))
    repo.ensure_ready()
    return repo


def get_dataset_service(
    settings: Settings = Depends(get_settings),
) -> DatasetService:
    """Return a DatasetService bound to the current settings."""
    repository = _dataset_repository(settings.storage_dir)
    return DatasetService(
        repository=repository,
        max_size_bytes=settings.max_upload_size_bytes,
    )


def get_eda_service(
    settings: Settings = Depends(get_settings),
) -> EDAService:
    """Return an EDAService bound to the current settings."""
    repository = _dataset_repository(settings.storage_dir)
    return EDAService(repository=repository)


def get_cleaning_service(
    settings: Settings = Depends(get_settings),
) -> CleaningService:
    """Return a CleaningService bound to the current settings."""
    repository = _dataset_repository(settings.storage_dir)
    return CleaningService(repository=repository)
