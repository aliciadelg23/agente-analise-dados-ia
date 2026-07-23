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
from app.repositories.model_repository import ModelRepository
from app.repositories.vector_repository import VectorRepository
from app.services.ai_insight_service import AIInsightService
from app.services.chart_service import ChartService
from app.services.cleaning_service import CleaningService
from app.services.dataset_service import DatasetService
from app.services.eda_service import EDAService
from app.services.explainability_service import ExplainabilityService
from app.services.ml_pipeline_service import MLPipelineService
from app.services.vector_index_service import VectorIndexService


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


def get_chart_service(
    settings: Settings = Depends(get_settings),
) -> ChartService:
    """Return a ChartService bound to the current settings."""
    repository = _dataset_repository(settings.storage_dir)
    charts_dir = Path(settings.storage_dir) / "charts"
    return ChartService(
        repository=repository,
        charts_dir=charts_dir,
        url_prefix=settings.charts_static_url_prefix,
    )


@lru_cache(maxsize=1)
def _model_repository(base_dir: str, subdir: str) -> ModelRepository:
    repo = ModelRepository(Path(base_dir) / subdir)
    repo.ensure_ready()
    return repo


def get_ml_pipeline_service(
    settings: Settings = Depends(get_settings),
) -> MLPipelineService:
    """Return an MLPipelineService bound to the current settings."""
    dataset_repo = _dataset_repository(settings.storage_dir)
    model_repo = _model_repository(settings.storage_dir, settings.models_dir_name)
    return MLPipelineService(dataset_repository=dataset_repo, model_repository=model_repo)


def get_explainability_service(
    settings: Settings = Depends(get_settings),
) -> ExplainabilityService:
    """Return an ExplainabilityService bound to the current settings."""
    dataset_repo = _dataset_repository(settings.storage_dir)
    model_repo = _model_repository(settings.storage_dir, settings.models_dir_name)
    charts_dir = Path(settings.storage_dir) / "charts"
    return ExplainabilityService(
        dataset_repository=dataset_repo,
        model_repository=model_repo,
        charts_dir=charts_dir,
        url_prefix=settings.charts_static_url_prefix,
    )


def get_ai_insight_service(
    eda: EDAService = Depends(get_eda_service),
) -> AIInsightService:
    """Return an AIInsightService bound to the current settings."""
    return AIInsightService(eda_service=eda)


@lru_cache(maxsize=1)
def _vector_repository(base_dir: str, subdir: str) -> VectorRepository:
    return VectorRepository(Path(base_dir) / subdir)


def get_vector_index_service(
    settings: Settings = Depends(get_settings),
) -> VectorIndexService:
    """Return a VectorIndexService bound to the current settings."""
    repository = _vector_repository(settings.storage_dir, settings.chromadb_dir_name)
    return VectorIndexService(repository=repository)
