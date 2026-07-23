"""Dataset endpoints (upload, inspection, ...)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Path, UploadFile, status

from app.api.dependencies import (
    get_chart_service,
    get_cleaning_service,
    get_dataset_service,
    get_eda_service,
)
from app.models.charts import DatasetChartsResponse
from app.models.cleaning import CleaningOptions, DatasetCleanResponse
from app.models.dataset import DatasetUploadResponse
from app.models.eda import DatasetSummaryResponse
from app.services.chart_service import ChartService
from app.services.cleaning_service import CleaningService
from app.services.dataset_service import DatasetService
from app.services.eda_service import EDAService

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a CSV dataset",
)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV file to upload."),
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetUploadResponse:
    """Accept a CSV file, persist it, and return dataset metadata."""
    content = await file.read()
    filename = file.filename or "unnamed.csv"
    return service.upload(filename=filename, content=content)


@router.get(
    "/{dataset_id}/summary",
    response_model=DatasetSummaryResponse,
    summary="Exploratory summary for a stored dataset",
)
async def get_dataset_summary(
    dataset_id: UUID = Path(..., description="Server-generated dataset identifier."),
    service: EDAService = Depends(get_eda_service),
) -> DatasetSummaryResponse:
    """Return descriptive statistics for the stored dataset."""
    return service.summarize(dataset_id)


@router.post(
    "/{dataset_id}/clean",
    response_model=DatasetCleanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Clean a dataset and store the result as a new version",
)
async def clean_dataset(
    dataset_id: UUID = Path(..., description="Source dataset identifier."),
    options: CleaningOptions | None = None,
    service: CleaningService = Depends(get_cleaning_service),
) -> DatasetCleanResponse:
    """Run the cleaning pipeline and return the report plus the new id.

    The original dataset is left untouched; the cleaned version is
    stored under a fresh dataset id and can be queried by any of the
    other dataset endpoints.
    """
    return service.clean(dataset_id, options or CleaningOptions())


@router.get(
    "/{dataset_id}/charts",
    response_model=DatasetChartsResponse,
    summary="Generate exploratory charts for a stored dataset",
)
async def get_dataset_charts(
    dataset_id: UUID = Path(..., description="Source dataset identifier."),
    service: ChartService = Depends(get_chart_service),
) -> DatasetChartsResponse:
    """Render histograms, boxplots, correlation heatmap, bar charts,
    and category distributions in PNG (matplotlib) and HTML (plotly).
    """
    return service.generate(dataset_id)
