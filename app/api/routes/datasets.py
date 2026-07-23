"""Dataset endpoints (upload, inspection, ...)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Path, UploadFile, status

from app.api.dependencies import get_dataset_service, get_eda_service
from app.models.dataset import DatasetUploadResponse
from app.models.eda import DatasetSummaryResponse
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
