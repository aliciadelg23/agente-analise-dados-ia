"""Dataset endpoints (upload, inspection, ...)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.dependencies import get_dataset_service
from app.models.dataset import DatasetUploadResponse
from app.services.dataset_service import DatasetService

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
