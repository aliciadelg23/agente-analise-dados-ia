"""Vector-database endpoints (index, query)."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi import Path as FastAPIPath

from app.api.dependencies import (
    get_eda_service,
    get_vector_index_service,
)
from app.config.settings import Settings, get_settings
from app.models.vector import (
    VectorIndexResponse,
    VectorMatch,
    VectorQueryRequest,
    VectorQueryResponse,
)
from app.repositories.model_repository import ModelRepository
from app.services.eda_service import EDAService
from app.services.vector_index_service import VectorIndexService

router = APIRouter(prefix="/vector", tags=["vector"])


@router.post(
    "/index/{dataset_id}",
    response_model=VectorIndexResponse,
    summary="Index EDA and known model manifests for a dataset",
)
async def index_dataset(
    dataset_id: UUID = FastAPIPath(..., description="Source dataset identifier."),
    eda_service: EDAService = Depends(get_eda_service),
    vector_service: VectorIndexService = Depends(get_vector_index_service),
    settings: Settings = Depends(get_settings),
) -> VectorIndexResponse:
    """Compute the EDA for a dataset and index any models that reference it."""
    summary = eda_service.summarize(dataset_id)
    vector_service.index_eda(dataset_id, summary.model_dump(mode="json"))

    models_dir = Path(settings.storage_dir) / settings.models_dir_name
    model_repo = ModelRepository(models_dir)
    indexed_models = 0
    if models_dir.exists():
        for manifest_path in models_dir.glob("*.json"):
            manifest = model_repo.read_manifest(UUID(manifest_path.stem))
            if manifest.get("dataset_id") == str(dataset_id):
                vector_service.index_model(UUID(manifest_path.stem), manifest)
                indexed_models += 1

    return VectorIndexResponse(
        dataset_id=dataset_id,
        indexed={"eda": True, "models": indexed_models > 0},
    )


@router.post(
    "/query",
    response_model=VectorQueryResponse,
    summary="Search the vector store for past dataset artifacts",
)
async def query(
    request: VectorQueryRequest,
    vector_service: VectorIndexService = Depends(get_vector_index_service),
) -> VectorQueryResponse:
    """Return the top-k matches for ``query`` across the configured collections."""
    matches = vector_service.query(
        request.query, top_k=request.top_k, type_filter=request.type_filter
    )
    return VectorQueryResponse(
        matches=[
            VectorMatch(
                collection=item["collection"],
                item_id=item["item_id"],
                document=item["document"],
                metadata=item["metadata"],
                distance=item["distance"],
            )
            for item in matches
        ]
    )
