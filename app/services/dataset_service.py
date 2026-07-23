"""Dataset business logic.

Coordinates validation, persistence, and inspection so the API layer
stays thin. Raises domain exceptions (see ``app.core.exceptions``)
which are translated to HTTP responses by the global handlers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.core.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    InvalidFileExtensionError,
)
from app.core.logging import get_logger
from app.models.dataset import DatasetUploadResponse
from app.repositories.dataset_repository import DatasetRepository
from app.utils.csv_inspector import inspect_csv
from app.utils.formatting import human_readable_size

logger = get_logger(__name__)

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".csv"})


class DatasetService:
    """Orchestrate the CSV upload use case."""

    def __init__(self, repository: DatasetRepository, max_size_bytes: int) -> None:
        self._repository = repository
        self._max_size_bytes = max_size_bytes

    def upload(self, filename: str, content: bytes) -> DatasetUploadResponse:
        """Validate, persist, and inspect an uploaded CSV file."""
        safe_name = filename or "unnamed.csv"
        self._validate_extension(safe_name)
        self._validate_size(len(content))
        self._validate_not_empty(content)

        dataset_id: UUID = uuid4()
        stored_path = self._repository.save(dataset_id, safe_name, content)
        logger.info("Stored upload %s at %s", dataset_id, stored_path)

        metadata = inspect_csv(stored_path)

        return DatasetUploadResponse(
            dataset_id=dataset_id,
            filename=safe_name,
            rows=metadata.rows,
            columns=metadata.column_count,
            size=human_readable_size(len(content)),
            uploaded_at=datetime.now(timezone.utc),
            encoding=metadata.encoding,
            separator=metadata.separator,
        )

    def _validate_extension(self, filename: str) -> None:
        lowered = filename.lower()
        if not any(lowered.endswith(ext) for ext in _ALLOWED_EXTENSIONS):
            raise InvalidFileExtensionError(
                "Unsupported file extension. Allowed: "
                + ", ".join(sorted(_ALLOWED_EXTENSIONS))
            )

    def _validate_size(self, size_bytes: int) -> None:
        if size_bytes > self._max_size_bytes:
            limit_mb = self._max_size_bytes // (1024 * 1024)
            raise FileTooLargeError(f"File exceeds the {limit_mb} MB limit.")

    def _validate_not_empty(self, content: bytes) -> None:
        if len(content) == 0:
            raise EmptyFileError("Uploaded file is empty.")
