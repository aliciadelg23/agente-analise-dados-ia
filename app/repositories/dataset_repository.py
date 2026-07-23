"""Filesystem-backed dataset repository.

Persists uploaded files under ``<base_dir>/uploads/<dataset_id><ext>``.
The service layer depends on this concrete class today; a future
migration to object storage should introduce a Protocol and swap the
implementation without touching services.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID


class DatasetRepository:
    """Store and locate uploaded datasets on the local filesystem."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._uploads_dir = base_dir / "uploads"

    @property
    def uploads_dir(self) -> Path:
        return self._uploads_dir

    def ensure_ready(self) -> None:
        """Create the uploads directory if it does not exist."""
        self._uploads_dir.mkdir(parents=True, exist_ok=True)

    def save(self, dataset_id: UUID, filename: str, content: bytes) -> Path:
        """Persist ``content`` and return the resulting path.

        The stored filename uses the dataset id plus the original
        extension, keeping filesystem paths predictable and avoiding
        collisions across uploads that share the same original name.
        """
        self.ensure_ready()
        extension = Path(filename).suffix.lower() or ".csv"
        target = self._uploads_dir / f"{dataset_id}{extension}"
        target.write_bytes(content)
        return target
