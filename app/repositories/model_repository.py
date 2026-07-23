"""Filesystem-backed model repository.

Persists trained sklearn pipelines to disk with joblib and stores
a JSON manifest alongside each artifact so callers can inspect
metadata (target column, features, algorithm) without loading the
pipeline into memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import joblib


class ModelRepository:
    """Save and locate trained models on the local filesystem."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def ensure_ready(self) -> None:
        """Create the models directory if it does not exist."""
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model_id: UUID,
        pipeline: object,
        manifest: dict[str, Any],
    ) -> Path:
        """Persist ``pipeline`` and its ``manifest``.

        Returns the path to the joblib artifact.
        """
        self.ensure_ready()
        joblib_path = self._base_dir / f"{model_id}.joblib"
        manifest_path = self._base_dir / f"{model_id}.json"
        joblib.dump(pipeline, joblib_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return joblib_path

    def find(self, model_id: UUID) -> Path | None:
        """Return the joblib path for ``model_id`` or None if missing."""
        candidate = self._base_dir / f"{model_id}.joblib"
        return candidate if candidate.exists() else None

    def load(self, model_id: UUID) -> object:
        """Load and return the persisted pipeline for ``model_id``.

        Raises FileNotFoundError when the artifact is missing so
        callers can translate to a domain-specific exception.

        Safety: only loads artifacts we wrote ourselves via ``save``
        under ``base_dir``; the path is derived from ``model_id`` and
        never crosses the trust boundary, so pickle deserialization
        stays inside server-controlled data.
        """
        path = self.find(model_id)
        if path is None:
            raise FileNotFoundError(f"Model '{model_id}' not found in {self._base_dir}")
        return joblib.load(path)

    def read_manifest(self, model_id: UUID) -> dict[str, Any]:
        """Return the JSON manifest for ``model_id``.

        Raises FileNotFoundError when the manifest is missing.
        """
        manifest_path = self._base_dir / f"{model_id}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest for model '{model_id}' not found")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
