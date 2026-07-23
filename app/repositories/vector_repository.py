"""ChromaDB-backed vector repository.

Wraps a single ``PersistentClient`` and exposes named collections
so callers depend on a stable interface instead of touching Chroma
primitives directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

_ALLOWED_COLLECTIONS = ("dataset_eda", "dataset_insights", "dataset_models", "dataset_reports")


class VectorRepository:
    """Persistent Chroma client with a fixed set of collections."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._client: ClientAPI | None = None

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def collection_names(self) -> tuple[str, ...]:
        return _ALLOWED_COLLECTIONS

    def _get_client(self) -> ClientAPI:
        if self._client is None:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._base_dir))
        return self._client

    def _get_collection(self, name: str) -> Collection:
        if name not in _ALLOWED_COLLECTIONS:
            raise ValueError(
                f"Unknown collection '{name}'. Allowed: {', '.join(_ALLOWED_COLLECTIONS)}"
            )
        return self._get_client().get_or_create_collection(name=name)

    def upsert(
        self,
        collection: str,
        *,
        item_id: str,
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        """Insert or update a single document in ``collection``."""
        target = self._get_collection(collection)
        target.upsert(ids=[item_id], documents=[document], metadatas=[metadata])

    def query(
        self,
        collections: list[str],
        *,
        query_text: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Run a similarity query across ``collections`` and merge the results."""
        matches: list[dict[str, Any]] = []
        for name in collections:
            target = self._get_collection(name)
            result = target.query(
                query_texts=[query_text], n_results=max(1, top_k), include=["documents", "metadatas", "distances"]
            )
            ids = (result.get("ids") or [[]])[0]
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            for item_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=False
            ):
                matches.append(
                    {
                        "collection": name,
                        "item_id": item_id,
                        "document": document,
                        "metadata": metadata or {},
                        "distance": float(distance),
                    }
                )
        matches.sort(key=lambda entry: entry["distance"])
        return matches[:top_k]

    def count(self, collection: str) -> int:
        """Return the total number of documents in ``collection``."""
        return int(self._get_collection(collection).count())
