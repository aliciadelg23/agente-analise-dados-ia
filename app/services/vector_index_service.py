"""Vector index service.

Translates domain payloads (EDA summary, insights, model manifest,
final workflow report) into text documents and stores them in the
Chroma-backed vector repository. The API and workflow layers call
into this service so the vector store stays in sync as new
artifacts are produced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.core.logging import get_logger
from app.repositories.vector_repository import VectorRepository

logger = get_logger(__name__)


class VectorIndexService:
    """Index domain artifacts and query the vector store."""

    def __init__(self, repository: VectorRepository) -> None:
        self._repository = repository

    def index_eda(self, dataset_id: UUID, summary: dict[str, Any]) -> None:
        """Index an EDA summary under the dataset id."""
        document = self._eda_to_text(summary)
        metadata = {
            "dataset_id": str(dataset_id),
            "type": "eda",
            "created_at": _now_iso(),
            "rows": int(summary.get("rows", 0)),
            "columns": int(summary.get("columns", 0)),
        }
        self._repository.upsert(
            "dataset_eda",
            item_id=str(dataset_id),
            document=document,
            metadata=metadata,
        )
        logger.info("Indexed EDA for dataset %s", dataset_id)

    def index_insights(self, dataset_id: UUID, insights: dict[str, Any]) -> None:
        """Index an AI insights payload under the dataset id."""
        document = self._insights_to_text(insights)
        metadata = {
            "dataset_id": str(dataset_id),
            "type": "insights",
            "created_at": _now_iso(),
            "provider": str(insights.get("provider", "")),
        }
        self._repository.upsert(
            "dataset_insights",
            item_id=str(dataset_id),
            document=document,
            metadata=metadata,
        )
        logger.info("Indexed insights for dataset %s", dataset_id)

    def index_model(self, model_id: UUID, manifest: dict[str, Any]) -> None:
        """Index a trained-model manifest under the model id."""
        document = self._model_to_text(manifest)
        metadata = {
            "model_id": str(model_id),
            "dataset_id": str(manifest.get("dataset_id", "")),
            "type": "model",
            "created_at": _now_iso(),
            "algorithm": str(manifest.get("chosen_algorithm", "")),
            "problem_type": str(manifest.get("problem_type", "")),
        }
        self._repository.upsert(
            "dataset_models",
            item_id=str(model_id),
            document=document,
            metadata=metadata,
        )
        logger.info("Indexed model %s", model_id)

    def index_report(self, dataset_id: UUID, report_text: str) -> None:
        """Index a workflow report; the id is dataset_id plus a timestamp."""
        timestamp = _now_iso()
        item_id = f"{dataset_id}:{timestamp}"
        metadata = {
            "dataset_id": str(dataset_id),
            "type": "report",
            "created_at": timestamp,
        }
        self._repository.upsert(
            "dataset_reports",
            item_id=item_id,
            document=report_text or "",
            metadata=metadata,
        )
        logger.info("Indexed report for dataset %s", dataset_id)

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a similarity query across the configured collections."""
        collections = self._collections_for_filter(type_filter)
        return self._repository.query(collections, query_text=query_text, top_k=top_k)

    def _collections_for_filter(self, type_filter: str | None) -> list[str]:
        if type_filter is None:
            return list(self._repository.collection_names)
        mapping = {
            "eda": "dataset_eda",
            "insights": "dataset_insights",
            "model": "dataset_models",
            "report": "dataset_reports",
        }
        collection = mapping.get(type_filter)
        if collection is None:
            raise ValueError(f"Unknown type '{type_filter}'. Allowed: {', '.join(sorted(mapping))}")
        return [collection]

    @staticmethod
    def _eda_to_text(summary: dict[str, Any]) -> str:
        parts: list[str] = []
        parts.append(
            f"Dataset {summary.get('dataset_id', '')} has {summary.get('rows', 0)} rows "
            f"and {summary.get('columns', 0)} columns."
        )
        parts.append(f"Memory footprint: {summary.get('memory', 'unknown')}.")
        parts.append(f"Duplicated rows: {summary.get('duplicates', 0)}.")
        numeric = summary.get("numeric_columns") or []
        categorical = summary.get("categorical_columns") or []
        if numeric:
            parts.append("Numeric columns: " + ", ".join(map(str, numeric)) + ".")
        if categorical:
            parts.append("Categorical columns: " + ", ".join(map(str, categorical)) + ".")
        null_percentages = summary.get("null_percentages") or {}
        heavy_nulls = [f"{col} ({pct:.1f}%)" for col, pct in null_percentages.items() if pct > 10]
        if heavy_nulls:
            parts.append("Columns with heavy nulls: " + ", ".join(heavy_nulls) + ".")
        return " ".join(parts)

    @staticmethod
    def _insights_to_text(insights: dict[str, Any]) -> str:
        parts: list[str] = []
        summary = insights.get("executive_summary")
        if summary:
            parts.append(str(summary))
        for label in ("insights", "anomalies", "suggestions", "risks"):
            items = insights.get(label) or []
            if items:
                parts.append(f"{label.title()}: " + " | ".join(str(item) for item in items))
        return " ".join(parts) or "No insights available."

    @staticmethod
    def _model_to_text(manifest: dict[str, Any]) -> str:
        features = manifest.get("features") or []
        return (
            f"Model {manifest.get('model_id', '')} trained on dataset "
            f"{manifest.get('dataset_id', '')} for {manifest.get('problem_type', '')} "
            f"using {manifest.get('chosen_algorithm', '')}. Target column: "
            f"{manifest.get('target_column', '')}. Features: "
            + ", ".join(map(str, features))
            + f". CV mean score: {manifest.get('cv_score_mean', 'n/a')}."
        )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
