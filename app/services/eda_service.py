"""Exploratory data analysis service.

Owns the statistical logic that turns a Pandas DataFrame into a
``DatasetSummaryResponse``. Kept independent of FastAPI so it can
be reused by CLI tools or background jobs without changes.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd

from app.core.exceptions import DatasetNotFoundError, EmptyFileError
from app.core.logging import get_logger
from app.models.eda import (
    CategoricalColumnStats,
    DatasetSummaryResponse,
    NumericColumnStats,
    TopValue,
)
from app.repositories.dataset_repository import DatasetRepository
from app.utils.csv_inspector import detect_encoding, detect_separator
from app.utils.formatting import human_readable_size

logger = get_logger(__name__)

_NUMERIC_KINDS: tuple[str, ...] = ("number",)
_CATEGORICAL_KINDS: tuple[str, ...] = ("object", "str", "category", "bool")


class EDAService:
    """Produce descriptive summaries for stored datasets."""

    def __init__(self, repository: DatasetRepository, top_values_limit: int = 5) -> None:
        self._repository = repository
        self._top_values_limit = top_values_limit

    def summarize(self, dataset_id: UUID) -> DatasetSummaryResponse:
        """Return the exploratory summary for the given dataset id."""
        path = self._repository.find(dataset_id)
        if path is None:
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' was not found.")

        frame = self._load_dataframe(path)
        if frame.empty and len(frame.columns) == 0:
            raise EmptyFileError("Stored dataset has no columns or rows.")

        logger.info(
            "Summarizing dataset %s (rows=%d, cols=%d)", dataset_id, len(frame), len(frame.columns)
        )

        numeric_frame = frame.select_dtypes(include=list(_NUMERIC_KINDS))
        categorical_frame = frame.select_dtypes(include=list(_CATEGORICAL_KINDS))

        return DatasetSummaryResponse(
            dataset_id=dataset_id,
            rows=len(frame),
            columns=len(frame.columns),
            memory=human_readable_size(int(frame.memory_usage(deep=True).sum())),
            duplicates=int(frame.duplicated().sum()),
            null_counts=self._null_counts(frame),
            null_percentages=self._null_percentages(frame),
            dtypes={str(col): str(frame[col].dtype) for col in frame.columns},
            numeric_columns=[str(c) for c in numeric_frame.columns],
            categorical_columns=[str(c) for c in categorical_frame.columns],
            numeric_stats=self._numeric_stats(numeric_frame),
            categorical_stats=self._categorical_stats(categorical_frame),
        )

    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        raw = path.read_bytes()
        encoding = detect_encoding(raw)
        try:
            sample = raw[:8192].decode(encoding, errors="replace")
        except LookupError:
            encoding = "utf-8"
            sample = raw[:8192].decode(encoding, errors="replace")
        separator = detect_separator(sample)
        return pd.read_csv(path, sep=separator, encoding=encoding)

    @staticmethod
    def _null_counts(frame: pd.DataFrame) -> dict[str, int]:
        return {str(col): int(frame[col].isna().sum()) for col in frame.columns}

    @staticmethod
    def _null_percentages(frame: pd.DataFrame) -> dict[str, float]:
        total = len(frame)
        if total == 0:
            return {str(col): 0.0 for col in frame.columns}
        return {
            str(col): round(float(frame[col].isna().sum()) / total * 100, 2)
            for col in frame.columns
        }

    @staticmethod
    def _numeric_stats(frame: pd.DataFrame) -> dict[str, NumericColumnStats]:
        stats: dict[str, NumericColumnStats] = {}
        for column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            if series.empty:
                stats[str(column)] = NumericColumnStats(
                    mean=0.0, median=0.0, min=0.0, max=0.0, std=0.0, q25=0.0, q50=0.0, q75=0.0
                )
                continue
            quartiles = np.quantile(series.to_numpy(), [0.25, 0.5, 0.75])
            stats[str(column)] = NumericColumnStats(
                mean=float(series.mean()),
                median=float(series.median()),
                min=float(series.min()),
                max=float(series.max()),
                std=float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                q25=float(quartiles[0]),
                q50=float(quartiles[1]),
                q75=float(quartiles[2]),
            )
        return stats

    def _categorical_stats(self, frame: pd.DataFrame) -> dict[str, CategoricalColumnStats]:
        stats: dict[str, CategoricalColumnStats] = {}
        for column in frame.columns:
            series = frame[column]
            counts = series.dropna().value_counts().head(self._top_values_limit)
            top_values = [
                TopValue(value=str(value), count=int(count)) for value, count in counts.items()
            ]
            stats[str(column)] = CategoricalColumnStats(
                unique_count=int(series.nunique(dropna=True)),
                top_values=top_values,
            )
        return stats
