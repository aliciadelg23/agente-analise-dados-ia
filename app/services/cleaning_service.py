"""Dataset cleaning service.

Applies a configurable pipeline of cleaning operations to a stored
dataset and persists the result under a new dataset id. The original
file is never modified.

Order of operations (each is opt-in via CleaningOptions):

1. Standardize column names to snake_case.
2. Strip whitespace from string columns; empty strings become nulls.
3. Drop rows that are fully empty.
4. Drop duplicated rows.
5. Attempt automatic type conversion (numeric, then datetime).
6. Fill remaining nulls (median for numeric, mode for categorical).
"""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from app.core.exceptions import DatasetNotFoundError
from app.core.logging import get_logger
from app.models.cleaning import (
    CleaningOptions,
    CleaningReport,
    DatasetCleanResponse,
    TypeConversion,
)
from app.repositories.dataset_repository import DatasetRepository
from app.utils.csv_inspector import detect_encoding, detect_separator

logger = get_logger(__name__)

_STRING_DTYPES: tuple[str, ...] = ("object", "str", "string")


class CleaningService:
    """Clean a stored dataset and persist the result as a new version."""

    def __init__(self, repository: DatasetRepository) -> None:
        self._repository = repository

    def clean(self, dataset_id: UUID, options: CleaningOptions) -> DatasetCleanResponse:
        """Run the cleaning pipeline and return the change report."""
        path = self._repository.find(dataset_id)
        if path is None:
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' was not found.")

        frame = self._load_dataframe(path)

        report = CleaningReport(
            rows_before=len(frame),
            rows_after=len(frame),
            rows_removed=0,
            duplicates_removed=0,
            empty_rows_removed=0,
        )
        applied: list[str] = []

        if options.standardize_column_names:
            mapping = self._standardize_column_names(frame)
            if mapping:
                report.columns_renamed = mapping
            applied.append("standardize_column_names")

        if options.strip_whitespace:
            stripped = self._strip_whitespace(frame)
            report.whitespace_stripped_columns = stripped
            applied.append("strip_whitespace")

        if options.remove_empty_rows:
            before = len(frame)
            frame = frame.dropna(how="all").reset_index(drop=True)
            report.empty_rows_removed = before - len(frame)
            applied.append("remove_empty_rows")

        if options.remove_duplicates:
            before = len(frame)
            frame = frame.drop_duplicates().reset_index(drop=True)
            report.duplicates_removed = before - len(frame)
            applied.append("remove_duplicates")

        if options.convert_types:
            conversions = self._convert_types(frame)
            if conversions:
                report.types_converted = conversions
            applied.append("convert_types")

        if options.fill_nulls:
            filled = self._fill_nulls(frame)
            if filled:
                report.nulls_filled = filled
            applied.append("fill_nulls")

        report.rows_after = len(frame)
        report.rows_removed = report.rows_before - report.rows_after
        report.operations_applied = applied

        cleaned_id = self._persist(frame)
        logger.info(
            "Cleaned dataset %s into %s (rows %d -> %d)",
            dataset_id,
            cleaned_id,
            report.rows_before,
            report.rows_after,
        )
        return DatasetCleanResponse(
            original_dataset_id=dataset_id,
            cleaned_dataset_id=cleaned_id,
            report=report,
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

    def _persist(self, frame: pd.DataFrame) -> UUID:
        new_id = uuid4()
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        self._repository.save(new_id, "cleaned.csv", buffer.getvalue().encode("utf-8"))
        return new_id

    def _standardize_column_names(self, frame: pd.DataFrame) -> dict[str, str]:
        mapping: dict[str, str] = {}
        used: set[str] = set()
        for original in frame.columns:
            slug = _slugify(str(original))
            candidate = slug or "column"
            counter = 1
            while candidate in used:
                counter += 1
                candidate = f"{slug or 'column'}_{counter}"
            used.add(candidate)
            mapping[str(original)] = candidate

        frame.rename(columns=mapping, inplace=True)
        return {original: new for original, new in mapping.items() if original != new}

    def _strip_whitespace(self, frame: pd.DataFrame) -> list[str]:
        touched: list[str] = []
        for column in frame.columns:
            series = frame[column]
            if series.dtype.name not in _STRING_DTYPES:
                continue
            trimmed = (
                series.astype("object")
                .where(series.notna(), None)
                .map(lambda value: value.strip() if isinstance(value, str) else value)
            )
            emptied = trimmed.map(lambda value: np.nan if value == "" else value)
            if not emptied.equals(series):
                frame[column] = emptied
                touched.append(str(column))
        return touched

    def _convert_types(self, frame: pd.DataFrame) -> dict[str, TypeConversion]:
        conversions: dict[str, TypeConversion] = {}
        for column in frame.columns:
            series = frame[column]
            if series.dtype.name not in _STRING_DTYPES:
                continue
            non_null = series.dropna()
            if non_null.empty:
                continue

            numeric = pd.to_numeric(non_null, errors="coerce")
            if numeric.notna().all():
                new_series = pd.to_numeric(series, errors="coerce")
                conversions[str(column)] = TypeConversion(
                    before=series.dtype.name, after=new_series.dtype.name
                )
                frame[column] = new_series
                continue

            try:
                dates = pd.to_datetime(non_null, errors="coerce")
            except (ValueError, TypeError):
                dates = pd.Series([pd.NaT] * len(non_null))
            if dates.notna().all():
                new_series = pd.to_datetime(series, errors="coerce")
                conversions[str(column)] = TypeConversion(
                    before=series.dtype.name, after=new_series.dtype.name
                )
                frame[column] = new_series
        return conversions

    def _fill_nulls(self, frame: pd.DataFrame) -> dict[str, int]:
        filled: dict[str, int] = {}
        for column in frame.columns:
            series = frame[column]
            missing = int(series.isna().sum())
            if missing == 0:
                continue

            fill_value = self._fill_value_for(series)
            if fill_value is None:
                continue
            frame[column] = series.fillna(fill_value)
            filled[str(column)] = missing
        return filled

    @staticmethod
    def _fill_value_for(series: pd.Series) -> object | None:
        if series.isna().all():
            return None
        if pd.api.types.is_numeric_dtype(series):
            return float(series.median())
        if pd.api.types.is_datetime64_any_dtype(series):
            return series.dropna().iloc[0]
        mode = series.dropna().mode()
        if mode.empty:
            return None
        return mode.iloc[0]


_ACCENT_PATTERN = re.compile(r"[^\w]+")


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower().strip()
    replaced = _ACCENT_PATTERN.sub("_", lowered)
    return replaced.strip("_")
