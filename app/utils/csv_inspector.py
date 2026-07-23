"""CSV inspection helpers.

Given a raw CSV file, produce a ``DatasetMetadata`` with encoding,
separator, row count, and per-column dtype information. Detection
falls back to conservative defaults (utf-8, comma) so callers never
have to reason about ``None`` values.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from charset_normalizer import from_bytes

from app.core.exceptions import EmptyFileError, InvalidCsvError
from app.models.dataset import ColumnInfo, DatasetMetadata

_ENCODING_SAMPLE_BYTES = 64 * 1024
_SEPARATOR_SAMPLE_BYTES = 8 * 1024
_DEFAULT_SEPARATOR = ","
_DEFAULT_ENCODING = "utf-8"
_CANDIDATE_SEPARATORS = ",;|\t"


def detect_encoding(raw: bytes) -> str:
    """Detect the encoding of ``raw`` bytes.

    Returns 'utf-8' when detection is inconclusive.
    """
    if not raw:
        return _DEFAULT_ENCODING
    sample = raw[:_ENCODING_SAMPLE_BYTES]
    best = from_bytes(sample).best()
    if best is None or best.encoding is None:
        return _DEFAULT_ENCODING
    return str(best.encoding).lower()


def detect_separator(text_sample: str) -> str:
    """Detect the column separator of a CSV text sample.

    Falls back to comma when the sniffer cannot determine one.
    """
    if not text_sample.strip():
        return _DEFAULT_SEPARATOR
    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=_CANDIDATE_SEPARATORS)
        return dialect.delimiter
    except csv.Error:
        return _DEFAULT_SEPARATOR


def inspect_csv(path: Path) -> DatasetMetadata:
    """Inspect the CSV at ``path`` and return its metadata.

    Raises ``EmptyFileError`` when the file has no bytes or no data
    rows, and ``InvalidCsvError`` when the file cannot be parsed.
    """
    raw = path.read_bytes()
    if not raw:
        raise EmptyFileError("Uploaded file is empty.")

    if b"\x00" in raw[:_ENCODING_SAMPLE_BYTES]:
        raise InvalidCsvError("File contains null bytes and is not a text CSV.")

    encoding = detect_encoding(raw)
    try:
        text_sample = raw[:_SEPARATOR_SAMPLE_BYTES].decode(encoding, errors="replace")
    except LookupError:
        encoding = _DEFAULT_ENCODING
        text_sample = raw[:_SEPARATOR_SAMPLE_BYTES].decode(encoding, errors="replace")

    separator = detect_separator(text_sample)

    try:
        frame = pd.read_csv(path, sep=separator, encoding=encoding)
    except pd.errors.EmptyDataError as exc:
        raise EmptyFileError("CSV has no data rows.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise InvalidCsvError(f"Could not parse CSV file: {exc}") from exc

    if frame.empty and len(frame.columns) == 0:
        raise EmptyFileError("CSV has no columns or rows.")

    columns = tuple(ColumnInfo(name=str(col), dtype=str(frame[col].dtype)) for col in frame.columns)
    return DatasetMetadata(
        rows=len(frame),
        columns=columns,
        encoding=encoding,
        separator=separator,
    )
