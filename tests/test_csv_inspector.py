"""Unit tests for the CSV inspection helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import EmptyFileError, InvalidCsvError
from app.utils.csv_inspector import detect_separator, inspect_csv


def _write(path: Path, content: str, encoding: str = "utf-8") -> Path:
    path.write_text(content, encoding=encoding)
    return path


def test_inspect_csv_with_comma_separator(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "sample.csv", "a,b,c\n1,2,3\n4,5,6\n")

    metadata = inspect_csv(csv_path)

    assert metadata.rows == 2
    assert metadata.column_count == 3
    assert metadata.separator == ","
    assert {c.name for c in metadata.columns} == {"a", "b", "c"}


def test_inspect_csv_detects_semicolon_separator(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "sample.csv", "a;b;c\n1;2;3\n4;5;6\n")

    metadata = inspect_csv(csv_path)

    assert metadata.separator == ";"
    assert metadata.column_count == 3
    assert metadata.rows == 2


def test_inspect_csv_raises_on_empty_file(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "empty.csv", "")

    with pytest.raises(EmptyFileError):
        inspect_csv(csv_path)


def test_inspect_csv_raises_on_garbage_content(tmp_path: Path) -> None:
    binary = tmp_path / "binary.csv"
    binary.write_bytes(b"\x00\x01\x02\x03\x04\x05")

    with pytest.raises((InvalidCsvError, EmptyFileError)):
        inspect_csv(binary)


def test_detect_separator_falls_back_to_comma_on_ambiguous_input() -> None:
    assert detect_separator("single_line_no_separator") == ","
