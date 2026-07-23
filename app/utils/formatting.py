"""Small formatting helpers shared across layers."""

from __future__ import annotations

_UNITS = ("B", "KB", "MB", "GB", "TB")


def human_readable_size(num_bytes: int) -> str:
    """Return a human-friendly representation of a byte count.

    Uses binary units (1 KB = 1024 B). Bytes are printed as integers,
    larger units keep one decimal place.
    """
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")

    size = float(num_bytes)
    for unit in _UNITS:
        if size < 1024 or unit == _UNITS[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} {_UNITS[-1]}"
