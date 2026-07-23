"""Centralized logging configuration.

Kept intentionally small at this stage: a single configure function
that installs a consistent stdout handler on the root logger. Later
stages can extend this with structured JSON output, correlation
ids, or third-party integrations without touching call sites.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger.

    Safe to call more than once: ``force=True`` replaces any handlers
    that were installed by libraries during import.
    """
    logging.basicConfig(
        level=level.upper(),
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger for the given module."""
    return logging.getLogger(name)
