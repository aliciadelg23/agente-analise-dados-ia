"""Global exception handlers wired into the FastAPI application.

Keeps error responses in a single JSON shape so clients can rely on
the same structure regardless of the failure mode.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _error_payload(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning(
        "%s while handling %s %s: %s",
        exc.code,
        request.method,
        request.url.path,
        exc.message,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(exc.code, exc.message),
    )


async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info(
        "validation_error while handling %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request payload failed validation.",
                "details": exc.errors(),
            }
        },
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error while handling %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=_error_payload("internal_error", "Unexpected error."),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the given app."""
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
