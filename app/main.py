"""FastAPI application entrypoint.

Uses the application-factory pattern (``create_app``) so tests and
alternative deployments can build a fresh instance with overrides.
"""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.api.routes import health
from app.config.settings import get_settings
from app.core.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    logger = get_logger(__name__)
    logger.info("Starting application in env=%s", settings.app_env)

    app = FastAPI(
        title="Agente de Analise de Dados com IA",
        version=__version__,
        description="Plataforma de analise de dados orquestrada por agentes de IA.",
    )
    app.include_router(health.router)
    return app


app = create_app()
