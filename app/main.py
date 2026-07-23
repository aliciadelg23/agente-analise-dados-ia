"""FastAPI application entrypoint.

Uses the application-factory pattern (``create_app``) so tests and
alternative deployments can build a fresh instance with overrides.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes import agent, datasets, health, info, workflows
from app.api.routes import models as models_routes
from app.config.settings import get_settings
from app.core.exception_handlers import register_exception_handlers
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
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    register_exception_handlers(app)

    app.include_router(info.router)
    app.include_router(health.router)
    app.include_router(datasets.router)
    app.include_router(models_routes.router)
    app.include_router(agent.router)
    app.include_router(workflows.router)

    charts_dir = Path(settings.storage_dir) / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        settings.charts_static_url_prefix,
        StaticFiles(directory=charts_dir),
        name="charts",
    )

    return app


app = create_app()
