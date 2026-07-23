"""Application settings loaded from environment variables.

Uses pydantic-settings so values can come from process environment,
an ``.env`` file, or be overridden explicitly in tests via
``get_settings.cache_clear()``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings validated at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = Field(default="development", description="Runtime environment name.")
    app_host: str = Field(default="0.0.0.0", description="Bind host for the HTTP server.")
    app_port: int = Field(default=8000, description="Bind port for the HTTP server.")
    log_level: str = Field(default="INFO", description="Root logger level.")

    storage_dir: str = Field(
        default="storage",
        description="Base directory (relative to project root) for persisted artifacts.",
    )
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        description="Maximum accepted upload size in megabytes.",
    )

    charts_static_url_prefix: str = Field(
        default="/static/charts",
        description="URL prefix under which generated charts are served.",
    )
    models_dir_name: str = Field(
        default="models",
        description="Subdirectory under storage_dir where trained models are persisted.",
    )

    default_llm_provider: str = Field(
        default="openai",
        description="Provider used when callers do not specify one: openai, anthropic, or gemini.",
    )
    openai_api_key: str | None = Field(default=None, description="OpenAI API key.")
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Default model for the OpenAI provider.",
    )
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key.")
    anthropic_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Default model for the Anthropic provider.",
    )
    gemini_api_key: str | None = Field(default=None, description="Google Gemini API key.")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Default model for the Gemini provider.",
    )

    chromadb_dir_name: str = Field(
        default="chromadb",
        description="Subdirectory under storage_dir where ChromaDB persists its data.",
    )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Tests can reset the cache by calling ``get_settings.cache_clear()``.
    """
    return Settings()
