"""LLM provider factory.

Resolves a provider name to a concrete instance backed by the
current settings. Callers depend on this function so switching
providers is a matter of changing the env variable, not the code.
"""

from __future__ import annotations

from collections.abc import Callable

from app.config.settings import Settings, get_settings
from app.core.exceptions import UnknownProviderError
from app.llms.anthropic_provider import AnthropicProvider
from app.llms.base import LLMProvider
from app.llms.gemini_provider import GeminiProvider
from app.llms.openai_provider import OpenAIProvider


def _build_openai(settings: Settings) -> LLMProvider:
    return OpenAIProvider(api_key=settings.openai_api_key, default_model=settings.openai_model)


def _build_anthropic(settings: Settings) -> LLMProvider:
    return AnthropicProvider(
        api_key=settings.anthropic_api_key,
        default_model=settings.anthropic_model,
    )


def _build_gemini(settings: Settings) -> LLMProvider:
    return GeminiProvider(api_key=settings.gemini_api_key, default_model=settings.gemini_model)


_BUILDERS: dict[str, Callable[[Settings], LLMProvider]] = {
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "gemini": _build_gemini,
}


def available_providers() -> tuple[str, ...]:
    """Return the tuple of provider names the factory knows about."""
    return tuple(_BUILDERS)


def get_llm_provider(
    name: str | None = None,
    settings: Settings | None = None,
) -> LLMProvider:
    """Return an LLM provider instance.

    ``name`` overrides the default provider from settings. Raises
    UnknownProviderError for unregistered names and delegates
    credential validation to the provider's own constructor.
    """
    resolved_settings = settings or get_settings()
    provider_name = (name or resolved_settings.default_llm_provider).lower()
    builder = _BUILDERS.get(provider_name)
    if builder is None:
        raise UnknownProviderError(
            f"Unknown LLM provider '{provider_name}'. Available: {', '.join(sorted(_BUILDERS))}."
        )
    return builder(resolved_settings)
