"""Unit tests for the LLM provider factory."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.config.settings import Settings
from app.core.exceptions import MissingCredentialsError, UnknownProviderError
from app.llms.anthropic_provider import AnthropicProvider
from app.llms.factory import available_providers, get_llm_provider
from app.llms.gemini_provider import GeminiProvider
from app.llms.openai_provider import OpenAIProvider


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "default_llm_provider": "openai",
        "openai_api_key": "sk-test",
        "anthropic_api_key": "anthropic-test",
        "gemini_api_key": "gemini-test",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_available_providers_lists_all_three() -> None:
    assert set(available_providers()) == {"openai", "anthropic", "gemini"}


def test_get_llm_provider_returns_openai_by_default() -> None:
    with patch("app.llms.openai_provider.OpenAI"):
        provider = get_llm_provider(settings=_settings())

    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "openai"


def test_get_llm_provider_returns_anthropic_when_requested() -> None:
    with patch("app.llms.anthropic_provider.Anthropic"):
        provider = get_llm_provider("anthropic", settings=_settings())

    assert isinstance(provider, AnthropicProvider)


def test_get_llm_provider_returns_gemini_when_requested() -> None:
    with patch("app.llms.gemini_provider.genai.configure"):
        provider = get_llm_provider("gemini", settings=_settings())

    assert isinstance(provider, GeminiProvider)


def test_get_llm_provider_is_case_insensitive() -> None:
    with patch("app.llms.openai_provider.OpenAI"):
        provider = get_llm_provider("OpenAI", settings=_settings())

    assert provider.name == "openai"


def test_get_llm_provider_raises_for_unknown_name() -> None:
    with pytest.raises(UnknownProviderError):
        get_llm_provider("cohere", settings=_settings())


def test_get_llm_provider_raises_when_key_missing() -> None:
    with pytest.raises(MissingCredentialsError):
        get_llm_provider("openai", settings=_settings(openai_api_key=None))
