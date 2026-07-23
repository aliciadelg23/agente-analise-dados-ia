"""Common interface for LLM providers.

Every provider adapts a third-party SDK to a small, uniform API so
higher layers depend on this module instead of vendor-specific
clients. Keeping the surface small (a single ``chat`` method) makes
it cheap to add or replace providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """Single message exchanged with an LLM."""

    role: Role = Field(..., description="Message author role: system, user, or assistant.")
    content: str = Field(..., description="Message text.")


class LLMResponse(BaseModel):
    """Uniform response returned by every provider."""

    content: str = Field(..., description="Generated text.")
    model: str = Field(..., description="Model identifier that produced the response.")
    provider: str = Field(..., description="Provider name (openai, anthropic, gemini).")
    usage: dict[str, Any] | None = Field(
        default=None, description="Token usage info when the SDK exposes it."
    )


class LLMProvider(ABC):
    """Abstract provider contract.

    Concrete subclasses store their SDK client and translate messages
    to and from the vendor-specific format. They MUST NOT leak vendor
    types back to the caller.
    """

    name: str
    default_model: str

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send ``messages`` to the provider and return the response.

        ``model`` overrides the provider default. Extra keyword
        arguments are forwarded to the underlying SDK when supported.
        """
