"""AI-powered dataset insights service.

Runs EDA on a stored dataset, feeds the resulting statistics to an
LLM, and turns the LLM's structured response into a
``DatasetInsightsResponse``. The LLM is accessed exclusively through
the abstraction in ``app.llms`` so this service never talks to a
vendor SDK directly.
"""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import UUID

from app.core.logging import get_logger
from app.llms.base import LLMProvider, Message
from app.models.insights import DatasetInsightsResponse
from app.services.eda_service import EDAService

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior data analyst. You receive a JSON summary of an "
    "exploratory data analysis (EDA) and must produce structured "
    "insights. Reply with a single JSON object containing exactly "
    "these keys: executive_summary (string), insights (list of short "
    "strings), anomalies (list of short strings), suggestions (list "
    "of short strings), risks (list of short strings). Do not wrap "
    "the JSON in markdown code fences and do not include commentary "
    "outside the JSON object."
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class AIInsightService:
    """Turn EDA output into AI-authored insights."""

    def __init__(self, eda_service: EDAService) -> None:
        self._eda = eda_service

    def analyze(
        self,
        dataset_id: UUID,
        llm_provider: LLMProvider,
        model: str | None = None,
    ) -> DatasetInsightsResponse:
        """Run EDA, ask the LLM for structured insights, and return them."""
        summary = self._eda.summarize(dataset_id)
        summary_json = summary.model_dump_json()

        user_prompt = (
            "Analyze the following EDA summary and return the structured "
            "JSON response described in the system prompt.\n\n"
            f"EDA_SUMMARY_JSON:\n{summary_json}"
        )

        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]

        response = llm_provider.chat(messages, model=model)
        parsed = self._parse(response.content)

        return DatasetInsightsResponse(
            dataset_id=dataset_id,
            provider=response.provider,
            model=response.model,
            executive_summary=str(parsed.get("executive_summary", "") or ""),
            insights=_as_str_list(parsed.get("insights")),
            anomalies=_as_str_list(parsed.get("anomalies")),
            suggestions=_as_str_list(parsed.get("suggestions")),
            risks=_as_str_list(parsed.get("risks")),
            raw_llm_response=None if parsed else response.content,
        )

    def _parse(self, content: str) -> dict[str, Any]:
        """Best-effort JSON parse of the LLM output.

        Returns an empty dict when parsing fails so the caller can
        populate ``raw_llm_response`` for the client to inspect.
        """
        stripped = content.strip()
        if not stripped:
            return {}
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            match = _JSON_BLOCK_RE.search(stripped)
            if match is None:
                logger.warning("LLM response could not be parsed as JSON")
                return {}
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("LLM response contained a JSON-looking block that failed to parse")
                return {}
        return data if isinstance(data, dict) else {}


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
