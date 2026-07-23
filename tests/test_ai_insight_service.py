"""Unit tests for AIInsightService."""

from __future__ import annotations

import json
import random
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import DatasetNotFoundError
from app.llms.base import LLMResponse
from app.repositories.dataset_repository import DatasetRepository
from app.services.ai_insight_service import AIInsightService
from app.services.eda_service import EDAService


def _store_csv(repository: DatasetRepository) -> UUID:
    rng = random.Random(42)
    rows = ["age,salary,city"]
    for _ in range(120):
        rows.append(
            f"{rng.randint(18, 70)},{rng.randint(1000, 10000)},"
            f"{rng.choice(['Lisbon', 'Porto', 'Braga'])}"
        )
    dataset_id = uuid4()
    repository.save(dataset_id, "insight.csv", ("\n".join(rows) + "\n").encode("utf-8"))
    return dataset_id


def _canonical_llm_payload() -> str:
    return json.dumps(
        {
            "executive_summary": "Dataset covers demographic and salary features.",
            "insights": ["Age median is 44.", "Salary is roughly uniform."],
            "anomalies": ["No obvious outliers detected."],
            "suggestions": ["Consider adding a churn column."],
            "risks": ["Small sample size may limit conclusions."],
        }
    )


def _mock_provider(response_content: str) -> MagicMock:
    provider = MagicMock()
    provider.chat.return_value = LLMResponse(
        content=response_content,
        model="gpt-4o-mini",
        provider="openai",
        usage={"total_tokens": 42},
    )
    return provider


def test_analyze_parses_clean_json_response(
    dataset_repository: DatasetRepository, eda_service: EDAService
) -> None:
    dataset_id = _store_csv(dataset_repository)
    service = AIInsightService(eda_service=eda_service)
    provider = _mock_provider(_canonical_llm_payload())

    result = service.analyze(dataset_id, provider)

    assert result.dataset_id == dataset_id
    assert result.provider == "openai"
    assert result.model == "gpt-4o-mini"
    assert result.executive_summary.startswith("Dataset")
    assert len(result.insights) == 2
    assert result.anomalies == ["No obvious outliers detected."]
    assert result.suggestions == ["Consider adding a churn column."]
    assert result.risks == ["Small sample size may limit conclusions."]
    assert result.raw_llm_response is None


def test_analyze_extracts_json_block_from_noisy_response(
    dataset_repository: DatasetRepository, eda_service: EDAService
) -> None:
    dataset_id = _store_csv(dataset_repository)
    service = AIInsightService(eda_service=eda_service)
    noisy = (
        "Sure! Here is the analysis:\n\n```json\n"
        + _canonical_llm_payload()
        + "\n```\nHope that helps."
    )
    provider = _mock_provider(noisy)

    result = service.analyze(dataset_id, provider)

    assert result.executive_summary.startswith("Dataset")
    assert result.raw_llm_response is None


def test_analyze_returns_raw_content_when_response_is_not_json(
    dataset_repository: DatasetRepository, eda_service: EDAService
) -> None:
    dataset_id = _store_csv(dataset_repository)
    service = AIInsightService(eda_service=eda_service)
    provider = _mock_provider("The dataset looks fine, no JSON here.")

    result = service.analyze(dataset_id, provider)

    assert result.executive_summary == ""
    assert result.insights == []
    assert result.anomalies == []
    assert result.suggestions == []
    assert result.risks == []
    assert result.raw_llm_response == "The dataset looks fine, no JSON here."


def test_analyze_forwards_model_override_to_provider(
    dataset_repository: DatasetRepository, eda_service: EDAService
) -> None:
    dataset_id = _store_csv(dataset_repository)
    service = AIInsightService(eda_service=eda_service)
    provider = _mock_provider(_canonical_llm_payload())

    service.analyze(dataset_id, provider, model="gpt-5.1")

    provider.chat.assert_called_once()
    call_kwargs = provider.chat.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5.1"


def test_analyze_sends_eda_summary_in_user_prompt(
    dataset_repository: DatasetRepository, eda_service: EDAService
) -> None:
    dataset_id = _store_csv(dataset_repository)
    service = AIInsightService(eda_service=eda_service)
    provider = _mock_provider(_canonical_llm_payload())

    service.analyze(dataset_id, provider)

    messages = provider.chat.call_args.args[0]
    roles = [message.role for message in messages]
    assert roles == ["system", "user"]
    user_content = messages[1].content
    assert "EDA_SUMMARY_JSON" in user_content
    assert '"rows"' in user_content


def test_analyze_raises_for_unknown_dataset(eda_service: EDAService) -> None:
    service = AIInsightService(eda_service=eda_service)
    provider = _mock_provider(_canonical_llm_payload())

    with pytest.raises(DatasetNotFoundError):
        service.analyze(uuid4(), provider)
