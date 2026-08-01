"""Settings drive failover and secret handling — both are worth pinning down."""

import pytest
from pydantic import ValidationError

from core.config import Environment, LLMProvider
from tests.conftest import build_settings


def test_configured_providers_respects_failover_order() -> None:
    settings = build_settings(
        groq_api_key="g",
        cerebras_api_key="c",
        openrouter_api_key="o",
    )
    assert settings.configured_providers() == [
        LLMProvider.GROQ,
        LLMProvider.CEREBRAS,
        LLMProvider.OPENROUTER,
    ]


def test_providers_without_keys_are_skipped() -> None:
    settings = build_settings(groq_api_key=None, cerebras_api_key="c")
    assert settings.configured_providers() == [LLMProvider.CEREBRAS]


def test_empty_provider_order_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one provider"):
        build_settings(llm_provider_order=[])


def test_secrets_are_masked_in_repr() -> None:
    settings = build_settings(groq_api_key="super-secret")
    assert "super-secret" not in repr(settings)
    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "super-secret"


def test_retrieval_attempts_are_bounded() -> None:
    """An unbounded critic loop is a runaway cost bug, so the cap is validated."""
    with pytest.raises(ValidationError):
        build_settings(max_retrieval_attempts=99)


def test_is_local_flag() -> None:
    assert build_settings().is_local is True
    assert build_settings(environment=Environment.PRODUCTION).is_local is False
