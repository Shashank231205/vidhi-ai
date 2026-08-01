"""Classifier wiring.

The model itself is measured in ml/risk_classifier/metrics.json. What matters
here is that its absence or failure never breaks an audit — the LLM path has to
remain a working fallback, since it is also the baseline the model is compared
against.
"""

import pytest

from core.classifiers import ClassifierRegistry, Prediction
from tests.conftest import build_settings


def test_registry_is_empty_without_configured_models() -> None:
    registry = ClassifierRegistry(build_settings())

    assert registry.has_risk_classifier is False
    assert registry.has_stance_classifier is False


def test_registry_reports_configured_models() -> None:
    registry = ClassifierRegistry(
        build_settings(risk_classifier_model="some/model", stance_classifier_model="other/model")
    )

    assert registry.has_risk_classifier is True
    assert registry.has_stance_classifier is True


async def test_risk_returns_none_when_unconfigured() -> None:
    """Callers treat None as 'keep the LLM's answer', so it must not raise."""
    assert await ClassifierRegistry(build_settings()).risk("some clause") is None


async def test_stance_returns_none_when_unconfigured() -> None:
    assert await ClassifierRegistry(build_settings()).stance("facts", "passage") is None


async def test_a_broken_model_degrades_rather_than_raising() -> None:
    """A missing or corrupt model must not fail the whole audit."""
    registry = ClassifierRegistry(
        build_settings(risk_classifier_model="definitely/not-a-real-model-xyz")
    )

    assert await registry.risk("a clause") is None


def test_prediction_records_its_source() -> None:
    """Eval needs to separate classifier decisions from LLM ones."""
    assert Prediction(label="high", confidence=0.9).source == "classifier"
    assert Prediction(label="high", confidence=0.9, source="classifier-cached").source == (
        "classifier-cached"
    )


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_prediction_accepts_the_confidence_range(confidence: float) -> None:
    assert Prediction(label="low", confidence=confidence).confidence == confidence
