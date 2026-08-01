"""Fine-tuned classifiers.

Two narrow, measurable subtasks are handled by trained models rather than
prompts: clause risk (ComplianceGuard) and precedent stance (CaseLens). Both
sit behind an interface that degrades to the LLM path when no model is
configured, so the system runs identically with or without them — which is what
makes the before/after comparison in `ml/*/metrics.json` meaningful.

Predictions are cached by content hash. A clause's risk does not change between
runs, and re-running a model over boilerplate that appears in every contract is
pure waste.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.cache import Cache, cache_key
from core.config import Settings
from core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - avoids importing torch at runtime
    from transformers import Pipeline

log = get_logger(__name__)


@dataclass(slots=True)
class Prediction:
    label: str
    confidence: float
    #: Which path produced this — recorded so eval can separate the two.
    source: str = "classifier"


class TransformerClassifier:
    """A fine-tuned sequence classifier loaded from a local path or the Hub.

    Loading is lazy and guarded: the model is a couple of hundred megabytes and
    most requests never touch it.
    """

    def __init__(
        self,
        model_id: str,
        cache: Cache | None = None,
        *,
        namespace: str,
        ttl_s: int,
    ) -> None:
        self._model_id = model_id
        self._cache = cache
        self._namespace = namespace
        self._ttl_s = ttl_s
        self._pipeline: Pipeline | None = None
        self._lock = asyncio.Lock()

    async def _get_pipeline(self) -> Pipeline:
        existing = self._pipeline
        if existing is not None:
            return existing

        async with self._lock:
            already = self._pipeline
            if already is not None:
                return already

            log.info("loading_classifier", model=self._model_id)
            from transformers import pipeline

            loaded: Pipeline = await asyncio.to_thread(
                pipeline,
                "text-classification",
                model=self._model_id,
                truncation=True,
                max_length=256,
            )
            self._pipeline = loaded
            log.info("classifier_ready", model=self._model_id)
            return loaded

    async def predict(self, text: str) -> Prediction:
        key = cache_key(self._namespace, self._model_id, text)

        if self._cache is not None:
            hit = await self._cache.get_json(key)
            if isinstance(hit, dict) and "label" in hit:
                return Prediction(
                    label=str(hit["label"]),
                    confidence=float(hit.get("confidence", 0.0)),
                    source="classifier-cached",
                )

        classifier = await self._get_pipeline()
        raw: Any = await asyncio.to_thread(classifier, text)
        best = raw[0] if isinstance(raw, list) else raw

        prediction = Prediction(label=str(best["label"]).lower(), confidence=float(best["score"]))

        if self._cache is not None:
            await self._cache.set_json(
                key,
                {"label": prediction.label, "confidence": prediction.confidence},
                self._ttl_s,
            )
        return prediction

    async def predict_many(self, texts: list[str]) -> list[Prediction]:
        return list(await asyncio.gather(*(self.predict(t) for t in texts)))


class ClassifierRegistry:
    """Holds whichever classifiers are configured.

    Absent configuration is the normal case before Phase 6 training completes,
    and is not an error: callers fall back to the LLM path.
    """

    def __init__(self, settings: Settings, cache: Cache | None = None) -> None:
        self._risk = (
            TransformerClassifier(
                settings.risk_classifier_model,
                cache,
                namespace="risk",
                ttl_s=settings.prediction_cache_ttl_s,
            )
            if settings.risk_classifier_model
            else None
        )
        self._stance = (
            TransformerClassifier(
                settings.stance_classifier_model,
                cache,
                namespace="stance",
                ttl_s=settings.prediction_cache_ttl_s,
            )
            if settings.stance_classifier_model
            else None
        )

    @property
    def has_risk_classifier(self) -> bool:
        return self._risk is not None

    @property
    def has_stance_classifier(self) -> bool:
        return self._stance is not None

    async def risk(self, clause: str) -> Prediction | None:
        """Predicted risk, or None when no model is configured."""
        if self._risk is None:
            return None
        try:
            return await self._risk.predict(clause)
        except Exception as exc:
            # A classifier failure must degrade to the LLM path, not fail the
            # audit: the LLM already produced a risk level.
            log.warning("risk_classifier_failed", error=str(exc))
            return None

    async def stance(self, facts: str, passage: str) -> Prediction | None:
        if self._stance is None:
            return None
        try:
            # The pair is encoded as one sequence, matching how the model was
            # trained; a bare passage would lose the position being argued.
            return await self._stance.predict(f"{facts}\n[SEP]\n{passage}")
        except Exception as exc:
            log.warning("stance_classifier_failed", error=str(exc))
            return None
