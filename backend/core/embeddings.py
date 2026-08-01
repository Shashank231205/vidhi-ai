"""Embedding service — BGE-M3 via the HuggingFace Inference API.

Two properties matter more than raw throughput here:

- **Cached by content hash.** Legal corpora repeat heavily (boilerplate clauses,
  re-ingested statutes), and an identical span must never be re-embedded. The
  cache is content-addressed, so it stays correct across runs and documents.
- **Batched and retried.** The free tier cold-starts and rate-limits, so
  requests are batched and retried with backoff rather than failing a whole
  ingestion run on one 503.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from core.cache import Cache, cache_key
from core.config import Settings
from core.logging import get_logger

log = get_logger(__name__)

#: The legacy api-inference host is retired; the router is the current path.
HF_ROUTER = "https://router.huggingface.co/hf-inference/models"

Vector = list[float]


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced after retries."""


class EmbeddingService:
    def __init__(
        self,
        settings: Settings,
        cache: Cache | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions
        self._batch_size = settings.embedding_batch_size
        self._cache = cache
        # Generous timeout: a cold model can take tens of seconds to load.
        self._client = client or httpx.AsyncClient(timeout=120.0)
        self._owns_client = client is None

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _key(self, text: str) -> str:
        # Model is part of the key: vectors from different models are not
        # interchangeable, so a model swap must miss rather than return stale.
        return cache_key("embed", self._model, text)

    async def _post(self, inputs: list[str]) -> list[Vector]:
        """One batch, with backoff over cold starts and rate limits."""
        url = f"{HF_ROUTER}/{self._model}/pipeline/feature-extraction"
        headers = {"Authorization": f"Bearer {self._settings.hf_api_token.get_secret_value()}"}
        payload: dict[str, Any] = {
            "inputs": inputs,
            "options": {"wait_for_model": True},
        }

        last_error = "unknown"
        for attempt in range(4):
            try:
                response = await self._client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_error = str(exc)
            else:
                if response.status_code == 200:
                    return self._parse(response.json(), expected=len(inputs))
                # 429 rate limit, 503 cold start — both are worth retrying.
                if response.status_code not in (429, 503):
                    raise EmbeddingError(
                        f"embedding request failed: {response.status_code} {response.text[:200]}"
                    )
                last_error = f"HTTP {response.status_code}"

            delay = 2**attempt
            log.warning("embedding_retry", attempt=attempt + 1, error=last_error, delay=delay)
            await asyncio.sleep(delay)

        raise EmbeddingError(f"embedding failed after retries: {last_error}")

    def _parse(self, data: Any, *, expected: int) -> list[Vector]:
        """Normalise the response shape and assert dimensionality.

        A wrong dimensionality must fail loudly here: pgvector would otherwise
        reject the insert far from the cause, or worse, a mixed-dimension
        corpus would silently degrade retrieval.
        """
        if not isinstance(data, list):
            raise EmbeddingError(f"unexpected embedding response: {type(data).__name__}")

        # Some pipelines return token-level vectors; mean-pool to one per input.
        vectors: list[Vector] = []
        for item in data:
            if item and isinstance(item[0], list):
                width = len(item[0])
                pooled = [sum(tok[i] for tok in item) / len(item) for i in range(width)]
                vectors.append(pooled)
            else:
                vectors.append(list(item))

        if len(vectors) != expected:
            raise EmbeddingError(f"expected {expected} embeddings, received {len(vectors)}")
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise EmbeddingError(
                    f"expected {self._dimensions}-dim embeddings, received {len(vector)}"
                )
        return vectors

    async def embed(self, texts: list[str]) -> list[Vector]:
        """Embed texts, preserving input order.

        Cache hits are served without a request; only genuine misses reach the
        network, and duplicates within one call are collapsed.
        """
        if not texts:
            return []

        results: dict[int, Vector] = {}
        pending: dict[str, list[int]] = {}

        if self._cache is not None:
            keys = [self._key(t) for t in texts]
            cached = await self._cache.get_many_json(list(dict.fromkeys(keys)))
            for index, (text, key) in enumerate(zip(texts, keys, strict=True)):
                hit = cached.get(key)
                if isinstance(hit, list) and len(hit) == self._dimensions:
                    results[index] = hit
                else:
                    pending.setdefault(text, []).append(index)
        else:
            for index, text in enumerate(texts):
                pending.setdefault(text, []).append(index)

        unique = list(pending)
        if unique:
            log.info(
                "embedding_batch",
                total=len(texts),
                cached=len(results),
                to_embed=len(unique),
            )

        for start in range(0, len(unique), self._batch_size):
            batch = unique[start : start + self._batch_size]
            vectors = await self._post(batch)
            for text, vector in zip(batch, vectors, strict=True):
                for index in pending[text]:
                    results[index] = vector
                if self._cache is not None:
                    await self._cache.set_json(
                        self._key(text), vector, self._settings.embedding_cache_ttl_s
                    )

        return [results[i] for i in range(len(texts))]

    async def embed_one(self, text: str) -> Vector:
        return (await self.embed([text]))[0]

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
