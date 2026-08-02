"""Embedding service.

Two backends behind one interface:

- **Local** (default). BGE-M3 in-process via sentence-transformers. Measured on
  this machine: ~30ms per query versus ~500ms-1.3s through HuggingFace's free
  Inference API, with no cold starts and no rate limits. Latency is a feature
  here, so this is the default despite the model download.
- **Remote**. The same model over HF Inference. Kept for deployment targets
  where a 2GB model and its memory footprint are not welcome — a small
  container, or a serverless function.

Both produce 1024-dimension vectors from the same weights, so switching between
them does not invalidate an existing corpus.

Above either backend sits a content-addressed cache: identical text is never
embedded twice, across runs or documents.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import httpx

from core.cache import Cache, cache_key
from core.config import EmbeddingBackend, Settings
from core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from sentence_transformers import SentenceTransformer

log = get_logger(__name__)

#: The legacy api-inference host is retired; the router is the current path.
HF_ROUTER = "https://router.huggingface.co/hf-inference/models"

Vector = list[float]


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


class EmbeddingBackendProtocol(ABC):
    """Anything that turns text into vectors."""

    @abstractmethod
    async def encode(self, texts: list[str]) -> list[Vector]: ...

    async def close(self) -> None:  # pragma: no cover - default no-op
        return None


class LocalBackend(EmbeddingBackendProtocol):
    """In-process sentence-transformers.

    The model is loaded lazily on first use and reused for the process
    lifetime; loading is slow (seconds to minutes on a cold cache) but happens
    once. Encoding runs in a worker thread because it is CPU-bound and would
    otherwise block the event loop for the whole batch.
    """

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._model: SentenceTransformer | None = None
        self._load_lock = asyncio.Lock()

    async def _get_model(self) -> SentenceTransformer:
        cached = self._model
        if cached is not None:
            return cached

        async with self._load_lock:
            # Re-check under the lock: another task may have loaded it while
            # this one waited, and loading twice would double the memory.
            existing = self._model
            if existing is not None:
                return existing

            log.info("loading_local_embedding_model", model=self._model_name)
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on install extras
                # The deployed image omits this extra deliberately, so the
                # likely cause is EMBEDDING_BACKEND=local in an environment
                # built for the remote backend. Say that, rather than let a
                # bare ImportError suggest a broken install.
                raise EmbeddingError(
                    "EMBEDDING_BACKEND=local needs sentence-transformers, which is not "
                    "installed. Either install it (`uv pip install -e '.[local-embeddings]'`) "
                    "or set EMBEDDING_BACKEND=remote, which uses the same model over the "
                    "HuggingFace API and produces identical vectors."
                ) from exc

            loaded: SentenceTransformer = await asyncio.to_thread(
                SentenceTransformer, self._model_name
            )
            self._model = loaded
            log.info("local_embedding_model_ready", model=self._model_name)
            return loaded

    async def encode(self, texts: list[str]) -> list[Vector]:
        model = await self._get_model()
        vectors = await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=self._batch_size,
            # Cosine distance on normalised vectors is a dot product, which is
            # what the pgvector HNSW index is built for.
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vectors]

    async def warm(self) -> None:
        """Pay the load cost at startup rather than on a user's first query."""
        await self.encode(["warm"])


class RemoteBackend(EmbeddingBackendProtocol):
    """HuggingFace Inference API, with backoff over cold starts and throttling."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._model = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        # Generous: a cold model can take tens of seconds to load remotely.
        self._client = client or httpx.AsyncClient(timeout=120.0)
        self._owns_client = client is None

    async def encode(self, texts: list[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(await self._post(texts[start : start + self._batch_size]))
        return vectors

    async def _post(self, inputs: list[str]) -> list[Vector]:
        url = f"{HF_ROUTER}/{self._model}/pipeline/feature-extraction"
        headers = {"Authorization": f"Bearer {self._settings.hf_api_token.get_secret_value()}"}
        payload: dict[str, Any] = {"inputs": inputs, "options": {"wait_for_model": True}}

        last_error = "unknown"
        for attempt in range(4):
            try:
                response = await self._client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_error = str(exc)
            else:
                if response.status_code == 200:
                    return _normalise_response(response.json(), expected=len(inputs))
                # 429 throttle, 503 cold start — both worth retrying.
                if response.status_code not in (429, 503):
                    raise EmbeddingError(
                        f"embedding request failed: {response.status_code} {response.text[:200]}"
                    )
                last_error = f"HTTP {response.status_code}"

            delay = 2**attempt
            log.warning("embedding_retry", attempt=attempt + 1, error=last_error)
            await asyncio.sleep(delay)

        raise EmbeddingError(f"embedding failed after retries: {last_error}")

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _normalise_response(data: Any, *, expected: int) -> list[Vector]:
    """Flatten the API's response into one vector per input."""
    if not isinstance(data, list):
        raise EmbeddingError(f"unexpected embedding response: {type(data).__name__}")

    vectors: list[Vector] = []
    for item in data:
        # Some pipelines return token-level vectors; mean-pool to one per input.
        if item and isinstance(item[0], list):
            width = len(item[0])
            vectors.append([sum(t[i] for t in item) / len(item) for i in range(width)])
        else:
            vectors.append(list(item))

    if len(vectors) != expected:
        raise EmbeddingError(f"expected {expected} embeddings, received {len(vectors)}")
    return vectors


class EmbeddingService:
    """Cache-fronted embeddings over a configurable backend."""

    def __init__(
        self,
        settings: Settings,
        cache: Cache | None = None,
        client: httpx.AsyncClient | None = None,
        backend: EmbeddingBackendProtocol | None = None,
    ) -> None:
        self._settings = settings
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions
        self._batch_size = settings.embedding_batch_size
        self._cache = cache
        self._backend = backend or (
            LocalBackend(settings)
            if settings.embedding_backend is EmbeddingBackend.LOCAL
            else RemoteBackend(settings, client)
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _key(self, text: str) -> str:
        # The model is part of the key: vectors from different models are not
        # interchangeable, so a model swap must miss rather than serve stale.
        return cache_key("embed", self._model, text)

    def _validate(self, vectors: list[Vector]) -> list[Vector]:
        """Fail loudly on a dimensionality mismatch.

        pgvector would otherwise reject the insert far from the cause, and a
        mixed-dimension corpus degrades retrieval silently.
        """
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise EmbeddingError(
                    f"expected {self._dimensions}-dim embeddings, got {len(vector)}"
                )
        return vectors

    async def embed(self, texts: list[str]) -> list[Vector]:
        """Embed texts, preserving input order.

        Cache hits never reach the backend, and duplicates within one call are
        collapsed into a single encode.
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
            vectors = self._validate(await self._backend.encode(unique))
            for text, vector in zip(unique, vectors, strict=True):
                for index in pending[text]:
                    results[index] = vector
            if self._cache is not None:
                await asyncio.gather(
                    *(
                        self._cache.set_json(self._key(t), v, self._settings.embedding_cache_ttl_s)
                        for t, v in zip(unique, vectors, strict=True)
                    )
                )

        return [results[i] for i in range(len(texts))]

    async def embed_one(self, text: str) -> Vector:
        return (await self.embed([text]))[0]

    async def warm(self) -> None:
        """Load the model before serving traffic, where the backend supports it."""
        if isinstance(self._backend, LocalBackend):
            await self._backend.warm()

    async def close(self) -> None:
        await self._backend.close()
