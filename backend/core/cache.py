"""Content-addressed cache over Upstash Redis.

Upstash speaks HTTP rather than the Redis wire protocol, which suits a
serverless deployment: no connection pool to keep warm, no socket to lose.

The cache is deliberately best-effort. A cache outage must degrade latency, not
correctness — every method swallows transport errors and reports a miss, so the
caller recomputes instead of failing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx

from core.config import Settings
from core.logging import get_logger

log = get_logger(__name__)


def cache_key(namespace: str, *parts: str) -> str:
    """Content-addressed key: identical inputs always collide, by design."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"vidhi:{namespace}:{digest}"


class Cache:
    """Async client for the Upstash REST API."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._base = settings.upstash_redis_url.rstrip("/")
        self._token = settings.upstash_redis_token.get_secret_value()
        self._client = client or httpx.AsyncClient(timeout=10.0)
        self._owns_client = client is None

    async def _command(self, *args: Any) -> Any:
        """Issue one Redis command; None on any transport or protocol failure."""
        try:
            response = await self._client.post(
                self._base,
                headers={"Authorization": f"Bearer {self._token}"},
                json=[str(a) for a in args],
            )
            if response.status_code != 200:
                log.warning("cache_command_failed", status=response.status_code)
                return None
            return response.json().get("result")
        except Exception as exc:
            # Never propagate: a cache failure must not fail the request.
            log.warning("cache_unavailable", error=str(exc), command=str(args[0]))
            return None

    async def get_json(self, key: str) -> Any | None:
        raw = await self._command("GET", key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            # A corrupt entry is a miss, not an error.
            return None

    async def set_json(self, key: str, value: Any, ttl_s: int) -> None:
        await self._command("SET", key, json.dumps(value), "EX", ttl_s)

    async def get_many_json(self, keys: list[str]) -> dict[str, Any]:
        """Batch lookup — one round trip instead of N for a chunk batch."""
        if not keys:
            return {}
        results = await self._command("MGET", *keys)
        if not isinstance(results, list):
            return {}

        found: dict[str, Any] = {}
        for key, raw in zip(keys, results, strict=False):
            if raw is None:
                continue
            try:
                found[key] = json.loads(raw)
            except (TypeError, ValueError):
                continue
        return found

    async def ping(self) -> bool:
        result = await self._command("PING")
        return bool(result == "PONG")

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
