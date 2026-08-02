"""LLM router with provider failover.

Every configured provider speaks the OpenAI chat-completions protocol, so one
client covers all of them and failover is a matter of changing the base URL and
key. Providers are tried in configured order; a provider that rate-limits or
errors is skipped for the rest of the request, and the next one is tried.

Structured output is requested via JSON mode where the provider supports it,
but the response is always validated against a Pydantic schema rather than
trusted — a model that returns prose instead of JSON is a normal occurrence,
not an exceptional one.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from core.config import LLMProvider, Settings
from core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

#: Base URL per provider. All are OpenAI-compatible.
PROVIDER_BASE_URL: dict[LLMProvider, str] = {
    LLMProvider.GROQ: "https://api.groq.com/openai/v1",
    LLMProvider.CEREBRAS: "https://api.cerebras.ai/v1",
    LLMProvider.OPENROUTER: "https://openrouter.ai/api/v1",
}


@dataclass(frozen=True, slots=True)
class ModelTarget:
    """One model on one provider, with the throughput it is allowed.

    Rate limits are enforced **per model**, not per key, so a pool of several
    models multiplies the available budget. That is the whole point of this
    type: on the free tier a single model caps at 12k tokens/minute, which a
    contract audit exhausts in seconds, while the pool below totals ~110k.
    """

    provider: LLMProvider
    model: str
    #: Tokens per minute, read from the provider's own rate-limit headers.
    #: Used to weight selection, not to enforce anything locally.
    tokens_per_minute: int
    #: Measured round trip for a trivial call. Ties are broken by speed.
    typical_ms: int

    @property
    def id(self) -> str:
        return f"{self.provider.value}:{self.model}"


#: The pool, ordered by throughput then latency. Every entry was verified
#: against live keys: it answers 200, returns usable JSON, and reports the
#: limit shown here in `x-ratelimit-limit-tokens`.
#:
#: Models excluded after testing, and why:
#:   - qwen/qwen3.6-27b emits <think> reasoning ahead of its JSON
#:   - groq/compound-mini was rate-limited on every attempt
#:   - cerebras returns 402 without billing enabled
#:   - openrouter free models beyond nemotron were slow (0.9-8.4s) or 429
DEFAULT_POOL: tuple[ModelTarget, ...] = (
    # 70k/min — nearly six times the next best, so it carries the load.
    ModelTarget(LLMProvider.GROQ, "groq/compound", 70_000, 926),
    ModelTarget(LLMProvider.GROQ, "llama-3.3-70b-versatile", 12_000, 331),
    ModelTarget(LLMProvider.GROQ, "openai/gpt-oss-120b", 8_000, 684),
    ModelTarget(LLMProvider.GROQ, "openai/gpt-oss-20b", 8_000, 400),
    # Fastest in the pool; used when the larger models are saturated.
    ModelTarget(LLMProvider.GROQ, "llama-3.1-8b-instant", 6_000, 151),
    # Different provider entirely, so it survives a Groq-wide outage.
    ModelTarget(LLMProvider.OPENROUTER, "nvidia/nemotron-3-ultra-550b-a55b:free", 20_000, 733),
)

#: Status codes worth trying the next provider for. A 400 means our request is
#: wrong and would fail identically everywhere, so it is raised immediately.
FAILOVER_STATUS = frozenset({401, 402, 403, 408, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """No provider could serve the request."""


@dataclass(slots=True)
class Usage:
    """Token accounting, logged per call for the cost metric in Phase 8."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class _Failure:
    """A provider attempt that did not yield a completion."""

    reason: str
    retryable: bool
    status: int | None = None
    detail: str = ""
    #: Seconds the provider asked us to wait, when it said so.
    retry_after: float | None = None


@dataclass(slots=True)
class Completion:
    text: str
    provider: LLMProvider
    model: str
    usage: Usage = field(default_factory=Usage)


def _extract_json(text: str) -> str:
    """Recover a JSON object from a response that may wrap it.

    Models routinely emit ```json fences or a sentence of preamble even under
    explicit instruction. Reaching for the outermost braces is more reliable
    than re-prompting, and costs nothing when the response is already clean.
    """
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        return text[start : end + 1]

    return text


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """How long a 429 asks us to wait, from either header form.

    Groq reports a float in `retry-after`; other providers use
    `x-ratelimit-reset-tokens` with a suffix like "7.2s".
    """
    for header in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = response.headers.get(header)
        if not raw:
            continue
        try:
            return float(raw.rstrip("sm"))
        except ValueError:
            continue
    return None


class LLMRouter:
    """Routes each call to whichever pooled model is free.

    Because rate limits are per model, spreading calls across the pool
    multiplies throughput rather than merely providing a fallback. A model that
    answers 429 is parked for the window it asks for and skipped until then, so
    a saturated model costs one failed request rather than blocking the run.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(timeout=settings.llm_request_timeout_s)
        self._owns_client = client is None
        #: model id -> monotonic time it becomes usable again.
        self._parked: dict[str, float] = {}
        #: Round-robin cursor, so consecutive calls do not all pile onto the
        #: highest-throughput model and exhaust it together.
        self._cursor = 0

    def _pool(self) -> list[ModelTarget]:
        """Configured targets whose provider has a key."""
        available = {
            target
            for target in DEFAULT_POOL
            if self._settings.api_key_for(target.provider) is not None
        }
        if not available:
            raise LLMError(
                "No LLM provider is configured. Set GROQ_API_KEY, "
                "CEREBRAS_API_KEY, or OPENROUTER_API_KEY."
            )
        return [target for target in DEFAULT_POOL if target in available]

    def _park(self, target: ModelTarget, seconds: float) -> None:
        """Take a rate-limited model out of rotation for a bounded window."""
        capped = min(max(seconds, 1.0), 60.0)
        self._parked[target.id] = time.monotonic() + capped
        log.info("model_parked", model=target.id, seconds=round(capped, 1))

    def _order(self) -> list[ModelTarget]:
        """Pool ordered for this attempt: free models first, parked last.

        Rotation starts one past the previous call so concurrent clauses fan
        out across models instead of contending for the same one.
        """
        pool = self._pool()
        self._cursor = (self._cursor + 1) % len(pool)
        rotated = pool[self._cursor :] + pool[: self._cursor]

        now = time.monotonic()
        free = [t for t in rotated if self._parked.get(t.id, 0.0) <= now]
        parked = [t for t in rotated if self._parked.get(t.id, 0.0) > now]
        # Parked models are still tried as a last resort: the window is an
        # estimate, and failing the request outright would be worse.
        return free + parked

    def _headers(self, provider: LLMProvider) -> dict[str, str]:
        key = self._settings.api_key_for(provider)
        assert key is not None  # guaranteed by configured_providers()
        return {"Authorization": f"Bearer {key.get_secret_value()}"}

    def _payload(
        self,
        target: ModelTarget,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._settings.llm_model_overrides.get(target.provider.value, target.model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stream:
            payload["stream"] = True
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            # Groq requires the literal word "json" somewhere in the messages
            # before it will honour json_object. Our prompts say "JSON", which
            # satisfies it, but a caller's might not.
            if not any("json" in m.get("content", "").lower() for m in messages):
                payload["messages"] = [
                    *messages,
                    {"role": "user", "content": "Respond with json only."},
                ]
        return payload

    async def _attempt(
        self,
        target: ModelTarget,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> Completion | _Failure:
        """One request to one model. Never raises for an expected failure."""
        provider = target.provider
        try:
            # Bounded per attempt, not just per request. The pool's whole value
            # is that another model is usually idle, so waiting out a slow one
            # is strictly worse than moving on — a model that has not answered
            # in `llm_attempt_timeout_s` is treated as failed and the next is
            # tried immediately. Without this a single degraded model could
            # hold a clause for the full request timeout while five others sat
            # free.
            response = await asyncio.wait_for(
                self._client.post(
                    f"{PROVIDER_BASE_URL[provider]}/chat/completions",
                    headers=self._headers(provider),
                    json=self._payload(
                        target,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                        stream=False,
                    ),
                ),
                timeout=self._settings.llm_attempt_timeout_s,
            )
        except TimeoutError:
            log.warning(
                "llm_attempt_timeout",
                model=target.id,
                seconds=self._settings.llm_attempt_timeout_s,
            )
            # Parked briefly: a model that just timed out is likely saturated,
            # and concurrent clauses should skip it rather than each spending
            # the same timeout discovering that independently.
            self._park(target, self._settings.llm_attempt_timeout_s)
            return _Failure(f"{target.id}: timed out", retryable=True)
        except httpx.HTTPError as exc:
            log.warning("llm_transport_error", provider=provider.value, error=str(exc))
            return _Failure(f"{provider}: {type(exc).__name__}", retryable=True)

        if response.status_code != 200:
            detail = response.text[:200]
            log.warning(
                "llm_provider_failed",
                provider=provider.value,
                model=target.model,
                status=response.status_code,
                detail=detail,
            )
            # `json_validate_failed` is a 400, but it is not a malformed
            # request: Groq returns it when a model runs out of tokens
            # mid-object, so the generation is truncated rather than wrong.
            # Another model with a different verbosity will often succeed, so
            # this must not abort the whole call the way a real 400 does.
            truncated_json = "json_validate_failed" in detail
            return _Failure(
                f"{target.id}: HTTP {response.status_code}",
                retryable=response.status_code in FAILOVER_STATUS or truncated_json,
                status=response.status_code,
                detail=detail,
                retry_after=_retry_after_seconds(response),
            )

        body = response.json()
        # A 200 does not guarantee a completion: OpenRouter returns upstream
        # errors (rate limits, model unavailable) in the body with a 200
        # status, so the shape must be checked rather than indexed.
        choices = body.get("choices")
        if not choices:
            reason = str(body.get("error") or body)[:200]
            log.warning("llm_empty_response", provider=provider.value, detail=reason)
            return _Failure(f"{provider}: {reason}", retryable=True)

        usage_body = body.get("usage") or {}
        completion = Completion(
            text=choices[0].get("message", {}).get("content") or "",
            provider=provider,
            model=body.get("model", target.model),
            usage=Usage(
                prompt_tokens=usage_body.get("prompt_tokens", 0),
                completion_tokens=usage_body.get("completion_tokens", 0),
            ),
        )
        log.info(
            "llm_completion",
            provider=provider.value,
            model=target.model,
            tokens=completion.usage.total,
        )
        return completion

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> Completion:
        """One completion, rotating through the pool until a model answers.

        Waiting out a rate limit is the last resort rather than the first: with
        several models available, moving to a free one is faster than sitting
        out a window on a saturated one.
        """
        errors: list[str] = []
        pool = self._order()

        for index, target in enumerate(pool):
            outcome = await self._attempt(target, messages, temperature, max_tokens, json_mode)
            if isinstance(outcome, Completion):
                return outcome

            errors.append(outcome.reason)

            if not outcome.retryable:
                # The request itself is malformed; every model rejects it.
                raise LLMError(f"{target.id} rejected the request: {outcome.detail}")

            if outcome.status == 429:
                # Park it so concurrent clauses skip this model rather than
                # each discovering its limit independently.
                self._park(target, outcome.retry_after or 10.0)
                continue

            # Anything else transient: just move to the next model.
            if index < len(pool) - 1:
                continue

            # Last model, and the failure was transient — a brief wait is
            # better than failing the clause outright.
            wait = outcome.retry_after
            if wait is not None and wait <= self._settings.llm_max_retry_wait_s:
                log.info("llm_waiting", model=target.id, seconds=wait)
                await asyncio.sleep(wait)
                retried = await self._attempt(target, messages, temperature, max_tokens, json_mode)
                if isinstance(retried, Completion):
                    return retried
                errors.append(retried.reason)

        raise LLMError(f"all models failed: {'; '.join(errors[:4])}")

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Yield content deltas as they arrive.

        Failover applies only until the first token: once bytes have reached
        the caller, switching provider mid-stream would corrupt the output, so
        a late failure propagates instead.
        """
        errors: list[str] = []

        for target in self._order():
            provider = target.provider
            started = False
            try:
                async with self._client.stream(
                    "POST",
                    f"{PROVIDER_BASE_URL[provider]}/chat/completions",
                    headers=self._headers(provider),
                    json=self._payload(
                        target,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=False,
                        stream=True,
                    ),
                    # Bounds the wait for response headers only. The read
                    # timeout stays at the client default: once tokens are
                    # flowing a long generation is legitimate, and the
                    # `started` guard means a mid-stream failure propagates
                    # rather than silently switching models.
                    timeout=httpx.Timeout(
                        self._settings.llm_request_timeout_s,
                        connect=self._settings.llm_attempt_timeout_s,
                        pool=self._settings.llm_attempt_timeout_s,
                    ),
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        errors.append(f"{target.id}: HTTP {response.status_code}")
                        if response.status_code == 429:
                            self._park(target, _retry_after_seconds(response) or 10.0)
                        continue

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            delta = json.loads(data)["choices"][0].get("delta", {})
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                        content = delta.get("content")
                        if content:
                            started = True
                            yield content
                    return
            except TimeoutError:
                # Only reachable before the first token — see the guard on the
                # connection below. A model that has not started producing
                # within the attempt budget is stalled, and another is free.
                log.warning(
                    "llm_stream_timeout",
                    model=target.id,
                    seconds=self._settings.llm_attempt_timeout_s,
                )
                self._park(target, self._settings.llm_attempt_timeout_s)
                errors.append(f"{target.id}: timed out before first token")
                continue
            except httpx.HTTPError as exc:
                if started:
                    raise LLMError(f"stream interrupted: {exc}") from exc
                errors.append(f"{target.id}: {type(exc).__name__}")
                continue

        raise LLMError(f"all models failed: {'; '.join(errors[:4])}")

    async def structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        retries: int = 2,
    ) -> T:
        """A completion parsed into `schema`.

        On a parse failure the validation error is fed back to the model, which
        corrects far more often than a bare retry at higher temperature.
        """
        attempt_messages = list(messages)

        for attempt in range(retries + 1):
            completion = await self.complete(
                attempt_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
            )
            raw = _extract_json(completion.text)

            try:
                return schema.model_validate_json(raw)
            except ValidationError as exc:
                log.warning(
                    "llm_schema_violation",
                    attempt=attempt + 1,
                    schema=schema.__name__,
                    error=str(exc)[:300],
                )
                if attempt == retries:
                    raise LLMError(
                        f"model did not produce valid {schema.__name__} "
                        f"after {retries + 1} attempts"
                    ) from exc

                attempt_messages = [
                    *messages,
                    {"role": "assistant", "content": completion.text[:1000]},
                    {
                        "role": "user",
                        "content": (
                            "That response did not match the required schema:\n"
                            f"{str(exc)[:500]}\n\n"
                            "Return only valid JSON matching the schema. No prose."
                        ),
                    },
                ]

        raise LLMError("unreachable")  # pragma: no cover

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
