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
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from core.config import LLMProvider, Settings
from core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

#: Base URL and default model per provider. All are OpenAI-compatible.
PROVIDER_CONFIG: dict[LLMProvider, dict[str, str]] = {
    LLMProvider.GROQ: {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    LLMProvider.CEREBRAS: {
        "base_url": "https://api.cerebras.ai/v1",
        "model": "gpt-oss-120b",
    },
    LLMProvider.OPENROUTER: {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    },
}

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
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(timeout=settings.llm_request_timeout_s)
        self._owns_client = client is None

    def _providers(self) -> list[LLMProvider]:
        available = self._settings.configured_providers()
        if not available:
            raise LLMError(
                "No LLM provider is configured. Set GROQ_API_KEY, "
                "CEREBRAS_API_KEY, or OPENROUTER_API_KEY."
            )
        return available

    def _headers(self, provider: LLMProvider) -> dict[str, str]:
        key = self._settings.api_key_for(provider)
        assert key is not None  # guaranteed by configured_providers()
        return {"Authorization": f"Bearer {key.get_secret_value()}"}

    def model_for(self, provider: LLMProvider) -> str:
        return self._settings.llm_model_overrides.get(
            provider.value, PROVIDER_CONFIG[provider]["model"]
        )

    def _payload(
        self,
        provider: LLMProvider,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_for(provider),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stream:
            payload["stream"] = True
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def _attempt(
        self,
        provider: LLMProvider,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> Completion | _Failure:
        """One request to one provider. Never raises for an expected failure."""
        config = PROVIDER_CONFIG[provider]
        try:
            response = await self._client.post(
                f"{config['base_url']}/chat/completions",
                headers=self._headers(provider),
                json=self._payload(
                    provider,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    stream=False,
                ),
            )
        except httpx.HTTPError as exc:
            log.warning("llm_transport_error", provider=provider.value, error=str(exc))
            return _Failure(f"{provider}: {type(exc).__name__}", retryable=True)

        if response.status_code != 200:
            detail = response.text[:200]
            log.warning(
                "llm_provider_failed",
                provider=provider.value,
                status=response.status_code,
                detail=detail,
            )
            return _Failure(
                f"{provider}: HTTP {response.status_code}",
                retryable=response.status_code in FAILOVER_STATUS,
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
            model=body.get("model", self.model_for(provider)),
            usage=Usage(
                prompt_tokens=usage_body.get("prompt_tokens", 0),
                completion_tokens=usage_body.get("completion_tokens", 0),
            ),
        )
        log.info("llm_completion", provider=provider.value, tokens=completion.usage.total)
        return completion

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> Completion:
        """One completion, trying providers in order until one succeeds."""
        errors: list[str] = []

        for provider in self._providers():
            outcome = await self._attempt(provider, messages, temperature, max_tokens, json_mode)
            if isinstance(outcome, Completion):
                return outcome

            errors.append(outcome.reason)
            if not outcome.retryable:
                # Our request is malformed; another provider rejects it too.
                raise LLMError(f"{provider} rejected the request: {outcome.detail}")

            # A short rate-limit window is worth waiting out on the primary:
            # Groq's free tier is 12k tokens/min and recovers in seconds, while
            # the fallbacks are markedly slower. Long waits fail over instead.
            wait = outcome.retry_after
            if (
                outcome.status == 429
                and wait is not None
                and wait <= self._settings.llm_max_retry_wait_s
            ):
                log.info("llm_rate_limited_waiting", provider=provider.value, seconds=wait)
                await asyncio.sleep(wait)
                retried = await self._attempt(
                    provider, messages, temperature, max_tokens, json_mode
                )
                if isinstance(retried, Completion):
                    return retried
                errors.append(retried.reason)

        raise LLMError(f"all providers failed: {'; '.join(errors)}")

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

        for provider in self._providers():
            config = PROVIDER_CONFIG[provider]
            started = False
            try:
                async with self._client.stream(
                    "POST",
                    f"{config['base_url']}/chat/completions",
                    headers=self._headers(provider),
                    json=self._payload(
                        provider,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=False,
                        stream=True,
                    ),
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        errors.append(f"{provider}: HTTP {response.status_code}")
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
            except httpx.HTTPError as exc:
                if started:
                    raise LLMError(f"stream interrupted: {exc}") from exc
                errors.append(f"{provider}: {type(exc).__name__}")
                continue

        raise LLMError(f"all providers failed: {'; '.join(errors)}")

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
