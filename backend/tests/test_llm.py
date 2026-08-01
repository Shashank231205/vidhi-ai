"""LLM routing: failover, JSON recovery, and schema enforcement."""

import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import BaseModel

from core.config import LLMProvider
from core.llm import LLMError, LLMRouter, _extract_json
from tests.conftest import build_settings


class Verdict(BaseModel):
    sufficient: bool
    reasoning: str


def completion_body(content: str, model: str = "test-model") -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


Handler = Callable[[httpx.Request], httpx.Response]


def router_with(handler: Handler, **settings_overrides: object) -> LLMRouter:
    settings = build_settings(**settings_overrides)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LLMRouter(settings, client=client)


async def test_uses_first_configured_provider() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(200, json=completion_body("hello"))

    router = router_with(handler)
    result = await router.complete([{"role": "user", "content": "hi"}])

    assert result.text == "hello"
    assert result.provider is LLMProvider.GROQ
    assert "groq" in seen[0]


async def test_fails_over_when_rate_limited() -> None:
    """A saturated provider must transparently reach a different one.

    Groq contributes several models to the pool, so a provider-wide 429 is
    several failed attempts before OpenRouter is reached — the point is that
    the caller still gets a completion.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if "groq" in request.url.host:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=completion_body("from fallback"))

    router = router_with(handler, openrouter_api_key="or-key")
    result = await router.complete([{"role": "user", "content": "hi"}])

    assert result.text == "from fallback"
    assert result.provider is LLMProvider.OPENROUTER
    assert any("groq" in host for host in seen)


async def test_rate_limited_models_are_parked() -> None:
    """A model that 429s is skipped on the next call rather than retried.

    Without this every concurrent clause rediscovers the same limit, turning
    one saturated model into a stall across the whole audit.
    """
    attempts: list[str] = []
    limited: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal limited
        model = json.loads(request.content.decode())["model"]
        attempts.append(model)

        # Rate-limit whichever model the rotation reaches first, rather than
        # naming one: the cursor advances per call, so the first model tried is
        # not fixed.
        if limited is None:
            limited = model
        if model == limited:
            return httpx.Response(429, headers={"retry-after": "30"}, text="limited")
        return httpx.Response(200, json=completion_body("ok"))

    router = router_with(handler)
    await router.complete([{"role": "user", "content": "x"}])

    for _ in range(3):
        await router.complete([{"role": "user", "content": "y"}])

    # Tried once, then skipped for the whole 30s window it asked for.
    assert limited is not None
    assert attempts.count(limited) == 1


async def test_pool_spreads_calls_across_models() -> None:
    """Consecutive calls must not all land on the same model.

    Rate limits are per model, so rotation is what multiplies the budget —
    hitting one model repeatedly would forfeit the entire benefit.
    """
    used: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        used.append(json.loads(request.content.decode())["model"])
        return httpx.Response(200, json=completion_body("ok"))

    router = router_with(handler)
    for _ in range(4):
        await router.complete([{"role": "user", "content": "x"}])

    assert len(set(used)) > 1


async def test_truncated_json_is_retryable() -> None:
    """Groq reports a token-truncated object as a 400, not a 5xx.

    It is not a malformed request — the generation ran out of room — so it must
    fall through to another model instead of aborting the call.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content.decode())["model"]
        seen.append(model)
        if len(seen) == 1:
            return httpx.Response(
                400,
                text='{"error":{"code":"json_validate_failed","failed_generation":""}}',
            )
        return httpx.Response(200, json=completion_body('{"ok":true}'))

    router = router_with(handler)
    result = await router.complete([{"role": "user", "content": "x"}], json_mode=True)

    assert result.text == '{"ok":true}'
    assert len(seen) >= 2


async def test_payment_required_fails_over() -> None:
    """Cerebras answers 402 on a free account; that must not end the request."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "groq" in request.url.host:
            return httpx.Response(402, text="payment required")
        return httpx.Response(200, json=completion_body("ok"))

    router = router_with(handler, openrouter_api_key="or-key")
    assert (await router.complete([{"role": "user", "content": "x"}])).text == "ok"


async def test_bad_request_does_not_fail_over() -> None:
    """A malformed request fails identically everywhere; retrying wastes quota."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="invalid model")

    router = router_with(handler, openrouter_api_key="or-key")
    with pytest.raises(LLMError, match="rejected the request"):
        await router.complete([{"role": "user", "content": "x"}])

    assert calls == 1


async def test_raises_when_every_provider_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    router = router_with(handler, openrouter_api_key="or-key")
    with pytest.raises(LLMError, match="all models failed"):
        await router.complete([{"role": "user", "content": "x"}])


async def test_no_configured_provider_is_a_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should never be called")

    router = router_with(handler, groq_api_key=None)
    with pytest.raises(LLMError, match="No LLM provider is configured"):
        await router.complete([{"role": "user", "content": "x"}])


async def test_transport_error_fails_over() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "groq" in request.url.host:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=completion_body("recovered"))

    router = router_with(handler, openrouter_api_key="or-key")
    assert (await router.complete([{"role": "user", "content": "x"}])).text == "recovered"


async def test_structured_parses_valid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=completion_body('{"sufficient": true, "reasoning": "covers it"}')
        )

    router = router_with(handler)
    verdict = await router.structured([{"role": "user", "content": "x"}], Verdict)

    assert verdict.sufficient is True
    assert verdict.reasoning == "covers it"


async def test_structured_retries_with_the_validation_error() -> None:
    """Feeding the error back corrects far more often than a blind retry."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, json=completion_body('{"wrong": "shape"}'))
        return httpx.Response(
            200, json=completion_body('{"sufficient": false, "reasoning": "fixed"}')
        )

    router = router_with(handler)
    verdict = await router.structured([{"role": "user", "content": "x"}], Verdict)

    assert verdict.reasoning == "fixed"
    assert attempts == 2


async def test_structured_gives_up_after_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body("not json at all"))

    router = router_with(handler)
    with pytest.raises(LLMError, match="did not produce valid Verdict"):
        await router.structured([{"role": "user", "content": "x"}], Verdict, retries=1)


def test_extract_json_unwraps_code_fences() -> None:
    assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_ignores_preamble_prose() -> None:
    """Models add 'Here is the JSON:' despite instructions not to."""
    assert _extract_json('Sure! Here is the result:\n{"a": 1}\nHope that helps.') == '{"a": 1}'


def test_extract_json_handles_arrays() -> None:
    assert _extract_json("prefix [1, 2] suffix") == "[1, 2]"


def test_extract_json_passes_through_clean_json() -> None:
    assert _extract_json('{"a": 1}') == '{"a": 1}'


async def test_usage_is_recorded_for_cost_accounting() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body("hi"))

    result = await router_with(handler).complete([{"role": "user", "content": "x"}])
    assert result.usage.total == 15


async def test_two_hundred_with_an_error_body_fails_over() -> None:
    """OpenRouter reports upstream failures with a 200 and no choices.

    Indexing choices[0] blindly crashed a live audit; the shape is checked now.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if "groq" in request.url.host:
            return httpx.Response(200, json={"error": {"message": "upstream is down"}})
        return httpx.Response(200, json=completion_body("recovered"))

    router = router_with(handler, openrouter_api_key="or-key")
    assert (await router.complete([{"role": "user", "content": "x"}])).text == "recovered"


async def test_short_rate_limit_waits_rather_than_failing_over() -> None:
    """Groq recovers in seconds and is faster than the fallbacks."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0.01"}, text="slow down")
        return httpx.Response(200, json=completion_body("after waiting"))

    router = router_with(handler, openrouter_api_key="or-key")
    result = await router.complete([{"role": "user", "content": "x"}])

    assert result.text == "after waiting"
    assert result.provider is LLMProvider.GROQ


async def test_long_rate_limit_fails_over_instead_of_waiting() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "groq" in request.url.host:
            return httpx.Response(429, headers={"retry-after": "600"}, text="try later")
        return httpx.Response(200, json=completion_body("fallback"))

    router = router_with(handler, openrouter_api_key="or-key")
    result = await router.complete([{"role": "user", "content": "x"}])

    assert result.provider is LLMProvider.OPENROUTER
