"""Shared fixtures.

Settings are constructed explicitly rather than read from a `.env`, so the suite
is hermetic and never depends on a developer's local credentials.
"""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import create_app
from core.config import Environment, Settings


def build_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "environment": Environment.LOCAL,
        "database_url": "postgresql://postgres:pw@db.example.supabase.com:6543/postgres",
        "upstash_redis_url": "https://example.upstash.io",
        "upstash_redis_token": "test-redis-token",
        "hf_api_token": "test-hf-token",
        "groq_api_key": "test-groq-key",
    }
    return Settings(**{**defaults, **overrides})  # type: ignore[arg-type]


@pytest.fixture
def settings() -> Settings:
    return build_settings()


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    # create_app already binds these settings to the settings dependency.
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
