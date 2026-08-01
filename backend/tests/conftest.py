"""Shared fixtures.

Unit tests construct Settings explicitly rather than reading a `.env`, so the
default suite is hermetic and never depends on a developer's credentials.

Tests marked `integration` are the deliberate exception: they exercise real
Postgres behaviour and are skipped unless a `.env` supplies a database.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import create_app
from core.config import Environment, Settings


def build_settings(**overrides: object) -> Settings:
    """Fully specified fake settings.

    Every optional credential is listed explicitly — including the ones set to
    None. Omitting a field would let pydantic-settings fall back to the real
    environment, so a developer with a populated .env would see different
    results from CI.
    """
    defaults: dict[str, object] = {
        "environment": Environment.LOCAL,
        "database_url": "postgresql://postgres:pw@db.example.supabase.com:6543/postgres",
        "upstash_redis_url": "https://example.upstash.io",
        "upstash_redis_token": "test-redis-token",
        "hf_api_token": "test-hf-token",
        "groq_api_key": "test-groq-key",
        "cerebras_api_key": None,
        "openrouter_api_key": None,
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


# --- Integration support -------------------------------------------------
#
# Integration tests need real credentials. Rather than fail without them, they
# skip — so `make test` works for a contributor who has not set up Supabase,
# while still running for real locally and against a provisioned environment.


def _read_env_file() -> dict[str, str]:
    """Parse backend/.env into a dict.

    Deliberately does not mutate os.environ: unit tests construct settings
    explicitly, and leaking real credentials into the process environment would
    make their results depend on whether the developer has a .env.
    """
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _database_configured() -> bool:
    return bool(os.environ.get("DATABASE_URL") or _read_env_file().get("DATABASE_URL"))


@pytest.fixture(scope="session")
def integration_settings() -> Settings:
    if not _database_configured():
        pytest.skip("DATABASE_URL not configured; skipping integration tests")
    # Real credentials, scoped to the fixture rather than the process.
    return Settings(**_read_env_file())  # type: ignore[arg-type]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless a database is configured."""
    if _database_configured():
        return
    skip = pytest.mark.skip(reason="no DATABASE_URL; integration tests skipped")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
