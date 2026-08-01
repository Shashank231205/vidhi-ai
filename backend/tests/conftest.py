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
from core.config import EmbeddingBackend, Environment, Settings
from core.embeddings import EmbeddingBackendProtocol, Vector


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
        # Unit tests must never load the local model: it costs seconds per
        # test and gigabytes of memory. Integration tests use real settings.
        "embedding_backend": EmbeddingBackend.REMOTE,
        # Pinned off for the same reason the credentials are pinned: a
        # developer with a trained model in .env must not get different
        # results from CI.
        "risk_classifier_model": None,
        "stance_classifier_model": None,
    }
    return Settings(**{**defaults, **overrides})  # type: ignore[arg-type]


@pytest.fixture
def settings() -> Settings:
    return build_settings()


class StubEmbeddings(EmbeddingBackendProtocol):
    """Deterministic vectors, no model and no network.

    Unit tests care that wiring and shapes are right, not that BGE-M3 produces
    good embeddings — that is what the retrieval eval measures.
    """

    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    async def encode(self, texts: list[str]) -> list[Vector]:
        self.calls.append(texts)
        return [[((hash(t) >> i) % 100) / 100.0 for i in range(self.dimensions)] for t in texts]


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


@pytest.fixture(scope="session")
def dpdp_pdf() -> bytes:
    """A small PDF that genuinely carries a text layer.

    Built by hand rather than downloaded: the test suite must not depend on a
    network round trip, and pypdf's blank pages have no extractable text.
    """
    body = (
        "5. Notice. Every request made to a Data Principal for consent shall be "
        "accompanied by a notice informing her of the personal data and the "
        "purpose of processing. The Data Fiduciary shall provide the notice in "
        "clear and plain language."
    )
    stream = f"BT /F1 11 Tf 40 700 Td ({body}) Tj ET".encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)
