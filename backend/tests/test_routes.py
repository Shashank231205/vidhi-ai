"""/health is the deploy gate — it must be honest about what is missing."""

from fastapi.testclient import TestClient

from api.main import create_app
from tests.conftest import build_settings


def test_health_ok_when_fully_configured(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "local"
    assert all(c["configured"] for c in body["components"].values())


def test_health_reports_degraded_with_no_llm_key() -> None:
    """A missing provider key must surface here, not at first user query."""
    app = create_app(build_settings(groq_api_key=None))

    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["components"]["llm"]["configured"] is False
    assert "no provider API key" in body["components"]["llm"]["detail"]


def test_health_never_leaks_secret_values(client: TestClient) -> None:
    raw = client.get("/health").text

    for secret in ("test-redis-token", "test-hf-token", "test-groq-key"):
        assert secret not in raw


def test_llm_detail_lists_failover_order(client: TestClient) -> None:
    detail = client.get("/health").json()["components"]["llm"]["detail"]
    assert "groq" in detail
