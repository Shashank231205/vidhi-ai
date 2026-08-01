"""App wiring: request correlation, error opacity, environment-gated docs."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import REQUEST_ID_HEADER, create_app
from core.config import Environment
from tests.conftest import build_settings


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    assert client.get("/health").headers[REQUEST_ID_HEADER]


def test_incoming_request_id_is_preserved(client: TestClient) -> None:
    """Lets a trace be followed across the frontend/API boundary."""
    response = client.get("/health", headers={REQUEST_ID_HEADER: "abc-123"})
    assert response.headers[REQUEST_ID_HEADER] == "abc-123"


def test_request_ids_are_unique_per_request(client: TestClient) -> None:
    first = client.get("/health").headers[REQUEST_ID_HEADER]
    second = client.get("/health").headers[REQUEST_ID_HEADER]
    assert first != second


def test_unhandled_errors_return_opaque_500(app: FastAPI) -> None:
    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("supabase password is hunter2")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert "hunter2" not in response.text


def test_docs_are_disabled_outside_local() -> None:
    app = create_app(build_settings(environment=Environment.PRODUCTION))
    assert app.docs_url is None

    with TestClient(app) as client:
        assert client.get("/docs").status_code == 404


def test_cors_allows_configured_origin(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_exposes_request_id_header(client: TestClient) -> None:
    """The browser can only read the header if it is explicitly exposed."""
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert REQUEST_ID_HEADER in response.headers["access-control-expose-headers"]
