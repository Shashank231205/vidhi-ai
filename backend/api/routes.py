"""Gateway routes.

Every endpoint owned by the API layer itself lives here — one routes module per
package, so there is a single place to look for what a module exposes. Domain
endpoints live in `compliance/routes.py` and `caselens/routes.py`.

Liveness and readiness are deliberately separate:

- `/health` answers from configuration alone and never touches the network, so
  a hosting platform can use it as a restart signal without a slow upstream
  turning into a restart loop.
- `/ready` actually probes the database, and is what a load balancer should
  gate traffic on.
"""

from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel

from api import __version__
from core.config import Environment, Settings, get_settings
from core.db import Database

router = APIRouter(tags=["health"])


class Status(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"


class ComponentReport(BaseModel):
    """Whether a dependency is *configured* — not whether it is reachable."""

    configured: bool
    detail: str


class HealthResponse(BaseModel):
    status: Status
    environment: Environment
    version: str
    components: dict[str, ComponentReport]


def _report(configured: bool, ready: str, missing: str) -> ComponentReport:
    return ComponentReport(configured=configured, detail=ready if configured else missing)


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    providers = settings.configured_providers()

    components = {
        "database": _report(
            bool(settings.database_url), "connection string set", "DATABASE_URL missing"
        ),
        "cache": _report(
            bool(settings.upstash_redis_token.get_secret_value()),
            "upstash credentials set",
            "UPSTASH_REDIS_TOKEN missing",
        ),
        "llm": _report(
            bool(providers),
            f"failover order: {', '.join(providers)}",
            "no provider API key set",
        ),
        "embeddings": _report(
            bool(settings.hf_api_token.get_secret_value()),
            f"{settings.embedding_model} ({settings.embedding_dimensions}d)",
            "HF_API_TOKEN missing",
        ),
    }

    degraded = any(not c.configured for c in components.values())
    return HealthResponse(
        status=Status.DEGRADED if degraded else Status.OK,
        environment=settings.environment,
        version=__version__,
        components=components,
    )


class ReadyResponse(BaseModel):
    ready: bool
    database: bool


def get_database(request: Request) -> Database:
    """The engine created at startup; one per process."""
    return request.app.state.db  # type: ignore[no-any-return]


@router.get("/ready", response_model=ReadyResponse)
async def ready(
    response: Response,
    database: Annotated[Database, Depends(get_database)],
) -> ReadyResponse:
    """Readiness: can this instance actually serve a request?

    Returns 503 when the database is unreachable so a load balancer stops
    sending traffic here, rather than reporting 200 and failing every query.
    """
    db_ok = await database.healthcheck()
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(ready=db_ok, database=db_ok)
