"""Application entrypoint.

Domain modules (compliance, caselens) mount their routers here as they land in
later phases; this file stays thin — wiring only, no business logic.

Served as a factory (`uvicorn api.main:create_app --factory`) so that importing
this module never reads settings — tests construct their own.
"""

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import __version__
from api.routes import router as api_router
from core.config import Settings, get_settings
from core.logging import configure_logging, get_logger

log = get_logger(__name__)

#: Correlates every log line and trace event emitted while serving one request.
REQUEST_ID_HEADER = "X-Request-ID"


def _lifespan(settings: Settings) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Bound to the app's own settings, so tests never touch the environment."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings)

        providers = settings.configured_providers()
        if not providers:
            # Not fatal at boot: /health reports it, and the LLM router raises at
            # call time. Failing startup would make the container un-debuggable.
            log.warning(
                "no_llm_provider_configured",
                checked=[p.value for p in settings.llm_provider_order],
            )

        log.info(
            "startup",
            version=__version__,
            environment=settings.environment,
            llm_providers=[p.value for p in providers],
        )
        yield
        log.info("shutdown")

    return lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    """Factory so tests can build an app with overridden settings."""
    settings = settings or get_settings()

    app = FastAPI(
        title="VidhiAI",
        description="Unified AI legal platform for Indian law.",
        version=__version__,
        lifespan=_lifespan(settings),
        docs_url="/docs" if settings.is_local else None,
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # SSE consumers read the request id off the response to correlate traces.
        expose_headers=[REQUEST_ID_HEADER],
    )

    @app.middleware("http")
    async def bind_request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Log the cause, return an opaque body — never leak internals upstream."""
        log.exception("unhandled_error", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    # Routes depend on `get_settings`; point it at this app's instance so the
    # factory argument is authoritative everywhere, not just at startup.
    app.dependency_overrides[get_settings] = lambda: settings

    # Domain routers (compliance, caselens) mount here in Phases 3 and 5.
    app.include_router(api_router)
    return app
