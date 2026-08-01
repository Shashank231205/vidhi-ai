"""Structured logging.

JSON in deployed environments so logs are queryable; human-readable colour in
local development. Configured once at app startup, never per-module.
"""

import logging
import sys
from typing import Any

import structlog

from core.config import Settings


def configure_logging(settings: Settings) -> None:
    """Idempotent — safe to call from both app startup and test fixtures."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
        force=True,
    )

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.dev.ConsoleRenderer()
        if settings.is_local
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, settings.log_level)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
