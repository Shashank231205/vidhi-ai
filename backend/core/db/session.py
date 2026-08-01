"""Engine and session lifecycle.

Supabase is reached through its transaction pooler (PgBouncer), which imposes
two constraints that are easy to get wrong and painful to debug:

- Server-side prepared statements must be disabled. PgBouncer multiplexes
  connections, so a statement prepared on one backend may be executed on
  another. asyncpg caches prepared statements by default, which surfaces as
  intermittent "prepared statement does not exist" errors under load.
- The pool is kept small. The free tier caps total connections, and the pooler
  is already doing the heavy multiplexing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import Settings
from core.logging import get_logger

log = get_logger(__name__)


def _asyncpg_url(dsn: str) -> str:
    """SQLAlchemy needs the driver named explicitly in the scheme."""
    if dsn.startswith("postgresql+"):
        return dsn
    return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        _asyncpg_url(str(settings.database_url)),
        pool_size=settings.db_pool_min_size,
        max_overflow=settings.db_pool_max_size - settings.db_pool_min_size,
        pool_pre_ping=True,  # a pooled connection can be closed server-side
        pool_recycle=1800,
        connect_args={
            # Required under PgBouncer transaction pooling — see module docstring.
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
            "command_timeout": settings.db_command_timeout_s,
            "server_settings": {"application_name": "vidhi-ai"},
        },
    )


class Database:
    """Owns the engine and hands out sessions.

    One instance per process, created at startup and disposed at shutdown.
    """

    def __init__(self, settings: Settings) -> None:
        self._engine = create_engine(settings)
        self._sessionmaker = async_sessionmaker(
            self._engine,
            expire_on_commit=False,  # objects stay usable after the session closes
            autoflush=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Transactional scope: commits on success, rolls back on any exception."""
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def healthcheck(self) -> bool:
        """True when the database answers a trivial query."""
        from sqlalchemy import text

        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            log.warning("db_healthcheck_failed", error=str(exc))
            return False

    async def dispose(self) -> None:
        await self._engine.dispose()
