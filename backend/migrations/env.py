"""Alembic environment.

The database URL comes from Settings rather than alembic.ini, so the credential
lives only in .env and migrations run against the same target as the app.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import AsyncEngine

from core.config import get_settings
from core.db.models import Base
from core.db.session import create_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _include_object(obj, name, type_, reflected, compare_to):  # type: ignore[no-untyped-def]
    """Ignore tables Supabase manages itself (auth, storage, realtime)."""
    return not (type_ == "table" and getattr(obj, "schema", None) not in (None, "public"))


def run_migrations_offline() -> None:
    settings = get_settings()
    context.configure(
        url=str(settings.database_url),
        target_metadata=target_metadata,
        literal_binds=True,
        include_object=_include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection) -> None:  # type: ignore[no-untyped-def]
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=_include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine: AsyncEngine = create_engine(get_settings())
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
