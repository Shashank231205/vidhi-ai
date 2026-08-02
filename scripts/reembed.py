"""Re-embed every chunk after an embedding-model change.

Vectors from different models occupy different spaces, so a model swap
invalidates the whole corpus: a migration that re-dimensions the column nulls it
with `USING NULL` rather than casting, and this refills it. Also backfills rows
that ingestion landed as text without embedding.

    uv run python ../scripts/reembed.py
    uv run python ../scripts/reembed.py --batch 128
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from core.config import get_settings
from core.db import ChunkRepository, Database
from core.embeddings import EmbeddingService
from core.logging import configure_logging


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)

    database = Database(settings)
    # No cache: entries are keyed by model, so a swap misses everything anyway,
    # and writing 9,000 entries we are about to stop using wastes the quota.
    embeddings = EmbeddingService(settings, cache=None)

    print(f"Re-embedding with {settings.embedding_model} ({settings.embedding_dimensions}d)")

    try:
        await embeddings.warm()

        async with database.session() as session:
            total = await ChunkRepository(session).count()
        print(f"{total} chunks in the corpus\n")

        started = time.perf_counter()
        done = 0
        while True:
            async with database.session() as session:
                repository = ChunkRepository(session)
                pending = list(await repository.pending_embedding(limit=args.batch))
                if not pending:
                    break

                vectors = await embeddings.embed([c.content for c in pending])
                await repository.set_embeddings(
                    dict(zip([c.id for c in pending], vectors, strict=True))
                )

            done += len(pending)
            elapsed = time.perf_counter() - started
            rate = done / max(elapsed, 0.001)
            remaining = (total - done) / max(rate, 0.001)
            print(
                f"  {done}/{total}  {rate:.0f}/s  ~{remaining / 60:.1f} min left",
                end="\r",
                flush=True,
            )

        print(f"\n\nDone: {done} chunks in {(time.perf_counter() - started) / 60:.1f} min")
        return 0
    finally:
        await embeddings.close()
        await database.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
