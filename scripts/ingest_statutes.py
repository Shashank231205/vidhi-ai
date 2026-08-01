"""Ingest Indian central Acts into the corpus.

Sources are official government PDFs (India Code and ministry sites), fetched
at run time rather than vendored: the corpus is reproducible from this script
and nothing large lands in git.

    uv run python ../scripts/ingest_statutes.py                 # priority 1
    uv run python ../scripts/ingest_statutes.py --all           # everything
    uv run python ../scripts/ingest_statutes.py --only dpdp it
    uv run python ../scripts/ingest_statutes.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from core.cache import Cache  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.db import Database, DocumentKind  # noqa: E402
from core.embeddings import EmbeddingService  # noqa: E402
from core.ingestion.fetch import fetch_statute_text  # noqa: E402
from core.ingestion.pipeline import IngestionPipeline  # noqa: E402
from core.ingestion.sources import BY_KEY, STATUTES, by_priority  # noqa: E402
from core.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("ingest")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", metavar="KEY", help="ingest these Acts")
    parser.add_argument("--all", action="store_true", help="ingest every Act")
    parser.add_argument(
        "--priority", type=int, default=1, help="ingest Acts up to this priority"
    )
    parser.add_argument("--list", action="store_true", help="list Acts and exit")
    parser.add_argument("--force", action="store_true", help="re-chunk unchanged Acts")
    parser.add_argument("--no-embed", action="store_true", help="skip embeddings")
    args = parser.parse_args()

    if args.list:
        print(f"{'key':16} {'priority':>8}  {'domain':22} title")
        for source in sorted(STATUTES, key=lambda s: (s.priority, s.key)):
            print(
                f"{source.key:16} {source.priority:>8}  "
                f"{source.domain.value:22} {source.title}"
            )
        return 0

    if args.only:
        unknown = [k for k in args.only if k not in BY_KEY]
        if unknown:
            print(f"unknown: {', '.join(unknown)}", file=sys.stderr)
            print(f"available: {', '.join(sorted(BY_KEY))}", file=sys.stderr)
            return 1
        selected = [BY_KEY[k] for k in args.only]
    else:
        selected = by_priority(99 if args.all else args.priority)

    settings = get_settings()
    configure_logging(settings)

    database = Database(settings)
    cache = Cache(settings)
    embeddings = EmbeddingService(settings, cache=cache)
    pipeline = IngestionPipeline(database, embeddings)

    print(f"Ingesting {len(selected)} Act(s) with {settings.embedding_backend} embeddings\n")

    failures: list[str] = []
    started = time.perf_counter()

    try:
        # Warm the model once rather than on the first Act's first batch.
        if not args.no_embed:
            await embeddings.warm()

        async with httpx.AsyncClient(timeout=120.0) as client:
            for source in selected:
                print(f"→ {source.title}")
                try:
                    text, resolved_url = await fetch_statute_text(client, source)
                    print(f"  fetched {len(text):,} characters")

                    began = time.perf_counter()
                    result = await pipeline.ingest(
                        kind=DocumentKind.STATUTE,
                        title=source.title,
                        source_ref=source.source_ref,
                        raw_text=text,
                        source_url=resolved_url,
                        meta=source.meta,
                        force=args.force,
                        embed=not args.no_embed,
                    )
                    print(f"  {result.summary} ({time.perf_counter() - began:.1f}s)")
                except Exception as exc:
                    failures.append(source.key)
                    log.exception("ingest_failed", key=source.key, error=str(exc))
                    print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        await embeddings.close()
        await cache.close()
        await database.dispose()

    elapsed = time.perf_counter() - started
    print(f"\nDone in {elapsed:.1f}s — {len(selected) - len(failures)}/{len(selected)} ingested")
    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
