"""Ingest Indian judgments and build the citation graph.

    uv run python ../scripts/ingest_judgments.py --limit 200
    uv run python ../scripts/ingest_judgments.py --limit 50 --no-embed
    uv run python ../scripts/ingest_judgments.py --resolve-only

Citation edges are recorded as reference strings when a judgment is parsed,
long before the cited case is necessarily in the corpus. `--resolve-only`
re-links those dangling references once more judgments have landed, which is
what turns a list of citations into a navigable graph.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from caselens.ingestion import HuggingFaceJudgments  # noqa: E402
from core.cache import Cache  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.db import (  # noqa: E402
    ChunkRepository,
    CitationRepository,
    Database,
    DocumentKind,
    DocumentRepository,
)
from core.embeddings import EmbeddingService  # noqa: E402
from core.ingestion.chunker import chunk_legal_text  # noqa: E402
from core.ingestion.pipeline import IngestionPipeline  # noqa: E402
from core.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("ingest_judgments")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100, help="judgments to ingest")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dataset", default="ninadn/indian-legal")
    parser.add_argument("--text-column", default="Text")
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="only re-link dangling citations; ingest nothing",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)

    database = Database(settings)
    cache = Cache(settings)
    embeddings = EmbeddingService(settings, cache=cache)
    pipeline = IngestionPipeline(database, embeddings)

    try:
        if args.resolve_only:
            async with database.session() as session:
                linked = await CitationRepository(session).resolve_targets()
            print(f"Linked {linked} previously dangling citation(s).")
            return 0

        source = HuggingFaceJudgments(
            dataset=args.dataset, text_column=args.text_column
        )
        if not args.no_embed:
            await embeddings.warm()

        started = time.perf_counter()
        ingested = skipped = failed = 0
        edges = 0

        try:
            async for judgment in source.stream(args.limit, args.offset):
                try:
                    # Text and citation edges are written in one transaction:
                    # a judgment whose edges failed to record would silently
                    # weaken the graph with nothing to indicate it.
                    async with database.session() as session:
                        documents = DocumentRepository(session)
                        document, changed = await documents.upsert(
                            kind=DocumentKind.JUDGMENT,
                            title=judgment.title,
                            source_ref=judgment.source_ref,
                            raw_text=judgment.text,
                            meta=judgment.meta,
                        )
                        if not changed:
                            skipped += 1
                            continue

                        chunks = chunk_legal_text(judgment.text)
                        await ChunkRepository(session).replace_for_document(
                            document.id, [c.as_row() for c in chunks]
                        )
                        edges += await CitationRepository(session).add_many(
                            document.id,
                            [
                                {"target_ref": c.target_ref, "context": c.context}
                                for c in judgment.citations
                            ],
                        )
                    ingested += 1

                    if ingested % 25 == 0:
                        print(f"  {ingested} ingested, {edges} citation edges")
                except Exception as exc:
                    failed += 1
                    log.exception(
                        "judgment_failed", ref=judgment.source_ref, error=str(exc)
                    )
        finally:
            await source.close()

        if not args.no_embed:
            print("Embedding new chunks...")
            embedded = await pipeline.embed_pending()
            print(f"  {embedded} chunks embedded")

        # Now that more judgments exist, previously dangling references may
        # resolve to real documents.
        async with database.session() as session:
            linked = await CitationRepository(session).resolve_targets()

        elapsed = time.perf_counter() - started
        print(
            f"\nDone in {elapsed:.1f}s — {ingested} ingested, {skipped} unchanged, "
            f"{failed} failed\n{edges} citation edges recorded, {linked} resolved to "
            f"ingested judgments"
        )
        return 1 if failed and not ingested else 0
    finally:
        await embeddings.close()
        await cache.close()
        await database.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
