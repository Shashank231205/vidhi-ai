"""Ingest Indian statutes into the corpus.

Sources are official government PDFs, fetched at run time rather than vendored:
the corpus is reproducible from the script, and nothing large lands in git.

    uv run python ../scripts/ingest_statutes.py            # all configured
    uv run python ../scripts/ingest_statutes.py --only dpdp
    uv run python ../scripts/ingest_statutes.py --force    # re-chunk unchanged
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import httpx
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from core.cache import Cache  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.db import Database, DocumentKind  # noqa: E402
from core.embeddings import EmbeddingService  # noqa: E402
from core.ingestion.pipeline import IngestionPipeline  # noqa: E402
from core.logging import configure_logging, get_logger  # noqa: E402

log = get_logger("ingest")


@dataclass(frozen=True, slots=True)
class StatuteSource:
    key: str
    title: str
    source_ref: str
    url: str
    year: int
    act_number: str


STATUTES: tuple[StatuteSource, ...] = (
    StatuteSource(
        key="dpdp",
        title="Digital Personal Data Protection Act, 2023",
        source_ref="DPDP-2023",
        url="https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf",
        year=2023,
        act_number="22 of 2023",
    ),
)


async def fetch_pdf_text(url: str) -> str:
    """Download a PDF and extract its text."""
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

    reader = PdfReader(BytesIO(response.content))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise RuntimeError(f"no extractable text in {url} (scanned image?)")
    return text


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="ingest a single statute by key")
    parser.add_argument(
        "--force", action="store_true", help="re-chunk even if unchanged"
    )
    parser.add_argument(
        "--no-embed", action="store_true", help="skip embeddings (text only)"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)

    selected = [s for s in STATUTES if not args.only or s.key == args.only]
    if not selected:
        available = ", ".join(s.key for s in STATUTES)
        print(f"unknown statute '{args.only}'. Available: {available}", file=sys.stderr)
        return 1

    database = Database(settings)
    cache = Cache(settings)
    embeddings = EmbeddingService(settings, cache=cache)
    pipeline = IngestionPipeline(database, embeddings)

    failures = 0
    try:
        for statute in selected:
            print(f"→ {statute.title}")
            try:
                text = await fetch_pdf_text(statute.url)
                print(f"  fetched {len(text):,} characters")

                result = await pipeline.ingest(
                    kind=DocumentKind.STATUTE,
                    title=statute.title,
                    source_ref=statute.source_ref,
                    raw_text=text,
                    source_url=statute.url,
                    meta={
                        "year": statute.year,
                        "act_number": statute.act_number,
                        "jurisdiction": "India",
                    },
                    force=args.force,
                    embed=not args.no_embed,
                )
                print(f"  {result.summary}")
            except Exception as exc:
                failures += 1
                log.exception("ingest_failed", statute=statute.key, error=str(exc))
                print(f"  FAILED: {exc}", file=sys.stderr)
    finally:
        await embeddings.close()
        await cache.close()
        await database.dispose()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
