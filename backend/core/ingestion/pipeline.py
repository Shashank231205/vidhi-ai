"""Ingestion: raw text in, embedded and searchable chunks out.

Idempotent by construction. Re-running against an unchanged source is nearly
free — the content hash short-circuits before any chunking or embedding — which
matters because embedding is the slow, rate-limited step.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from core.db import ChunkRepository, Database, DocumentKind, DocumentRepository
from core.embeddings import EmbeddingService
from core.ingestion.chunker import chunk_legal_text
from core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class IngestionResult:
    document_id: str
    title: str
    chunks_written: int
    chunks_embedded: int
    skipped: bool

    @property
    def summary(self) -> str:
        if self.skipped:
            return f"{self.title}: unchanged, skipped"
        return f"{self.title}: {self.chunks_written} chunks, {self.chunks_embedded} embedded"


class IngestionPipeline:
    def __init__(self, database: Database, embeddings: EmbeddingService) -> None:
        self._db = database
        self._embeddings = embeddings

    async def ingest(
        self,
        *,
        kind: DocumentKind,
        title: str,
        source_ref: str,
        raw_text: str,
        source_url: str | None = None,
        meta: dict[str, object] | None = None,
        force: bool = False,
        embed: bool = True,
    ) -> IngestionResult:
        """Ingest one document end to end.

        Text is committed before embedding starts, so a failure partway through
        a slow embedding run leaves the corpus searchable lexically rather than
        rolling back work that already succeeded.
        """
        async with self._db.session() as session:
            documents = DocumentRepository(session)
            document, changed = await documents.upsert(
                kind=kind,
                title=title,
                source_ref=source_ref,
                raw_text=raw_text,
                source_url=source_url,
                meta=dict(meta or {}),
            )
            document_id = document.id

            if not changed and not force:
                log.info("ingest_skipped_unchanged", source_ref=source_ref)
                return IngestionResult(
                    document_id=str(document_id),
                    title=title,
                    chunks_written=0,
                    chunks_embedded=0,
                    skipped=True,
                )

            chunks = chunk_legal_text(raw_text)
            await ChunkRepository(session).replace_for_document(
                document_id, [c.as_row() for c in chunks]
            )

        log.info("ingest_chunks_written", source_ref=source_ref, chunks=len(chunks))

        embedded = await self.embed_pending(document_id=document_id) if embed else 0
        return IngestionResult(
            document_id=str(document_id),
            title=title,
            chunks_written=len(chunks),
            chunks_embedded=embedded,
            skipped=False,
        )

    async def embed_pending(self, *, document_id: uuid.UUID | None = None, batch: int = 64) -> int:
        """Backfill embeddings for chunks that have none.

        Runs in batches with a session per batch, so a long backfill never holds
        one transaction open across many slow network calls.
        """
        total = 0
        while True:
            async with self._db.session() as session:
                repository = ChunkRepository(session)
                pending = list(
                    await repository.pending_embedding(limit=batch, document_id=document_id)
                )
                if not pending:
                    return total

                vectors = await self._embeddings.embed([c.content for c in pending])
                written = await repository.set_embeddings(
                    dict(zip([c.id for c in pending], vectors, strict=True))
                )
                total += written

            log.info("embedded_batch", written=written, cumulative=total)
