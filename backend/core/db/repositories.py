"""Typed data access.

Business logic depends on these methods, never on raw SQL. Two consequences
that matter beyond tidiness: the pgvector and full-text query syntax stays in
one place where it can be tuned, and ingestion idempotency is enforced here
rather than trusted to every caller.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.models import AuditLog, Chunk, Citation, Document, DocumentKind


def content_hash(text: str) -> str:
    """Stable identity for a document body, used to skip unchanged re-ingests."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self._session.get(Document, document_id)

    async def get_by_source_ref(self, kind: DocumentKind, source_ref: str) -> Document | None:
        result = await self._session.execute(
            select(Document).where(Document.kind == kind, Document.source_ref == source_ref)
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        *,
        kind: DocumentKind,
        title: str,
        source_ref: str,
        raw_text: str,
        source_url: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> tuple[Document, bool]:
        """Insert or update by (kind, source_ref).

        Returns `(document, changed)`. `changed` is False when the stored
        content hash already matches, which lets ingestion skip re-chunking and
        re-embedding — the expensive half of the pipeline.
        """
        digest = content_hash(raw_text)
        existing = await self.get_by_source_ref(kind, source_ref)

        if existing is not None:
            if existing.content_hash == digest:
                return existing, False
            existing.title = title
            existing.source_url = source_url
            existing.content_hash = digest
            existing.meta = meta or {}
            await self._session.flush()
            return existing, True

        document = Document(
            kind=kind,
            title=title,
            source_ref=source_ref,
            source_url=source_url,
            content_hash=digest,
            meta=meta or {},
        )
        self._session.add(document)
        await self._session.flush()
        return document, True

    async def list_by_kind(
        self, kind: DocumentKind, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Document]:
        result = await self._session.execute(
            select(Document)
            .where(Document.kind == kind)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def delete(self, document_id: uuid.UUID) -> bool:
        result = await self._session.execute(delete(Document).where(Document.id == document_id))
        return bool(cast(CursorResult[Any], result).rowcount)


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_for_document(
        self, document_id: uuid.UUID, chunks: Sequence[dict[str, Any]]
    ) -> list[Chunk]:
        """Atomically swap a document's chunks.

        Re-chunking must not leave stale spans behind: a shorter revision would
        otherwise keep the old tail, and the verifier would happily cite text
        that no longer exists in the source.
        """
        await self._session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        rows = [
            Chunk(
                document_id=document_id,
                ordinal=c["ordinal"],
                label=c.get("label"),
                content=c["content"],
                token_count=c.get("token_count", 0),
                embedding=c.get("embedding"),
                meta=c.get("meta") or {},
            )
            for c in chunks
        ]
        self._session.add_all(rows)
        await self._session.flush()
        return rows

    async def get(self, chunk_id: uuid.UUID) -> Chunk | None:
        return await self._session.get(Chunk, chunk_id)

    async def get_many(self, chunk_ids: Sequence[uuid.UUID]) -> Sequence[Chunk]:
        """Bulk fetch for the citation verifier — one query, not N."""
        if not chunk_ids:
            return []
        result = await self._session.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))
        return result.scalars().all()

    async def list_for_document(self, document_id: uuid.UUID) -> Sequence[Chunk]:
        result = await self._session.execute(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.ordinal)
        )
        return result.scalars().all()

    async def pending_embedding(
        self, *, limit: int = 128, document_id: uuid.UUID | None = None
    ) -> Sequence[Chunk]:
        """Chunks whose text has landed but whose vector has not.

        Filtering happens in the query rather than in the caller: a post-filter
        would let unrelated pending rows fill the batch and stall a
        document-scoped backfill indefinitely.
        """
        stmt = select(Chunk).where(Chunk.embedding.is_(None))
        if document_id is not None:
            stmt = stmt.where(Chunk.document_id == document_id)
        result = await self._session.execute(stmt.limit(limit))
        return result.scalars().all()

    async def set_embeddings(self, embeddings: dict[uuid.UUID, list[float]]) -> int:
        if not embeddings:
            return 0
        rows = await self.get_many(list(embeddings))
        for row in rows:
            row.embedding = embeddings[row.id]
        await self._session.flush()
        return len(rows)

    async def count(self) -> int:
        result = await self._session.execute(select(func.count()).select_from(Chunk))
        return int(result.scalar_one())


class CitationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_many(self, source_document_id: uuid.UUID, edges: Sequence[dict[str, Any]]) -> int:
        """Idempotent bulk insert.

        Re-ingesting a judgment must not multiply its citation edges, so
        conflicts on (source, target_ref) are ignored rather than erroring.
        """
        if not edges:
            return 0
        stmt = (
            insert(Citation)
            .values(
                [
                    {
                        "source_document_id": source_document_id,
                        "target_document_id": e.get("target_document_id"),
                        "target_ref": e["target_ref"],
                        "context": e.get("context"),
                    }
                    for e in edges
                ]
            )
            .on_conflict_do_nothing(constraint="uq_citations_source_target")
        )
        result = await self._session.execute(stmt)
        return int(cast(CursorResult[Any], result).rowcount or 0)

    async def cited_by(self, document_id: uuid.UUID) -> Sequence[Citation]:
        """Inbound edges — how often a judgment is relied upon."""
        result = await self._session.execute(
            select(Citation).where(Citation.target_document_id == document_id)
        )
        return result.scalars().all()

    async def cites(self, document_id: uuid.UUID) -> Sequence[Citation]:
        """Outbound edges — what this judgment relies upon."""
        result = await self._session.execute(
            select(Citation).where(Citation.source_document_id == document_id)
        )
        return result.scalars().all()

    async def resolve_targets(self) -> int:
        """Link edges whose target has since been ingested.

        Citations are recorded by reference string as soon as they are parsed,
        long before the cited judgment necessarily exists locally. This backfills
        the foreign key once it does, turning dangling refs into graph edges.
        """
        result = await self._session.execute(
            select(Citation).where(Citation.target_document_id.is_(None))
        )
        pending = result.scalars().all()
        if not pending:
            return 0

        refs = {c.target_ref for c in pending}
        matches = await self._session.execute(select(Document).where(Document.source_ref.in_(refs)))
        by_ref = {d.source_ref: d.id for d in matches.scalars().all()}

        linked = 0
        for citation in pending:
            target_id = by_ref.get(citation.target_ref)
            # Guard the self-reference constraint: a document citing its own
            # source_ref would otherwise violate ck_citations_no_self_reference.
            if target_id is not None and target_id != citation.source_document_id:
                citation.target_document_id = target_id
                linked += 1

        await self._session.flush()
        return linked


class AuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        run_id: str,
        module: str,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> AuditLog:
        entry = AuditLog(run_id=run_id, module=module, event=event, payload=payload or {})
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def for_run(self, run_id: str) -> Sequence[AuditLog]:
        result = await self._session.execute(
            select(AuditLog).where(AuditLog.run_id == run_id).order_by(AuditLog.created_at)
        )
        return result.scalars().all()
