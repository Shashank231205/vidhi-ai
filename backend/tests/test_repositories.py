"""Repository integration tests.

These run against the real Supabase instance, because the behaviour worth
testing here — pgvector round-tripping, the tsvector trigger, cascade deletes,
constraint enforcement — is behaviour of Postgres, not of Python. A mocked
session would assert only that SQLAlchemy was called.

Skipped automatically when DATABASE_URL is absent, so CI stays hermetic.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text

from core.db import (
    ChunkRepository,
    CitationRepository,
    Database,
    DocumentKind,
    DocumentRepository,
    content_hash,
)
from core.db.repositories import AuditRepository

pytestmark = pytest.mark.integration


@pytest.fixture
async def db(integration_settings) -> AsyncIterator[Database]:  # type: ignore[no-untyped-def]
    database = Database(integration_settings)
    yield database
    await database.dispose()


@pytest.fixture
async def clean_documents(db: Database) -> AsyncIterator[None]:
    """Remove only this test module's rows, identified by a source_ref prefix.

    Never truncates: the same database holds ingested corpus, and a test run
    must not be able to destroy it.
    """
    yield
    async with db.session() as session:
        await session.execute(text("DELETE FROM documents WHERE source_ref LIKE 'test-%'"))


def _ref() -> str:
    return f"test-{uuid.uuid4().hex[:12]}"


async def test_upsert_creates_then_detects_unchanged(db: Database, clean_documents: None) -> None:
    ref = _ref()
    async with db.session() as session:
        repo = DocumentRepository(session)
        doc, changed = await repo.upsert(
            kind=DocumentKind.STATUTE,
            title="DPDP Act 2023",
            source_ref=ref,
            raw_text="Section 8. Every Data Fiduciary shall...",
        )
        assert changed is True
        assert doc.content_hash == content_hash("Section 8. Every Data Fiduciary shall...")

    async with db.session() as session:
        repo = DocumentRepository(session)
        _, changed = await repo.upsert(
            kind=DocumentKind.STATUTE,
            title="DPDP Act 2023",
            source_ref=ref,
            raw_text="Section 8. Every Data Fiduciary shall...",
        )
        # Unchanged content must not re-trigger the expensive re-embed path.
        assert changed is False


async def test_upsert_detects_revision(db: Database, clean_documents: None) -> None:
    ref = _ref()
    async with db.session() as session:
        repo = DocumentRepository(session)
        await repo.upsert(kind=DocumentKind.STATUTE, title="Act", source_ref=ref, raw_text="v1")

    async with db.session() as session:
        repo = DocumentRepository(session)
        doc, changed = await repo.upsert(
            kind=DocumentKind.STATUTE, title="Act", source_ref=ref, raw_text="v2 amended"
        )
        assert changed is True
        assert doc.content_hash == content_hash("v2 amended")


async def test_chunks_round_trip_with_embedding(db: Database, clean_documents: None) -> None:
    """The vector column must return the same values it was given."""
    async with db.session() as session:
        doc, _ = await repo_doc(session, _ref())
        chunks = await ChunkRepository(session).replace_for_document(
            doc.id,
            [
                {
                    "ordinal": 0,
                    "label": "Section 8(3)",
                    "content": "The Data Fiduciary shall implement safeguards.",
                    "token_count": 7,
                    "embedding": [0.1] * 1024,
                }
            ],
        )
        assert len(chunks) == 1

    async with db.session() as session:
        stored = await ChunkRepository(session).list_for_document(doc.id)
        assert stored[0].label == "Section 8(3)"
        assert stored[0].embedding is not None
        assert len(stored[0].embedding) == 1024
        assert stored[0].embedding[0] == pytest.approx(0.1)


async def test_tsv_trigger_populates_lexical_index(db: Database, clean_documents: None) -> None:
    """Lexical search depends on the trigger, not on application code."""
    async with db.session() as session:
        doc, _ = await repo_doc(session, _ref())
        await ChunkRepository(session).replace_for_document(
            doc.id,
            [{"ordinal": 0, "content": "personal data breach notification duty"}],
        )

    async with db.session() as session:
        hit = await session.execute(
            text(
                "SELECT count(*) FROM chunks "
                "WHERE document_id = :d AND tsv @@ to_tsquery('english', 'breach')"
            ),
            {"d": doc.id},
        )
        assert hit.scalar_one() == 1


async def test_replace_for_document_removes_stale_chunks(
    db: Database, clean_documents: None
) -> None:
    """A shorter revision must not leave the old tail behind to be cited."""
    async with db.session() as session:
        doc, _ = await repo_doc(session, _ref())
        await ChunkRepository(session).replace_for_document(
            doc.id,
            [{"ordinal": i, "content": f"clause {i}"} for i in range(5)],
        )

    async with db.session() as session:
        await ChunkRepository(session).replace_for_document(
            doc.id, [{"ordinal": 0, "content": "only clause"}]
        )

    async with db.session() as session:
        remaining = await ChunkRepository(session).list_for_document(doc.id)
        assert [c.content for c in remaining] == ["only clause"]


async def test_deleting_document_cascades_to_chunks(db: Database, clean_documents: None) -> None:
    async with db.session() as session:
        doc, _ = await repo_doc(session, _ref())
        await ChunkRepository(session).replace_for_document(
            doc.id, [{"ordinal": 0, "content": "text"}]
        )

    async with db.session() as session:
        await DocumentRepository(session).delete(doc.id)

    async with db.session() as session:
        assert await ChunkRepository(session).list_for_document(doc.id) == []


async def test_pending_embedding_finds_unembedded_only(db: Database, clean_documents: None) -> None:
    async with db.session() as session:
        doc, _ = await repo_doc(session, _ref())
        await ChunkRepository(session).replace_for_document(
            doc.id,
            [
                {"ordinal": 0, "content": "embedded", "embedding": [0.2] * 1024},
                {"ordinal": 1, "content": "not yet embedded"},
            ],
        )

    async with db.session() as session:
        repo = ChunkRepository(session)
        pending = [c for c in await repo.pending_embedding(limit=500) if c.document_id == doc.id]
        assert [c.content for c in pending] == ["not yet embedded"]

        filled = await repo.set_embeddings({pending[0].id: [0.3] * 1024})
        assert filled == 1

    async with db.session() as session:
        repo = ChunkRepository(session)
        still = [c for c in await repo.pending_embedding(limit=500) if c.document_id == doc.id]
        assert still == []


async def test_citation_edges_are_idempotent(db: Database, clean_documents: None) -> None:
    """Re-ingesting a judgment must not multiply its citation edges."""
    async with db.session() as session:
        src, _ = await repo_doc(session, _ref(), kind=DocumentKind.JUDGMENT)
        repo = CitationRepository(session)
        first = await repo.add_many(
            src.id, [{"target_ref": "AIR-1973-SC-1461", "context": "relied upon"}]
        )
        assert first == 1

    async with db.session() as session:
        repo = CitationRepository(session)
        second = await repo.add_many(src.id, [{"target_ref": "AIR-1973-SC-1461"}])
        assert second == 0
        assert len(await repo.cites(src.id)) == 1


async def test_resolve_targets_links_late_arriving_documents(
    db: Database, clean_documents: None
) -> None:
    """Citations are parsed before the cited judgment is necessarily ingested."""
    target_ref = _ref()
    async with db.session() as session:
        src, _ = await repo_doc(session, _ref(), kind=DocumentKind.JUDGMENT)
        await CitationRepository(session).add_many(src.id, [{"target_ref": target_ref}])

    async with db.session() as session:
        edges = await CitationRepository(session).cites(src.id)
        assert edges[0].target_document_id is None

    async with db.session() as session:
        await repo_doc(session, target_ref, kind=DocumentKind.JUDGMENT)

    async with db.session() as session:
        repo = CitationRepository(session)
        assert await repo.resolve_targets() >= 1

    async with db.session() as session:
        edges = await CitationRepository(session).cites(src.id)
        assert edges[0].target_document_id is not None


async def test_audit_entries_are_ordered_by_time(db: Database, clean_documents: None) -> None:
    run_id = uuid.uuid4().hex
    async with db.session() as session:
        repo = AuditRepository(session)
        for event in ("retrieve", "analyze", "verify"):
            await repo.record(run_id=run_id, module="compliance", event=event)

    async with db.session() as session:
        entries = await AuditRepository(session).for_run(run_id)
        assert [e.event for e in entries] == ["retrieve", "analyze", "verify"]

    async with db.session() as session:
        await session.execute(text("DELETE FROM audit_log WHERE run_id = :r"), {"r": run_id})


async def test_session_rolls_back_on_error(db: Database, clean_documents: None) -> None:
    """A failed unit of work must leave nothing behind."""
    ref = _ref()
    with pytest.raises(RuntimeError):
        async with db.session() as session:
            await repo_doc(session, ref)
            raise RuntimeError("agent failed mid-run")

    async with db.session() as session:
        found = await DocumentRepository(session).get_by_source_ref(DocumentKind.STATUTE, ref)
        assert found is None


async def test_healthcheck_passes_against_live_db(db: Database) -> None:
    assert await db.healthcheck() is True


async def repo_doc(session, ref: str, kind: DocumentKind = DocumentKind.STATUTE):  # type: ignore[no-untyped-def]
    return await DocumentRepository(session).upsert(
        kind=kind, title=f"Doc {ref}", source_ref=ref, raw_text=f"body of {ref}"
    )
