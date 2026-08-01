"""Schema invariants that need no database connection."""

from sqlalchemy import inspect

from core.db import EMBEDDING_DIM, Base, Chunk, Citation, Document
from tests.conftest import build_settings


def test_embedding_dim_matches_settings() -> None:
    """A mismatch here only surfaces as an insert failure at ingestion time."""
    assert build_settings().embedding_dimensions == EMBEDDING_DIM


def test_all_tables_are_registered() -> None:
    assert set(Base.metadata.tables) == {
        "documents",
        "chunks",
        "citations",
        "audit_log",
    }


def test_chunks_cascade_on_document_delete() -> None:
    """Orphaned chunks would be citable text with no source document."""
    fk = next(iter(inspect(Chunk).persist_selectable.c.document_id.foreign_keys))
    assert fk.ondelete == "CASCADE"


def test_citation_target_survives_deleted_document() -> None:
    """The edge must remain as a dangling ref, not vanish with its target."""
    fk = next(iter(inspect(Citation).persist_selectable.c.target_document_id.foreign_keys))
    assert fk.ondelete == "SET NULL"


def test_document_identity_is_unique_per_kind() -> None:
    """Re-ingesting must update in place rather than duplicate."""
    constraints = {c.name for c in inspect(Document).persist_selectable.constraints if c.name}
    assert "uq_documents_kind_source_ref" in constraints


def test_embedding_column_is_nullable() -> None:
    """Ingestion lands text first and embeds in a later batch."""
    assert inspect(Chunk).persist_selectable.c.embedding.nullable is True
