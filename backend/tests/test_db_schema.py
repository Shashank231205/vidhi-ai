"""Schema invariants that need no database connection.

Read through Base.metadata rather than the ORM classes: metadata.tables is
typed as Table, so the constraint and foreign-key attributes are visible to the
type checker as well as at runtime.
"""

from sqlalchemy import Table

from core.db import EMBEDDING_DIM, Base
from tests.conftest import build_settings


def table(name: str) -> Table:
    return Base.metadata.tables[name]


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
    fk = next(iter(table("chunks").c.document_id.foreign_keys))
    assert fk.ondelete == "CASCADE"


def test_citation_target_survives_deleted_document() -> None:
    """The edge must remain as a dangling ref, not vanish with its target."""
    fk = next(iter(table("citations").c.target_document_id.foreign_keys))
    assert fk.ondelete == "SET NULL"


def test_document_identity_is_unique_per_kind() -> None:
    """Re-ingesting must update in place rather than duplicate."""
    names = {c.name for c in table("documents").constraints if c.name}
    assert "uq_documents_kind_source_ref" in names


def test_embedding_column_is_nullable() -> None:
    """Ingestion lands text first and embeds in a later batch."""
    assert table("chunks").c.embedding.nullable is True


def test_chunk_ordinal_is_unique_per_document() -> None:
    """Ordinal defines citation ordering, so duplicates would be ambiguous."""
    names = {c.name for c in table("chunks").constraints if c.name}
    assert "uq_chunks_document_ordinal" in names
