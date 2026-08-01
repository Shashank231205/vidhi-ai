"""Database schema.

The retrieval-critical fields are typed columns rather than JSON payload keys,
because the hybrid retriever filters on them and ranks on `embedding` + `tsv` —
those have to be indexable.

Vector dimensionality is pinned to the embedding model (BGE-M3, 1024) at the
column level, so swapping models is a migration rather than a silent corruption
of the corpus.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

#: Must equal Settings.embedding_dimensions — asserted by a test, since a
#: mismatch only shows up as a runtime insert failure otherwise.
EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    #: Maps annotated dict columns onto JSONB rather than the default JSON.
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSONB}


class DocumentKind(StrEnum):
    """What a document is — selects the parser and scopes retrieval filters."""

    STATUTE = "statute"
    JUDGMENT = "judgment"
    CONTRACT = "contract"
    POLICY = "policy"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Document(Base, TimestampMixin):
    """A source text: one statute, judgment, contract, or policy."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    kind: Mapped[DocumentKind] = mapped_column(String(16), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)

    #: Stable external identifier — "DPDP-2023", a judgment id, an upload id.
    #: Unique per kind, so re-ingesting updates in place instead of duplicating.
    source_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)

    #: SHA-256 of the raw text; lets ingestion skip unchanged documents entirely.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    #: Kind-specific fields (court, bench, year, act number, jurisdiction).
    meta: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint("kind", "source_ref", name="uq_documents_kind_source_ref"),
        Index("ix_documents_kind", "kind"),
        Index("ix_documents_content_hash", "content_hash"),
    )


class Chunk(Base, TimestampMixin):
    """A retrievable span of a document.

    Chunks are section- or clause-aligned rather than fixed-width, so `label`
    ("Section 8(3)", "Clause 4.2") is a citation the verifier can check against
    rather than a character offset.
    """

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )

    #: Position within the document; defines citation ordering.
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str | None] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: Nullable so ingestion can land text first and embed in a later batch.
    #: Retrieval skips un-embedded rows until they are backfilled.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    #: Maintained by a database trigger, never in Python — that keeps lexical
    #: search correct for rows written outside the application too.
    tsv: Mapped[str | None] = mapped_column(TSVECTOR)

    meta: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        CheckConstraint("token_count >= 0", name="ck_chunks_token_count_non_negative"),
        Index("ix_chunks_document_id", "document_id"),
    )


class Citation(Base, TimestampMixin):
    """A directed edge: `source` cites `target`.

    Powers CaseLens's citation-graph expansion — a judgment cited by several
    strong hits is usually worth retrieving even when its own text ranks poorly.
    `target_ref` is retained when `target_document_id` is null, so citations to
    not-yet-ingested judgments still contribute to the graph.
    """

    __tablename__ = "citations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    target_document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL")
    )
    target_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    context: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("source_document_id", "target_ref", name="uq_citations_source_target"),
        CheckConstraint(
            "source_document_id <> target_document_id",
            name="ck_citations_no_self_reference",
        ),
        Index("ix_citations_source", "source_document_id"),
        Index("ix_citations_target", "target_document_id"),
    )


class AuditLog(Base):
    """Append-only record of agent runs.

    Kept for two reasons beyond debugging: reviewer overrides are the training
    signal for Phase 6, and a stored trace is what makes a past answer
    reproducible when someone challenges it.
    """

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    module: Mapped[str] = mapped_column(String(32), nullable=False)
    event: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_audit_log_run_id", "run_id"),
        Index("ix_audit_log_created_at", "created_at"),
    )
