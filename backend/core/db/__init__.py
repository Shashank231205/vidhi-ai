"""Data layer: schema, session lifecycle, and typed repositories."""

from core.db.models import (
    EMBEDDING_DIM,
    AuditLog,
    Base,
    Chunk,
    Citation,
    Document,
    DocumentKind,
)
from core.db.repositories import (
    AuditRepository,
    ChunkRepository,
    CitationRepository,
    DocumentRepository,
    content_hash,
)
from core.db.session import Database, create_engine

__all__ = [
    "EMBEDDING_DIM",
    "AuditLog",
    "AuditRepository",
    "Base",
    "Chunk",
    "ChunkRepository",
    "Citation",
    "CitationRepository",
    "Database",
    "Document",
    "DocumentKind",
    "DocumentRepository",
    "content_hash",
    "create_engine",
]
