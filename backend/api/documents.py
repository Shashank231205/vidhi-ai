"""Document endpoints: upload, list, search.

Everything the corpus exposes over HTTP lives here — the API layer's own
liveness routes stay in routes.py, and the domain modules bring their own.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel

from core.config import Settings, get_settings
from core.db import ChunkRepository, Database, DocumentKind, DocumentRepository
from core.embeddings import EmbeddingService
from core.ingestion.pipeline import IngestionPipeline
from core.ingestion.upload import MAX_UPLOAD_BYTES, UploadError, extract_upload
from core.logging import get_logger
from core.retrieval import HybridRetriever

log = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def get_database(request: Request) -> Database:
    return request.app.state.db  # type: ignore[no-any-return]


def get_embeddings(request: Request) -> EmbeddingService:
    return request.app.state.embeddings  # type: ignore[no-any-return]


class DocumentSummary(BaseModel):
    id: uuid.UUID
    kind: DocumentKind
    title: str
    source_ref: str
    source_url: str | None
    chunk_count: int
    meta: dict[str, Any]


class UploadResponse(BaseModel):
    document_id: uuid.UUID
    title: str
    pages: int
    characters: int
    chunks: int
    embedded: int
    #: True when this exact file was already ingested; nothing was recomputed.
    already_present: bool


class SearchHit(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    label: str | None
    citation: str
    content: str
    score: float
    matched_by: list[str]


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    took_ms: int


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a contract or policy PDF",
)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF, up to 20MB")],
    kind: Annotated[DocumentKind, Query()] = DocumentKind.CONTRACT,
) -> UploadResponse:
    """Ingest a user-supplied PDF: extract, chunk, embed, index.

    Identical files are content-addressed, so re-uploading one returns the
    existing document instead of duplicating it or re-embedding.
    """
    # Read with a hard cap: never pull an unbounded body into memory.
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
        )

    try:
        extracted = extract_upload(content, file.filename or "document.pdf", kind=kind)
    except UploadError as exc:
        # UploadError messages are written to be shown to the user.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    database = get_database(request)
    pipeline = IngestionPipeline(database, get_embeddings(request))

    result = await pipeline.ingest(
        kind=extracted.kind,
        title=extracted.title,
        source_ref=extracted.source_ref,
        raw_text=extracted.text,
        meta={"pages": extracted.pages, "uploaded": True},
    )

    log.info(
        "document_uploaded",
        document_id=result.document_id,
        pages=extracted.pages,
        chunks=result.chunks_written,
        skipped=result.skipped,
    )

    return UploadResponse(
        document_id=uuid.UUID(result.document_id),
        title=extracted.title,
        pages=extracted.pages,
        characters=extracted.characters,
        chunks=result.chunks_written,
        embedded=result.chunks_embedded,
        already_present=result.skipped,
    )


@router.get("", response_model=list[DocumentSummary], summary="List ingested documents")
async def list_documents(
    request: Request,
    kind: Annotated[DocumentKind | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentSummary]:
    database = get_database(request)

    async with database.session() as session:
        # One query for the page of documents, one for their chunk counts.
        # Per-kind loops and per-document chunk loads made this take 20s once
        # the corpus held a few thousand chunks.
        found = await DocumentRepository(session).list_all(kind=kind, limit=limit, offset=offset)
        counts = await ChunkRepository(session).count_by_document([d.id for d in found])

        return [
            DocumentSummary(
                id=document.id,
                kind=document.kind,
                title=document.title,
                source_ref=document.source_ref,
                source_url=document.source_url,
                chunk_count=counts.get(document.id, 0),
                meta=document.meta,
            )
            for document in found
        ]


@router.get("/search", response_model=SearchResponse, summary="Search the corpus")
async def search_documents(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    q: Annotated[str, Query(min_length=2, max_length=1000, description="Query text")],
    limit: Annotated[int, Query(ge=1, le=50)] = 8,
    kind: Annotated[DocumentKind | None, Query()] = None,
    document_id: Annotated[uuid.UUID | None, Query()] = None,
) -> SearchResponse:
    """Hybrid retrieval over the ingested corpus."""
    import time

    started = time.perf_counter()
    database = get_database(request)

    async with database.session() as session:
        retriever = HybridRetriever(session, get_embeddings(request), settings)
        hits = await retriever.search(q, limit=limit, kind=kind, document_id=document_id)

    return SearchResponse(
        query=q,
        hits=[
            SearchHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                document_title=hit.document_title,
                label=hit.label,
                citation=hit.citation,
                content=hit.content,
                score=hit.score,
                matched_by=list(hit.matched_by),
            )
            for hit in hits
        ],
        took_ms=int((time.perf_counter() - started) * 1000),
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
)
async def delete_document(request: Request, document_id: uuid.UUID) -> None:
    database = get_database(request)
    async with database.session() as session:
        if not await DocumentRepository(session).delete(document_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found.")
