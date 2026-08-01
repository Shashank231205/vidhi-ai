"""ComplianceGuard endpoints.

Audits stream. A full contract audit takes tens of seconds — several LLM calls
per clause — and a request that returns nothing for that long is
indistinguishable from a hang. Streaming the trace makes the wait legible and
makes the agent's self-correction visible as it happens.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from compliance.agent import AuditResult, ComplianceAgent
from core.agents.trace import RunFinished, TraceEmitter
from core.config import Settings, get_settings
from core.db import Database
from core.embeddings import EmbeddingService
from core.ingestion.upload import MAX_UPLOAD_BYTES, UploadError, extract_upload
from core.llm import LLMRouter
from core.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/compliance", tags=["compliance"])


def get_database(request: Request) -> Database:
    return request.app.state.db  # type: ignore[no-any-return]


def get_embeddings(request: Request) -> EmbeddingService:
    return request.app.state.embeddings  # type: ignore[no-any-return]


def get_llm(request: Request) -> LLMRouter:
    return request.app.state.llm  # type: ignore[no-any-return]


class AuditRequest(BaseModel):
    text: str = Field(min_length=20, max_length=200_000)
    title: str = Field(default="Pasted contract", max_length=200)
    #: Caps a demo run. Auditing 200 clauses costs 400+ LLM calls.
    max_clauses: int | None = Field(default=None, ge=1, le=100)


class FindingResponse(BaseModel):
    clause_label: str | None
    clause_text: str
    issue: str
    explanation: str
    risk: str
    citation: str
    quote: str
    suggested_fix: str
    chunk_id: str


class AuditResponse(BaseModel):
    run_id: str
    document_title: str
    clauses_reviewed: int
    findings: list[FindingResponse]
    #: Findings the verifier could not ground. Reported, never hidden.
    discarded_findings: int
    risk_summary: dict[str, int]
    elapsed_ms: int


def _to_response(result: AuditResult) -> AuditResponse:
    return AuditResponse(
        run_id=result.run_id,
        document_title=result.document_title,
        clauses_reviewed=result.clauses_reviewed,
        findings=[FindingResponse.model_validate(f.as_dict()) for f in result.findings],
        discarded_findings=result.discarded_findings,
        risk_summary=result.by_risk,
        elapsed_ms=result.elapsed_ms,
    )


@router.post("/audit", response_model=AuditResponse, summary="Audit contract text")
async def audit_contract(
    request: Request,
    body: AuditRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AuditResponse:
    """Run a full audit and return the result once complete."""
    agent = ComplianceAgent(
        get_database(request), get_embeddings(request), get_llm(request), settings
    )
    result = await agent.audit(body.text, title=body.title, max_clauses=body.max_clauses)
    return _to_response(result)


@router.post("/audit/upload", response_model=AuditResponse, summary="Audit a PDF")
async def audit_upload(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File()],
    max_clauses: Annotated[int | None, Form()] = None,
) -> AuditResponse:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
        )

    try:
        extracted = extract_upload(content, file.filename or "contract.pdf")
    except UploadError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    agent = ComplianceAgent(
        get_database(request), get_embeddings(request), get_llm(request), settings
    )
    result = await agent.audit(extracted.text, title=extracted.title, max_clauses=max_clauses)
    return _to_response(result)


@router.post("/audit/stream", summary="Audit with a live reasoning trace")
async def audit_stream(
    request: Request,
    body: AuditRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> EventSourceResponse:
    """Stream trace events, then the result.

    The audit runs as a task writing into a queue the response drains, so a
    slow client never blocks the agent and a disconnect cancels the work
    instead of leaving it running.
    """
    run_id = uuid.uuid4().hex
    emitter = TraceEmitter(run_id)
    queue: asyncio.Queue[Any] = asyncio.Queue()
    emitter.subscribe(queue)

    agent = ComplianceAgent(
        get_database(request), get_embeddings(request), get_llm(request), settings
    )

    async def run() -> None:
        try:
            result = await agent.audit(
                body.text,
                title=body.title,
                emitter=emitter,
                max_clauses=body.max_clauses,
            )
            await queue.put(("result", _to_response(result).model_dump_json()))
            await queue.put(emitter.finished())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("audit_failed", run_id=run_id, error=str(exc))
            # The message is generic on purpose; details are in the logs.
            await queue.put(emitter.finished("The audit failed. Please retry."))

    task = asyncio.create_task(run())

    async def events() -> Any:
        try:
            while True:
                item = await queue.get()
                if isinstance(item, RunFinished):
                    yield item.sse()
                    return
                if isinstance(item, tuple):
                    yield {"event": item[0], "data": item[1]}
                    continue
                yield item.sse()
        finally:
            # Client disconnected or stream ended: stop the audit.
            if not task.done():
                task.cancel()

    return EventSourceResponse(events())
