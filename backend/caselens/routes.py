"""CaseLens endpoints."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from caselens.agent import CaseLensAgent, ResearchResult
from core.agents.trace import RunFinished, TraceEmitter
from core.config import Settings, get_settings
from core.db import Database
from core.embeddings import EmbeddingService
from core.llm import LLMRouter
from core.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/caselens", tags=["caselens"])


def get_database(request: Request) -> Database:
    return request.app.state.db  # type: ignore[no-any-return]


def get_embeddings(request: Request) -> EmbeddingService:
    return request.app.state.embeddings  # type: ignore[no-any-return]


def get_llm(request: Request) -> LLMRouter:
    return request.app.state.llm  # type: ignore[no-any-return]


class ResearchRequest(BaseModel):
    facts: str = Field(
        min_length=20,
        max_length=20_000,
        description="The fact pattern and the position being argued.",
    )
    limit: int = Field(default=6, ge=1, le=20)
    expand: bool = Field(default=True, description="Follow the citation graph from strong hits.")


class CaseResponse(BaseModel):
    document_id: str
    chunk_id: str
    citation: str
    case_title: str
    stance: str
    confidence: float
    reasoning: str
    holding: str
    quote: str
    cited_by_count: int
    via_citation_graph: bool


class MemoResponse(BaseModel):
    summary: str
    supporting_argument: str
    risks: str
    gaps: str | None


class ResearchResponse(BaseModel):
    run_id: str
    cases: list[CaseResponse]
    memo: MemoResponse | None
    stance_summary: dict[str, int]
    #: Assessments whose citation could not be verified. Reported, not hidden.
    discarded: int
    elapsed_ms: int


def _to_response(result: ResearchResult) -> ResearchResponse:
    return ResearchResponse(
        run_id=result.run_id,
        cases=[CaseResponse.model_validate(c.as_dict()) for c in result.cases],
        memo=MemoResponse.model_validate(result.memo.model_dump()) if result.memo else None,
        stance_summary=result.by_stance,
        discarded=result.discarded,
        elapsed_ms=result.elapsed_ms,
    )


@router.post("/research", response_model=ResearchResponse, summary="Research a fact pattern")
async def research(
    request: Request,
    body: ResearchRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ResearchResponse:
    agent = CaseLensAgent(
        get_database(request), get_embeddings(request), get_llm(request), settings
    )
    result = await agent.research(body.facts, limit=body.limit, expand=body.expand)
    return _to_response(result)


@router.post("/research/stream", summary="Research with a live reasoning trace")
async def research_stream(
    request: Request,
    body: ResearchRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> EventSourceResponse:
    run_id = uuid.uuid4().hex
    emitter = TraceEmitter(run_id)
    queue: asyncio.Queue[Any] = asyncio.Queue()
    emitter.subscribe(queue)

    agent = CaseLensAgent(
        get_database(request), get_embeddings(request), get_llm(request), settings
    )

    async def run() -> None:
        try:
            result = await agent.research(
                body.facts, emitter=emitter, limit=body.limit, expand=body.expand
            )
            await queue.put(("result", _to_response(result).model_dump_json()))
            await queue.put(emitter.finished())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("research_failed", run_id=run_id, error=str(exc))
            await queue.put(emitter.finished("The research run failed. Please retry."))

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
            if not task.done():
                task.cancel()

    return EventSourceResponse(events())
