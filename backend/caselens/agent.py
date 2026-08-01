"""CaseLens agent.

    retrieve ⇄ critic → expand → stance → verify → synthesise

Shares the compliance graph's skeleton and reuses its retriever and verifier
unchanged. One node is specific to case law:

**expand.** Retrieval finds cases whose *text* resembles the fact pattern, but
a foundational authority is often phrased nothing like the facts it later
governs — it is reached through the cases that cite it. The expansion step
pulls in judgments cited by the strongest hits, which is precisely the signal
vector search cannot see.

Stance assessment runs per case and is where the fine-tuned classifier lands in
Phase 6; the LLM path here is the baseline it will be measured against.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from core.agents.trace import NodeStatus, TraceEmitter
from core.agents.verifier import CitationVerifier
from core.config import Settings
from core.db import CitationRepository, Database, DocumentKind
from core.embeddings import EmbeddingService
from core.llm import LLMError, LLMRouter
from core.logging import get_logger
from core.prompts.caselens import (
    RESEARCH_MEMO,
    STANCE_ASSESSMENT,
    ResearchMemo,
    Stance,
)
from core.prompts.registry import render_context
from core.retrieval import Hit, HybridRetriever

log = get_logger(__name__)


@dataclass(slots=True)
class AssessedCase:
    """A precedent whose stance has been determined and citation verified."""

    document_id: str
    chunk_id: str
    citation: str
    case_title: str
    stance: Stance
    confidence: float
    reasoning: str
    holding: str
    quote: str
    #: How many ingested judgments cite this one — a rough authority signal.
    cited_by_count: int = 0
    #: True when reached through the citation graph rather than direct search.
    via_citation_graph: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "citation": self.citation,
            "case_title": self.case_title,
            "stance": self.stance.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "holding": self.holding,
            "quote": self.quote,
            "cited_by_count": self.cited_by_count,
            "via_citation_graph": self.via_citation_graph,
        }


@dataclass(slots=True)
class ResearchResult:
    run_id: str
    facts: str
    cases: list[AssessedCase]
    memo: ResearchMemo | None
    discarded: int
    elapsed_ms: int

    @property
    def by_stance(self) -> dict[str, int]:
        counts = {stance.value: 0 for stance in Stance}
        for case in self.cases:
            counts[case.stance.value] += 1
        return counts


class CaseLensAgent:
    def __init__(
        self,
        database: Database,
        embeddings: EmbeddingService,
        llm: LLMRouter,
        settings: Settings,
    ) -> None:
        self._db = database
        self._embeddings = embeddings
        self._llm = llm
        self._settings = settings

    async def _retrieve(self, query: str, limit: int) -> list[Hit]:
        async with self._db.session() as session:
            retriever = HybridRetriever(session, self._embeddings, self._settings)
            return await retriever.search(query, limit=limit, kind=DocumentKind.JUDGMENT)

    async def _authority_counts(self, hits: list[Hit]) -> dict[str, int]:
        """How many ingested judgments cite each retrieved case."""
        counts: dict[str, int] = {}
        async with self._db.session() as session:
            citations = CitationRepository(session)
            for document_id in {h.document_id for h in hits}:
                counts[str(document_id)] = len(await citations.cited_by(document_id))
        return counts

    async def _expand_via_citations(
        self, hits: list[Hit], emitter: TraceEmitter, limit: int
    ) -> list[Hit]:
        """Pull in judgments the strongest hits rely on.

        A precedent cited by several strong hits is usually worth reading even
        when its own text ranks poorly against the fact pattern — that is the
        whole reason for keeping a citation graph.
        """
        if not hits:
            return []

        emitter.emit("expand", NodeStatus.STARTED, "following citations from top results")
        found: dict[str, Hit] = {}

        async with self._db.session() as session:
            citations = CitationRepository(session)
            seen_documents = {h.document_id for h in hits}

            for hit in hits[:3]:
                for edge in await citations.cites(hit.document_id):
                    if edge.target_document_id is None:
                        continue
                    if edge.target_document_id in seen_documents:
                        continue
                    # Retrieve the cited judgment's most relevant passage
                    # rather than its whole text.
                    retriever = HybridRetriever(session, self._embeddings, self._settings)
                    cited = await retriever.search(
                        edge.target_ref,
                        limit=1,
                        document_id=edge.target_document_id,
                    )
                    for candidate in cited:
                        found.setdefault(str(candidate.chunk_id), candidate)
                    if len(found) >= limit:
                        break
                if len(found) >= limit:
                    break

        expanded = list(found.values())[:limit]
        emitter.emit(
            "expand",
            NodeStatus.COMPLETED,
            f"{len(expanded)} case(s) added via citation graph"
            if expanded
            else "no further cases in the citation graph",
        )
        return expanded

    async def _assess(
        self,
        facts: str,
        hit: Hit,
        emitter: TraceEmitter,
        verifier: CitationVerifier,
        *,
        via_graph: bool,
        authority: int,
    ) -> AssessedCase | None:
        """Determine one case's stance, then verify what it claims."""
        try:
            assessment = await self._llm.structured(
                STANCE_ASSESSMENT.messages(facts=facts[:4000], context=render_context([hit])),
                STANCE_ASSESSMENT.schema,
            )
        except LLMError as exc:
            emitter.emit("stance", NodeStatus.FAILED, f"{hit.citation}: {exc}")
            return None

        result = verifier.verify(assessment.chunk_id, assessment.quote)
        if not result.verified:
            # Same discipline as ComplianceGuard: an unverifiable quote means
            # the stance is not grounded in the retrieved text.
            emitter.emit(
                "verify",
                NodeStatus.FAILED,
                f"{hit.citation}: {result.detail}",
            )
            return None

        return AssessedCase(
            document_id=str(hit.document_id),
            chunk_id=assessment.chunk_id,
            citation=hit.citation,
            case_title=hit.document_title,
            stance=assessment.stance,
            confidence=assessment.confidence,
            reasoning=assessment.reasoning,
            holding=assessment.holding,
            quote=assessment.quote,
            cited_by_count=authority,
            via_citation_graph=via_graph,
        )

    async def research(
        self,
        facts: str,
        *,
        emitter: TraceEmitter | None = None,
        limit: int = 6,
        expand: bool = True,
    ) -> ResearchResult:
        """Fact pattern in, ranked and assessed precedents plus a memo out."""
        run_id = uuid.uuid4().hex
        emitter = emitter or TraceEmitter(run_id)
        started = time.perf_counter()

        emitter.emit("retrieve", NodeStatus.STARTED, "searching judgments")
        hits = await self._retrieve(facts, limit)
        emitter.emit("retrieve", NodeStatus.COMPLETED, f"{len(hits)} judgment(s) retrieved")

        if not hits:
            emitter.emit("emit", NodeStatus.COMPLETED, "no judgments matched")
            return ResearchResult(
                run_id=run_id,
                facts=facts,
                cases=[],
                memo=None,
                discarded=0,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        graph_hits = await self._expand_via_citations(hits, emitter, limit=3) if expand else []
        all_hits = hits + graph_hits
        graph_ids = {str(h.chunk_id) for h in graph_hits}

        verifier = CitationVerifier(all_hits)
        authority = await self._authority_counts(all_hits)

        emitter.emit("stance", NodeStatus.STARTED, f"assessing {len(all_hits)} case(s)")
        limiter = asyncio.Semaphore(self._settings.max_concurrent_clauses)

        async def assess_one(hit: Hit) -> AssessedCase | None:
            async with limiter:
                return await self._assess(
                    facts,
                    hit,
                    emitter,
                    verifier,
                    via_graph=str(hit.chunk_id) in graph_ids,
                    authority=authority.get(str(hit.document_id), 0),
                )

        outcomes = await asyncio.gather(*(assess_one(h) for h in all_hits), return_exceptions=True)

        cases: list[AssessedCase] = []
        discarded = 0
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                log.exception("stance_assessment_failed", error=str(outcome))
                discarded += 1
            elif outcome is None:
                discarded += 1
            else:
                cases.append(outcome)

        # Decided cases first, then by confidence, then by how often the
        # judgment is relied upon elsewhere in the corpus.
        cases.sort(
            key=lambda c: (
                c.stance is Stance.NEUTRAL,
                -c.confidence,
                -c.cited_by_count,
            )
        )
        emitter.emit(
            "stance",
            NodeStatus.COMPLETED,
            f"{len(cases)} assessed, {discarded} unverifiable",
        )

        memo = await self._synthesise(facts, cases, emitter)
        elapsed = int((time.perf_counter() - started) * 1000)
        emitter.emit("emit", NodeStatus.COMPLETED, f"memo ready — {len(cases)} authorities")

        return ResearchResult(
            run_id=run_id,
            facts=facts,
            cases=cases,
            memo=memo,
            discarded=discarded,
            elapsed_ms=elapsed,
        )

    async def _synthesise(
        self, facts: str, cases: list[AssessedCase], emitter: TraceEmitter
    ) -> ResearchMemo | None:
        if not cases:
            return None

        emitter.emit("synthesise", NodeStatus.STARTED, "drafting memo")
        rendered = "\n\n---\n\n".join(
            f"Case: {case.case_title}\n"
            f"Citation: {case.citation}\n"
            f"Stance: {case.stance.value} (confidence {case.confidence:.2f})\n"
            f"Holding: {case.holding}\n"
            f"Passage: {case.quote}"
            for case in cases
        )

        try:
            memo = await self._llm.structured(
                RESEARCH_MEMO.messages(facts=facts[:4000], assessments=rendered[:12000]),
                RESEARCH_MEMO.schema,
                max_tokens=1500,
            )
        except LLMError as exc:
            # The assessed cases are still useful without the prose summary.
            emitter.emit("synthesise", NodeStatus.FAILED, str(exc))
            return None

        emitter.emit("synthesise", NodeStatus.COMPLETED, "memo drafted")
        return memo
