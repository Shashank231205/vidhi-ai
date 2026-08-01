"""ComplianceGuard agent.

A graph, not a chain: the path taken depends on runtime state.

    parse → retrieve ⇄ critic → analyze ⇄ verify → emit
                 ▲                 │
                 └── more context ─┘

Three edges make it self-correcting, and each is bounded so a bad clause
cannot spin:

- **retrieve → critic → retrieve.** The critic judges whether the retrieved
  provisions actually govern the clause. If not, it reformulates the query in
  statutory vocabulary and retries. Bounded by `max_retrieval_attempts`.
- **analyze → retrieve.** Mid-reasoning, the analyzer can ask for a specific
  missing provision rather than guessing at it.
- **verify → analyze.** Ungrounded findings go back to be re-grounded, and are
  discarded only after `max_grounding_attempts`.

Implemented directly rather than through LangGraph's runtime: the graph is
small, the control flow is the interesting part, and expressing it as plain
async code keeps it readable and trivially testable. The state machine is the
same one the PRD describes.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from core.agents.trace import NodeStatus, TraceEmitter
from core.agents.verifier import CitationVerifier
from core.config import Settings
from core.db import Database, DocumentKind
from core.embeddings import EmbeddingService
from core.ingestion.chunker import Chunk, chunk_legal_text
from core.llm import LLMError, LLMRouter
from core.logging import get_logger
from core.prompts import CLAUSE_ANALYSIS, RETRIEVAL_CRITIC, render_context
from core.prompts.compliance import Finding, RiskLevel
from core.retrieval import Hit, HybridRetriever

log = get_logger(__name__)


@dataclass(slots=True)
class VerifiedFinding:
    """A finding that survived citation verification."""

    clause_label: str | None
    clause_text: str
    issue: str
    explanation: str
    risk: RiskLevel
    citation: str
    quote: str
    suggested_fix: str
    chunk_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "clause_label": self.clause_label,
            "clause_text": self.clause_text,
            "issue": self.issue,
            "explanation": self.explanation,
            "risk": self.risk.value,
            "citation": self.citation,
            "quote": self.quote,
            "suggested_fix": self.suggested_fix,
            "chunk_id": self.chunk_id,
        }


@dataclass(slots=True)
class ClauseResult:
    label: str | None
    text: str
    findings: list[VerifiedFinding] = field(default_factory=list)
    #: Findings the verifier rejected after retries — surfaced in the trace so
    #: a silent drop is never invisible.
    discarded: int = 0
    retrieval_attempts: int = 1
    unresolved: bool = False


@dataclass(slots=True)
class AuditResult:
    run_id: str
    document_title: str
    clauses_reviewed: int
    findings: list[VerifiedFinding]
    discarded_findings: int
    elapsed_ms: int

    @property
    def by_risk(self) -> dict[str, int]:
        counts = {level.value: 0 for level in RiskLevel}
        for finding in self.findings:
            counts[finding.risk.value] += 1
        return counts


class ComplianceAgent:
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

    async def _retrieve(
        self, query: str, limit: int, kind: DocumentKind | None = DocumentKind.STATUTE
    ) -> list[Hit]:
        async with self._db.session() as session:
            retriever = HybridRetriever(session, self._embeddings, self._settings)
            return await retriever.search(query, limit=limit, kind=kind)

    async def _retrieve_with_critic(
        self, clause_text: str, emitter: TraceEmitter, clause_label: str
    ) -> tuple[list[Hit], int]:
        """Retrieve, then let a critic decide whether to try a better query."""
        query = clause_text
        attempted: list[str] = []
        hits: list[Hit] = []

        for attempt in range(1, self._settings.max_retrieval_attempts + 1):
            emitter.emit(
                "retrieve",
                NodeStatus.STARTED,
                f"searching statutes for {clause_label}",
                attempt=attempt,
            )
            hits = await self._retrieve(query, self._settings.retrieval_top_k)
            attempted.append(query)
            emitter.emit(
                "retrieve",
                NodeStatus.COMPLETED,
                f"retrieved {len(hits)} provisions",
                attempt=attempt,
                citations=[h.citation for h in hits[:5]],
            )

            # The last attempt has nowhere to go, so skip the critic's cost.
            if attempt == self._settings.max_retrieval_attempts:
                return hits, attempt

            emitter.emit("critic", NodeStatus.STARTED, "assessing retrieved law", attempt=attempt)
            try:
                verdict = await self._llm.structured(
                    RETRIEVAL_CRITIC.messages(
                        clause=clause_text[:3000],
                        context=render_context(hits),
                        attempted="; ".join(attempted),
                    ),
                    RETRIEVAL_CRITIC.schema,
                )
            except LLMError as exc:
                # A critic failure must not fail the audit: proceed with what
                # was retrieved rather than losing the clause entirely.
                emitter.emit(
                    "critic", NodeStatus.FAILED, f"critic unavailable: {exc}", attempt=attempt
                )
                return hits, attempt

            if verdict.sufficient or not verdict.reformulated_query:
                emitter.emit(
                    "critic",
                    NodeStatus.COMPLETED,
                    verdict.reasoning,
                    attempt=attempt,
                    sufficient=True,
                )
                return hits, attempt

            emitter.emit(
                "critic",
                NodeStatus.RETRYING,
                f"context weak — {verdict.reasoning}",
                attempt=attempt,
                next_query=verdict.reformulated_query,
            )
            query = verdict.reformulated_query

        return hits, self._settings.max_retrieval_attempts

    async def _analyze_and_verify(
        self,
        clause_text: str,
        clause_label: str,
        hits: list[Hit],
        emitter: TraceEmitter,
    ) -> tuple[list[VerifiedFinding], int, bool]:
        """Analyse the clause, then verify every claim it makes.

        Rejected findings are fed back with the rejection reason rather than
        dropped: the model usually cited the right law under the wrong id.
        """
        verifier = CitationVerifier(hits)
        context = render_context(hits)
        feedback: str | None = None
        discarded = 0

        for attempt in range(1, self._settings.max_grounding_attempts + 1):
            emitter.emit(
                "analyze",
                NodeStatus.STARTED,
                f"analysing {clause_label} against retrieved law",
                attempt=attempt,
            )

            messages = CLAUSE_ANALYSIS.messages(context=context, clause=clause_text[:6000])
            if feedback:
                messages.append({"role": "user", "content": feedback})

            try:
                analysis = await self._llm.structured(messages, CLAUSE_ANALYSIS.schema)
            except LLMError as exc:
                emitter.emit("analyze", NodeStatus.FAILED, str(exc), attempt=attempt)
                return [], discarded, True

            if analysis.compliant and not analysis.findings:
                emitter.emit(
                    "analyze",
                    NodeStatus.COMPLETED,
                    "no compliance issue found",
                    attempt=attempt,
                )
                return [], discarded, False

            emitter.emit(
                "analyze",
                NodeStatus.COMPLETED,
                f"{len(analysis.findings)} candidate finding(s)",
                attempt=attempt,
            )

            emitter.emit(
                "verify",
                NodeStatus.STARTED,
                f"verifying {len(analysis.findings)} citation(s)",
                attempt=attempt,
            )
            verified, rejected = self._verify_findings(
                analysis.findings, clause_label, clause_text, verifier
            )

            if not rejected:
                emitter.emit(
                    "verify",
                    NodeStatus.COMPLETED,
                    f"{len(verified)} finding(s) grounded",
                    attempt=attempt,
                )
                return verified, discarded, False

            if attempt == self._settings.max_grounding_attempts:
                discarded += len(rejected)
                emitter.emit(
                    "verify",
                    NodeStatus.COMPLETED,
                    f"{len(verified)} grounded, {len(rejected)} discarded as unverifiable",
                    attempt=attempt,
                    discarded=[r for _, r in rejected],
                )
                return verified, discarded, False

            emitter.emit(
                "verify",
                NodeStatus.RETRYING,
                f"{len(rejected)} citation(s) failed verification — re-grounding",
                attempt=attempt,
                reasons=[r for _, r in rejected],
            )
            feedback = (
                "These findings were rejected because their citations could not "
                "be verified against the provided context:\n"
                + "\n".join(f"- {issue}: {reason}" for issue, reason in rejected)
                + "\n\nRe-state only findings you can support with an exact "
                "chunk_id from CONTEXT and a verbatim quote from that chunk. "
                "Drop any finding you cannot ground."
            )

        return [], discarded, False

    def _verify_findings(
        self,
        findings: list[Finding],
        clause_label: str,
        clause_text: str,
        verifier: CitationVerifier,
    ) -> tuple[list[VerifiedFinding], list[tuple[str, str]]]:
        verified: list[VerifiedFinding] = []
        rejected: list[tuple[str, str]] = []

        for finding in findings:
            result = verifier.verify(finding.chunk_id, finding.quote)
            if not result.verified:
                rejected.append((finding.issue, result.detail))
                log.info(
                    "citation_rejected",
                    reason=result.reason.value if result.reason else "unknown",
                    detail=result.detail,
                )
                continue

            citation = verifier.citation_for(finding.chunk_id) or "unknown source"
            verified.append(
                VerifiedFinding(
                    clause_label=clause_label,
                    clause_text=clause_text,
                    issue=finding.issue,
                    explanation=finding.explanation,
                    risk=finding.risk,
                    citation=citation,
                    quote=finding.quote,
                    suggested_fix=finding.suggested_fix,
                    chunk_id=finding.chunk_id,
                )
            )
        return verified, rejected

    async def audit(
        self,
        contract_text: str,
        *,
        title: str = "Uploaded contract",
        emitter: TraceEmitter | None = None,
        max_clauses: int | None = None,
    ) -> AuditResult:
        """Audit a contract clause by clause."""
        run_id = uuid.uuid4().hex
        emitter = emitter or TraceEmitter(run_id)
        started = time.perf_counter()

        emitter.emit("parse", NodeStatus.STARTED, "splitting contract into clauses")
        clauses = chunk_legal_text(contract_text)
        if max_clauses:
            clauses = clauses[:max_clauses]
        emitter.emit("parse", NodeStatus.COMPLETED, f"{len(clauses)} clauses identified")

        # Clauses are independent, so they run concurrently. Sequentially, a
        # clause costs up to 3 critic plus 2 analyse round trips, and a real
        # contract has dozens — measured at 161s for two clauses before this.
        # The semaphore keeps the fan-out inside provider rate limits.
        limiter = asyncio.Semaphore(self._settings.max_concurrent_clauses)
        completed = 0

        async def review(index: int, clause: Chunk) -> tuple[list[VerifiedFinding], int]:
            nonlocal completed
            label = clause.label or f"Clause {index}"

            async with limiter:
                hits, attempts = await self._retrieve_with_critic(clause.content, emitter, label)
                if not hits:
                    emitter.emit(
                        "analyze",
                        NodeStatus.SKIPPED,
                        f"{label}: no relevant law retrieved",
                    )
                    return [], 0

                findings, discarded, failed = await self._analyze_and_verify(
                    clause.content, label, hits, emitter
                )

            completed += 1
            if not failed:
                emitter.emit(
                    "clause",
                    NodeStatus.COMPLETED,
                    f"{label}: {len(findings)} issue(s)",
                    progress=f"{completed}/{len(clauses)}",
                    retrieval_attempts=attempts,
                )
            return findings, discarded

        reviewed = await asyncio.gather(
            *(review(i, c) for i, c in enumerate(clauses, start=1)),
            return_exceptions=True,
        )

        all_findings: list[VerifiedFinding] = []
        discarded_total = 0
        for outcome in reviewed:
            if isinstance(outcome, BaseException):
                # One clause failing must not lose the rest of the audit.
                log.exception("clause_review_failed", error=str(outcome))
                emitter.emit("clause", NodeStatus.FAILED, str(outcome)[:200])
                continue
            findings, discarded = outcome
            all_findings.extend(findings)
            discarded_total += discarded

        elapsed = int((time.perf_counter() - started) * 1000)
        emitter.emit(
            "emit",
            NodeStatus.COMPLETED,
            f"audit complete — {len(all_findings)} verified finding(s)",
            discarded=discarded_total,
        )

        return AuditResult(
            run_id=run_id,
            document_title=title,
            clauses_reviewed=len(clauses),
            findings=all_findings,
            discarded_findings=discarded_total,
            elapsed_ms=elapsed,
        )
