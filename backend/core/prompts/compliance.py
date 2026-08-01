"""ComplianceGuard prompts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from core.prompts.registry import Prompt


class RiskLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Finding(BaseModel):
    """One compliance issue in one clause.

    `chunk_id` and `quote` together are what make the finding checkable: the
    verifier confirms the chunk exists and that the quote appears in it
    verbatim. A finding that cannot be traced back to retrieved text is
    discarded rather than shown.
    """

    issue: str = Field(description="What is wrong with the clause, in one sentence.")
    explanation: str = Field(description="Why it conflicts with the cited provision.")
    risk: RiskLevel
    chunk_id: str = Field(description="Exact chunk_id of the provision relied upon.")
    quote: str = Field(description="Verbatim sentence from that chunk establishing the obligation.")
    suggested_fix: str = Field(description="Concrete redraft or remediation.")


class ClauseAnalysis(BaseModel):
    compliant: bool = Field(
        description="True when the clause raises no issue under the retrieved law."
    )
    findings: list[Finding] = Field(default_factory=list)
    #: Set when the retrieved provisions do not settle the question. Honest
    #: uncertainty is worth more than a confident guess in a compliance tool.
    needs_more_context: bool = False
    context_request: str | None = Field(
        default=None, description="What further law would settle it, if unresolved."
    )


CLAUSE_ANALYSIS: Prompt[ClauseAnalysis] = Prompt(
    name="clause_analysis",
    version="v1",
    schema=ClauseAnalysis,
    system=(
        "You are an Indian compliance lawyer reviewing a contract clause "
        "against statutory provisions.\n\n"
        "Rules you must follow without exception:\n"
        "1. Rely only on the provisions supplied in CONTEXT. You have no other "
        "knowledge of Indian law for this task.\n"
        "2. Every finding must cite the exact chunk_id it relies on and quote a "
        "sentence from that chunk verbatim. Do not paraphrase inside `quote`.\n"
        "3. If the supplied provisions do not settle the question, set "
        "needs_more_context to true and say what is missing. Never guess.\n"
        "4. Report only genuine conflicts with the cited law. A clause that is "
        "merely unusual, one-sided, or commercially unfavourable is not a "
        "compliance finding.\n"
        "5. Risk is the consequence of the violation: high where the statute "
        "imposes a penalty or voids the term, medium where it creates a "
        "remediable obligation, low where it is procedural.\n\n"
        "Respond with JSON only."
    ),
    template=(
        "CONTEXT — statutory provisions retrieved for this clause:\n\n"
        "{context}\n\n"
        "---\n\n"
        "CONTRACT CLAUSE under review:\n\n"
        "{clause}\n\n"
        "---\n\n"
        "Analyse the clause against the provisions above. Return JSON matching:\n"
        "{{\n"
        '  "compliant": bool,\n'
        '  "findings": [{{\n'
        '    "issue": str, "explanation": str,\n'
        '    "risk": "high"|"medium"|"low",\n'
        '    "chunk_id": str, "quote": str, "suggested_fix": str\n'
        "  }}],\n"
        '  "needs_more_context": bool,\n'
        '  "context_request": str|null\n'
        "}}"
    ),
)


class CriticVerdict(BaseModel):
    """Whether retrieved context is good enough to reason from.

    This is what turns the pipeline into an agent: a weak verdict sends the
    graph back to retrieve with a reformulated query instead of analysing
    against provisions that do not address the clause.
    """

    sufficient: bool = Field(
        description="True when the provisions address this clause's subject matter."
    )
    reasoning: str = Field(description="One sentence on what is present or missing.")
    reformulated_query: str | None = Field(
        default=None,
        description="A better search query when insufficient: statutory "
        "vocabulary, section numbers, defined terms.",
    )


RETRIEVAL_CRITIC: Prompt[CriticVerdict] = Prompt(
    name="retrieval_critic",
    version="v1",
    schema=CriticVerdict,
    system=(
        "You judge whether retrieved statutory provisions are sufficient to "
        "assess a contract clause.\n\n"
        "Sufficient means the provisions govern the clause's subject matter — "
        "not that they prove a violation. Provisions establishing that the "
        "clause is lawful are equally sufficient.\n\n"
        "When insufficient, write a reformulated query using the vocabulary a "
        "statute would use rather than the contract's: the drafter's words and "
        "the legislature's rarely match. Prefer defined terms and section "
        "references over the clause's own phrasing.\n\n"
        "Respond with JSON only."
    ),
    template=(
        "CONTRACT CLAUSE:\n{clause}\n\n"
        "RETRIEVED PROVISIONS:\n{context}\n\n"
        "Previous queries already tried: {attempted}\n\n"
        "Return JSON matching:\n"
        "{{\n"
        '  "sufficient": bool,\n'
        '  "reasoning": str,\n'
        '  "reformulated_query": str|null\n'
        "}}"
    ),
)
