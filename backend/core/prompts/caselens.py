"""CaseLens prompts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from core.prompts.registry import Prompt


class Stance(StrEnum):
    """How a precedent bears on the user's position."""

    SUPPORTS = "supports"
    UNDERMINES = "undermines"
    NEUTRAL = "neutral"


class StanceAssessment(BaseModel):
    """Whether a retrieved case helps or hurts.

    This is the judgement a keyword search cannot make and the reason CaseLens
    exists: knowing a case is *relevant* says nothing about which side it
    favours.
    """

    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Why the case cuts the way it does.")
    chunk_id: str = Field(description="Exact chunk_id of the passage relied upon.")
    quote: str = Field(description="Verbatim passage from that chunk.")
    #: The proposition the case establishes, in the user's terms.
    holding: str = Field(description="The principle this passage establishes.")


STANCE_ASSESSMENT: Prompt[StanceAssessment] = Prompt(
    name="stance_assessment",
    version="v1",
    schema=StanceAssessment,
    system=(
        "You assess whether an Indian precedent supports or undermines a "
        "litigant's position.\n\n"
        "Rules:\n"
        "1. Judge only from the passage supplied. You have no other knowledge "
        "of this case.\n"
        "2. 'supports' means the reasoning, applied to these facts, favours the "
        "user's position. 'undermines' means it favours the opposing side. "
        "'neutral' means it is relevant background but decides nothing either "
        "way.\n"
        "3. A case that is distinguishable on its facts undermines nothing — "
        "say so and mark it neutral.\n"
        "4. Confidence reflects how squarely the passage addresses these facts, "
        "not how strongly worded it is.\n"
        "5. Quote verbatim from the passage and cite its exact chunk_id.\n\n"
        "Respond with JSON only."
    ),
    template=(
        "USER'S FACT PATTERN AND POSITION:\n{facts}\n\n"
        "---\n\n"
        "CANDIDATE PRECEDENT:\n{context}\n\n"
        "---\n\n"
        "Return JSON matching:\n"
        "{{\n"
        '  "stance": "supports"|"undermines"|"neutral",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "reasoning": str,\n'
        '  "chunk_id": str,\n'
        '  "quote": str,\n'
        '  "holding": str\n'
        "}}"
    ),
)


class ResearchMemo(BaseModel):
    """The synthesised answer over all assessed precedents."""

    summary: str = Field(description="Two to four sentences on the position's strength.")
    supporting_argument: str = Field(
        description="The best argument available on these authorities."
    )
    risks: str = Field(description="What the opposing side will rely on.")
    #: Explicit when the retrieved authorities do not settle the question.
    gaps: str | None = Field(
        default=None, description="What the retrieved authorities do not address."
    )


RESEARCH_MEMO: Prompt[ResearchMemo] = Prompt(
    name="research_memo",
    version="v1",
    schema=ResearchMemo,
    system=(
        "You write a research memo for an Indian litigator from precedents that "
        "have already been assessed for stance.\n\n"
        "Rules:\n"
        "1. Use only the assessed precedents supplied. Do not introduce cases "
        "or principles from memory.\n"
        "2. Address the adverse authorities directly. A memo that omits what "
        "undermines the position is worse than useless — it will be relied on.\n"
        "3. Where the authorities do not settle the question, say so in `gaps` "
        "rather than papering over it.\n"
        "4. Refer to cases by the citation given. Do not invent citations.\n\n"
        "Respond with JSON only."
    ),
    template=(
        "FACT PATTERN AND POSITION:\n{facts}\n\n"
        "---\n\n"
        "ASSESSED PRECEDENTS:\n{assessments}\n\n"
        "---\n\n"
        "Return JSON matching:\n"
        "{{\n"
        '  "summary": str,\n'
        '  "supporting_argument": str,\n'
        '  "risks": str,\n'
        '  "gaps": str|null\n'
        "}}"
    ),
)
