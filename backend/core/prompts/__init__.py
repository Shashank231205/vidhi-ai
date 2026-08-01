"""Versioned prompts.

Prompts are assets, not string literals scattered through agent code. Each is a
module exporting a template plus the Pydantic schema its output must satisfy,
so a prompt change is reviewable, testable, and revertible.

Every prompt in this package enforces the same three rules, because they are
what make Phase 4's verification mechanically checkable:

1. Answer only from the supplied context.
2. Cite the chunk id relied upon for each claim.
3. Say so explicitly when the context is insufficient, rather than guessing.
"""

from core.prompts.compliance import (
    CLAUSE_ANALYSIS,
    RETRIEVAL_CRITIC,
    ClauseAnalysis,
    CriticVerdict,
    Finding,
    RiskLevel,
)
from core.prompts.registry import Prompt, render_context

__all__ = [
    "CLAUSE_ANALYSIS",
    "RETRIEVAL_CRITIC",
    "ClauseAnalysis",
    "CriticVerdict",
    "Finding",
    "Prompt",
    "RiskLevel",
    "render_context",
]
