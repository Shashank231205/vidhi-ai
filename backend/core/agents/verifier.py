"""Citation verification — the groundedness guarantee.

Shared by both modules. A claim survives only if:

1. Its `chunk_id` resolves to a chunk that was actually retrieved for this run.
2. Its `quote` appears in that chunk's text.

Both checks are mechanical, not model-judged. That is the point: a verifier
that asks an LLM whether a citation is accurate inherits the same failure mode
it exists to catch.

Rejected claims are not dropped silently. The graph sends them back to be
re-grounded, and only discards them after the retry budget is spent — a
fabricated citation usually means the model reasoned from the right law and
attributed it to the wrong chunk, which is recoverable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from core.logging import get_logger
from core.retrieval import Hit

log = get_logger(__name__)

#: A quote must match this fraction of its longest run of shared words to pass.
#: Not 1.0: PDF extraction introduces line breaks and ligature artefacts, so
#: byte-exact matching would reject correct citations. Not lower: below this,
#: unrelated legal boilerplate starts to match.
QUOTE_MATCH_THRESHOLD = 0.85

#: Quotes shorter than this cannot be verified meaningfully — "the Act" appears
#: everywhere. Treated as unsupported rather than trusted.
MIN_QUOTE_WORDS = 4


class Rejection(StrEnum):
    UNKNOWN_CHUNK = "unknown_chunk"
    QUOTE_NOT_FOUND = "quote_not_found"
    QUOTE_TOO_SHORT = "quote_too_short"


@dataclass(slots=True)
class VerificationResult:
    verified: bool
    reason: Rejection | None = None
    detail: str = ""


def normalise(text: str) -> str:
    """Collapse the differences PDF extraction introduces.

    Unicode punctuation, non-breaking spaces, and line wrapping all differ
    between a model's quote and the stored chunk without changing meaning.
    """
    text = unicodedata.normalize("NFKD", text)
    # Escapes rather than literals: these characters are indistinguishable
    # from ASCII in most editors, which is exactly why they cause mismatches.
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = re.sub(r"[^\w\s'\"-]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_appears_in(quote: str, source: str) -> tuple[bool, float]:
    """Whether `quote` is present in `source`, tolerating extraction noise.

    Returns `(matched, ratio)`. Exact containment short-circuits; otherwise the
    longest contiguous run of the quote's words found in the source decides.
    Contiguity matters — a scattered bag of words is not a quotation.
    """
    normalised_quote = normalise(quote)
    normalised_source = normalise(source)

    if not normalised_quote:
        return False, 0.0
    if normalised_quote in normalised_source:
        return True, 1.0

    words = normalised_quote.split()
    if len(words) < MIN_QUOTE_WORDS:
        return False, 0.0

    longest = 0
    for start in range(len(words)):
        # Only windows that could beat the current best are worth testing.
        for end in range(len(words), start + longest, -1):
            if " ".join(words[start:end]) in normalised_source:
                longest = end - start
                break

    return (longest / len(words)) >= QUOTE_MATCH_THRESHOLD, longest / len(words)


class CitationVerifier:
    """Checks claims against the chunks that were actually retrieved.

    Scoping to this run's hits rather than the whole corpus is deliberate: a
    model that cites a real provision it was never shown has not grounded its
    reasoning, it has recalled something from training. That is exactly the
    failure this catches.
    """

    def __init__(self, hits: list[Hit]) -> None:
        self._by_id = {str(hit.chunk_id): hit for hit in hits}

    def verify(self, chunk_id: str, quote: str) -> VerificationResult:
        hit = self._by_id.get(chunk_id.strip())
        if hit is None:
            return VerificationResult(
                verified=False,
                reason=Rejection.UNKNOWN_CHUNK,
                detail=f"chunk_id {chunk_id!r} was not retrieved for this run",
            )

        if len(normalise(quote).split()) < MIN_QUOTE_WORDS:
            return VerificationResult(
                verified=False,
                reason=Rejection.QUOTE_TOO_SHORT,
                detail=f"quote must be at least {MIN_QUOTE_WORDS} words",
            )

        matched, ratio = quote_appears_in(quote, hit.content)
        if not matched:
            return VerificationResult(
                verified=False,
                reason=Rejection.QUOTE_NOT_FOUND,
                detail=(f"quote does not appear in {hit.citation} (best match {ratio:.0%})"),
            )

        return VerificationResult(verified=True)

    def citation_for(self, chunk_id: str) -> str | None:
        hit = self._by_id.get(chunk_id.strip())
        return hit.citation if hit else None

    def hit_for(self, chunk_id: str) -> Hit | None:
        return self._by_id.get(chunk_id.strip())
