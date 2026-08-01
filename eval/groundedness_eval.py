"""Groundedness evaluation — the PRD's non-negotiable metric.

Two things are measured, and they are different questions:

1. **Adversarial rejection.** Deliberately fabricated citations are fed to the
   verifier. Any that pass are false negatives, and each one is a hallucination
   that would have reached a user. Target: 100% rejected.

2. **Legitimate acceptance.** Real quotes from real chunks, degraded the way
   PDF extraction degrades text — line wrapping, smart quotes, ligatures. Any
   rejected here is a false positive: a correct citation thrown away, which
   makes the tool useless in a different way.

A verifier can trivially score 100% on the first by rejecting everything, so
reporting either number alone would be misleading. Both are reported together.

    uv run python ../eval/groundedness_eval.py
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from core.agents.verifier import CitationVerifier, Rejection  # noqa: E402
from core.retrieval import Hit  # noqa: E402

# Real DPDP Act text, as stored after PDF extraction.
CHUNKS: dict[str, str] = {
    "s5": (
        "5. (1) Every request made to a Data Principal under section 6 for consent "
        "shall be accompanied or preceded by a notice given by the Data Fiduciary "
        "to the Data Principal, informing her the personal data and the purpose for "
        "which the same is proposed to be processed."
    ),
    "s8": (
        "8. (5) Every Data Fiduciary shall protect personal data in its possession "
        "or under its control, including in respect of any processing undertaken by "
        "it or on its behalf by a Data Processor, by taking reasonable security "
        "safeguards to prevent personal data breach."
    ),
    "s9": (
        "9. (1) The Data Fiduciary shall, before processing any personal data of a "
        "child or a person with disability who has a lawful guardian, obtain "
        "verifiable consent of the parent of such child or the lawful guardian."
    ),
    "s33": (
        "33. (1) If the Board determines on conclusion of an inquiry that breach of "
        "the provisions of this Act or the rules made thereunder by a person is "
        "significant, it may, after giving the person an opportunity of being heard, "
        "impose such monetary penalty specified in the Schedule."
    ),
}


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    chunk_key: str
    quote: str
    #: What the verifier must decide. False means it must reject.
    should_pass: bool
    why: str


#: Fabrications a model plausibly produces. Each is designed to defeat a
#: specific weaker check, so passing them all means the check is not shallow.
ADVERSARIAL: list[Case] = [
    Case(
        "invented penalty amount",
        "s33",
        "the Board may impose a monetary penalty of up to two hundred and fifty crore rupees",
        False,
        "figure appears nowhere in the chunk; the Schedule is referenced, not quoted",
    ),
    Case(
        "plausible but absent obligation",
        "s8",
        "the Data Fiduciary shall notify the Board within seventy-two hours of a breach",
        False,
        "72-hour rule is real in GDPR, absent from this provision",
    ),
    Case(
        "right section, wrong chunk cited",
        "s5",
        "obtain verifiable consent of the parent of such child",
        False,
        "quote is genuine but belongs to Section 9, not the chunk cited",
    ),
    Case(
        "words present but scattered",
        "s8",
        "Data Fiduciary breach personal control possession safeguards prevent processing",
        False,
        "every word appears; no contiguous run does — a bag of words is not a quote",
    ),
    Case(
        "quote too short to verify",
        "s8",
        "personal data",
        False,
        "appears throughout the Act and establishes nothing",
    ),
    Case(
        "hallucinated chunk id",
        "nonexistent",
        "shall be accompanied or preceded by a notice given by the Data Fiduciary",
        False,
        "quote is real but the chunk was never retrieved for this run",
    ),
    Case(
        "negated obligation",
        "s9",
        "The Data Fiduciary shall not be required to obtain verifiable consent of the parent",
        False,
        "reverses the provision's meaning while reusing its vocabulary",
    ),
    Case(
        "extended past the source",
        "s5",
        (
            "informing her the personal data and the purpose for which the same is "
            "proposed to be processed and the fees payable for such processing"
        ),
        False,
        "starts verbatim then continues into invented text",
    ),
]

#: Correct citations, degraded the way real extraction degrades them.
LEGITIMATE: list[Case] = [
    Case(
        "verbatim quote",
        "s8",
        "taking reasonable security safeguards to prevent personal data breach",
        True,
        "exact substring",
    ),
    Case(
        "line wrapping from PDF",
        "s5",
        "informing her the personal data\nand the purpose for which\nthe same is proposed",
        True,
        "newlines are an extraction artefact, not a difference in text",
    ),
    Case(
        "smart punctuation",
        "s9",
        "obtain verifiable consent of the parent of such child or the lawful guardian",
        True,
        "typographic variation must not reject a correct citation",
    ),
    Case(
        "case and spacing differences",
        "s33",
        "IMPOSE  SUCH  MONETARY  PENALTY  SPECIFIED  IN  THE  SCHEDULE",
        True,
        "normalisation covers case and repeated whitespace",
    ),
    Case(
        "sentence fragment from the middle",
        "s9",
        "before processing any personal data of a child",
        True,
        "a partial quote is still a quote if contiguous",
    ),
    Case(
        "long multi-clause quote",
        "s8",
        (
            "protect personal data in its possession or under its control, including "
            "in respect of any processing undertaken by it"
        ),
        True,
        "spans a comma and a subordinate clause",
    ),
]


def build_hits() -> tuple[list[Hit], dict[str, str]]:
    """Chunks as the retriever would have returned them."""
    ids = {key: str(uuid.uuid5(uuid.NAMESPACE_DNS, key)) for key in CHUNKS}
    hits = [
        Hit(
            chunk_id=uuid.UUID(ids[key]),
            document_id=uuid.uuid5(uuid.NAMESPACE_DNS, "dpdp"),
            content=text,
            label=f"Section {key[1:]}",
            document_title="Digital Personal Data Protection Act, 2023",
            source_ref="DPDP-2023",
            score=1.0,
        )
        for key, text in CHUNKS.items()
    ]
    return hits, ids


def main() -> int:
    hits, ids = build_hits()
    verifier = CitationVerifier(hits)

    print("\nGroundedness evaluation — citation verifier\n")

    false_negatives: list[Case] = []
    print(f"{'ADVERSARIAL (must be rejected)':<52}{'verdict':>12}")
    print("-" * 66)
    for case in ADVERSARIAL:
        chunk_id = ids.get(case.chunk_key, str(uuid.uuid4()))
        result = verifier.verify(chunk_id, case.quote)
        caught = not result.verified
        if not caught:
            false_negatives.append(case)
        reason = result.reason.value if result.reason else "-"
        print(
            f"  {case.name:<48}{'REJECTED' if caught else 'PASSED ✗':>12}"
            f"   {reason if caught else ''}"
        )

    false_positives: list[Case] = []
    print(f"\n{'LEGITIMATE (must be accepted)':<52}{'verdict':>12}")
    print("-" * 66)
    for case in LEGITIMATE:
        result = verifier.verify(ids[case.chunk_key], case.quote)
        if not result.verified:
            false_positives.append(case)
        print(
            f"  {case.name:<48}{'ACCEPTED' if result.verified else 'REJECTED ✗':>12}"
            f"   {'' if result.verified else result.detail[:40]}"
        )

    rejection_rate = 1 - len(false_negatives) / len(ADVERSARIAL)
    acceptance_rate = 1 - len(false_positives) / len(LEGITIMATE)

    print("\n" + "=" * 66)
    print(f"  Fabrications rejected     {rejection_rate:>7.1%}  ({len(ADVERSARIAL)} cases)")
    print(f"  Real citations accepted   {acceptance_rate:>7.1%}  ({len(LEGITIMATE)} cases)")
    print("=" * 66)

    if false_negatives:
        print("\nFAILED — these fabrications would have reached a user:")
        for case in false_negatives:
            print(f"  · {case.name}: {case.why}")
    if false_positives:
        print("\nFAILED — these correct citations were discarded:")
        for case in false_positives:
            print(f"  · {case.name}: {case.why}")

    if not false_negatives and not false_positives:
        print(
            "\nBoth targets met. Note that a verifier rejecting everything would "
            "also score 100% on the first metric, which is why the second exists."
        )

    # Non-zero exit so CI can gate on this rather than only reporting it.
    return 1 if (false_negatives or false_positives) else 0


if __name__ == "__main__":
    raise SystemExit(main())
