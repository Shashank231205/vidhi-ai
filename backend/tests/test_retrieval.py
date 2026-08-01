"""Fusion logic. Ranking behaviour is testable without a database."""

import uuid

from core.retrieval import Hit, reciprocal_rank_fusion


def hit(name: str, *, matched_by: str = "vector") -> Hit:
    return Hit(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_DNS, name),
        document_id=uuid.uuid4(),
        content=f"content of {name}",
        label=name,
        document_title="DPDP Act 2023",
        source_ref="DPDP-2023",
        score=0.0,
        matched_by=(matched_by,),
    )


def test_agreement_between_arms_outranks_a_single_strong_hit() -> None:
    """Two retrievers agreeing is the signal fusion exists to capture."""
    shared = hit("Section 9")
    vector = [hit("Section 3"), shared]
    lexical = [hit("Section 9", matched_by="lexical"), hit("Section 40")]

    fused = reciprocal_rank_fusion([vector, lexical], limit=3)
    assert fused[0].label == "Section 9"


def test_matched_by_records_every_contributing_arm() -> None:
    """The UI shows provenance, so it must survive fusion."""
    fused = reciprocal_rank_fusion(
        [[hit("Section 9")], [hit("Section 9", matched_by="lexical")]], limit=1
    )
    assert fused[0].matched_by == ("lexical", "vector")


def test_weights_scale_each_arm_contribution() -> None:
    """A down-weighted arm must not outvote a confident hit from the other.

    This is the failure that showed up on the DPDP eval set: the lexical arm's
    weak tail promoted mediocre chunks over the vector arm's top hit.
    """
    # "Section 40" is the vector arm's worst hit but the lexical arm's best.
    vector = [hit(f"Section {i}") for i in range(1, 10)] + [hit("Section 40")]
    lexical = [hit("Section 40", matched_by="lexical")]

    equal = reciprocal_rank_fusion([vector, lexical], limit=2)
    assert equal[0].label == "Section 40"

    # Down-weighted, the lexical arm can no longer overturn the dense ranking.
    weighted = reciprocal_rank_fusion([vector, lexical], limit=2, weights=[1.0, 0.1])
    assert weighted[0].label == "Section 1"


def test_zero_weight_removes_an_arm_from_ranking() -> None:
    vector = [hit("Section 5")]
    lexical = [hit("Section 40", matched_by="lexical")]

    fused = reciprocal_rank_fusion([vector, lexical], limit=2, weights=[1.0, 0.0])
    assert fused[0].label == "Section 5"


def test_limit_truncates_after_ranking_not_before() -> None:
    vector = [hit(f"Section {i}") for i in range(10)]
    lexical = [hit("Section 9", matched_by="lexical")]

    fused = reciprocal_rank_fusion([vector, lexical], limit=3)
    assert len(fused) == 3
    # Section 9 ranks last in the vector arm but is corroborated by lexical.
    assert "Section 9" in [h.label for h in fused]


def test_empty_rankings_are_safe() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([]) == []


def test_scores_are_monotonically_decreasing() -> None:
    vector = [hit(f"Section {i}") for i in range(5)]
    fused = reciprocal_rank_fusion([vector], limit=5)
    scores = [h.score for h in fused]
    assert scores == sorted(scores, reverse=True)


def test_citation_reads_as_a_source_reference() -> None:
    assert hit("Section 8(3)").citation == "DPDP Act 2023, Section 8(3)"


def test_citation_falls_back_to_title_without_a_label() -> None:
    unlabelled = hit("x")
    unlabelled.label = None
    assert unlabelled.citation == "DPDP Act 2023"
