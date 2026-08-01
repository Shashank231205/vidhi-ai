"""Citation verification is the groundedness guarantee — test it adversarially."""

import uuid

from core.agents.verifier import (
    CitationVerifier,
    Rejection,
    normalise,
    quote_appears_in,
)
from core.retrieval import Hit

SECTION_8 = (
    "8. (5) Every Data Fiduciary shall protect personal data in its possession "
    "or under its control by taking reasonable security safeguards to prevent "
    "personal data breach."
)


def make_hit(content: str = SECTION_8, label: str = "Section 8(5)") -> Hit:
    return Hit(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_DNS, label),
        document_id=uuid.uuid4(),
        content=content,
        label=label,
        document_title="DPDP Act 2023",
        source_ref="DPDP-2023",
        score=1.0,
    )


def test_accepts_verbatim_quote() -> None:
    hit = make_hit()
    verifier = CitationVerifier([hit])

    result = verifier.verify(
        str(hit.chunk_id), "taking reasonable security safeguards to prevent personal data breach"
    )
    assert result.verified


def test_rejects_fabricated_chunk_id() -> None:
    """A model citing a chunk it was never shown is the core failure mode."""
    verifier = CitationVerifier([make_hit()])

    result = verifier.verify(str(uuid.uuid4()), "any quote at all here")
    assert not result.verified
    assert result.reason is Rejection.UNKNOWN_CHUNK


def test_rejects_quote_absent_from_the_chunk() -> None:
    """Plausible but invented statutory language must not pass."""
    hit = make_hit()
    verifier = CitationVerifier([hit])

    result = verifier.verify(
        str(hit.chunk_id),
        "the Data Fiduciary shall pay compensation of ten crore rupees",
    )
    assert not result.verified
    assert result.reason is Rejection.QUOTE_NOT_FOUND


def test_rejects_quote_too_short_to_verify() -> None:
    """'the Act' appears everywhere and proves nothing."""
    hit = make_hit()
    result = CitationVerifier([hit]).verify(str(hit.chunk_id), "personal data")

    assert not result.verified
    assert result.reason is Rejection.QUOTE_TOO_SHORT


def test_tolerates_pdf_extraction_artefacts() -> None:
    """Line breaks and smart quotes must not reject a correct citation."""
    hit = make_hit("The Data Fiduciary shall\nimplement reasonable\nsecurity safeguards.")
    verifier = CitationVerifier([hit])

    result = verifier.verify(
        str(hit.chunk_id), "The Data Fiduciary shall implement reasonable security safeguards."
    )
    assert result.verified


def test_tolerates_smart_punctuation_differences() -> None:
    # The curly apostrophe is the subject of this test, not an accident: PDF
    # extraction produces it where a model's quote uses a straight one.
    hit = make_hit("A Data Principal’s consent shall be free and specific in nature.")
    verifier = CitationVerifier([hit])

    result = verifier.verify(
        str(hit.chunk_id), "A Data Principal's consent shall be free and specific"
    )
    assert result.verified


def test_scattered_words_do_not_count_as_a_quotation() -> None:
    """Contiguity is what distinguishes a quote from a bag of shared words."""
    hit = make_hit()
    verifier = CitationVerifier([hit])

    result = verifier.verify(
        str(hit.chunk_id),
        "Data Fiduciary breach personal control possession safeguards prevent",
    )
    assert not result.verified


def test_whitespace_in_chunk_id_is_forgiven() -> None:
    hit = make_hit()
    result = CitationVerifier([hit]).verify(
        f"  {hit.chunk_id}  ", "reasonable security safeguards to prevent personal data breach"
    )
    assert result.verified


def test_citation_lookup_resolves_display_text() -> None:
    hit = make_hit()
    verifier = CitationVerifier([hit])

    assert verifier.citation_for(str(hit.chunk_id)) == "DPDP Act 2023, Section 8(5)"
    assert verifier.citation_for(str(uuid.uuid4())) is None


def test_empty_retrieval_rejects_everything() -> None:
    """With nothing retrieved, no claim can be grounded."""
    verifier = CitationVerifier([])
    assert not verifier.verify(str(uuid.uuid4()), "some quoted text here").verified


def test_normalise_collapses_case_and_spacing() -> None:
    assert normalise("The   DATA\nFiduciary") == "the data fiduciary"


def test_quote_ratio_reported_for_partial_match() -> None:
    matched, ratio = quote_appears_in(
        "reasonable security safeguards and additional invented words", SECTION_8
    )
    assert not matched
    assert 0.0 < ratio < 1.0
