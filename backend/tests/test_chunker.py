"""Chunking determines what can be cited, so the boundaries are the contract."""

from core.ingestion.chunker import chunk_legal_text, estimate_tokens

STATUTE = """
CHAPTER II
OBLIGATIONS OF DATA FIDUCIARY

4. Grounds for processing personal data. A person may process the personal data
of a Data Principal only in accordance with the provisions of this Act and for a
lawful purpose for which the Data Principal has given her consent.

5. Notice. Every request made to a Data Principal for consent shall be
accompanied or preceded by a notice given by the Data Fiduciary informing the
Data Principal the personal data and the purpose of processing.

6. Consent. The consent given by the Data Principal shall be free, specific,
informed, unconditional and unambiguous with a clear affirmative action.
"""


def test_splits_on_section_boundaries() -> None:
    chunks = chunk_legal_text(STATUTE)
    labels = [c.label for c in chunks]

    assert "Section 4" in labels
    assert "Section 5" in labels
    assert "Section 6" in labels


def test_each_chunk_holds_exactly_one_section() -> None:
    """A chunk spanning two provisions makes its citation ambiguous."""
    chunks = chunk_legal_text(STATUTE)
    section_5 = next(c for c in chunks if c.label == "Section 5")

    assert "Notice" in section_5.content
    assert "Consent. The consent given" not in section_5.content


def test_chapter_is_recorded_as_metadata() -> None:
    chunks = chunk_legal_text(STATUTE)
    assert all("CHAPTER II" in c.meta.get("chapter", "") for c in chunks)


def test_ordinals_are_contiguous_from_zero() -> None:
    """Ordinal is the citation ordering and has a uniqueness constraint on it."""
    chunks = chunk_legal_text(STATUTE)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_cross_reference_does_not_start_a_new_chunk() -> None:
    """'...under section 5' mid-sentence is a reference, not a heading."""
    text = (
        "7. Duties. A Data Fiduciary shall comply with the obligations under "
        "section 5 and section 6 of this Act, and shall not process data in "
        "contravention of section 4 in any manner whatsoever."
    )
    chunks = chunk_legal_text(text, min_tokens=1)
    assert len(chunks) == 1
    assert chunks[0].label == "Section 7"


def test_oversized_section_splits_at_subsections() -> None:
    """Sub-sections are independently citable, so they are the split point."""
    # Each sub-section fits the budget on its own; only the whole section does not.
    body = " ".join(["The Data Fiduciary shall implement safeguards."] * 5)
    text = f"8. Security safeguards.\n(1) {body}\n(2) {body}\n(3) {body}"

    chunks = chunk_legal_text(text, max_tokens=120)
    labels = [c.label for c in chunks]

    assert "Section 8(1)" in labels
    assert "Section 8(2)" in labels
    assert "Section 8(3)" in labels


def test_subsection_carries_section_heading_for_context() -> None:
    """'(3) shall do X' is meaningless without knowing which section it is in."""
    body = " ".join(["obligation text here."] * 40)
    text = f"8. Security safeguards.\n(1) {body}\n(2) {body}"

    chunks = chunk_legal_text(text, max_tokens=80)
    assert all("Security safeguards" in c.content for c in chunks)


def test_single_huge_provision_falls_back_to_windows() -> None:
    """A section with no sub-structure must still be broken up."""
    text = "9. Long provision. " + " ".join(
        [f"Sentence number {i} of this provision." for i in range(200)]
    )
    chunks = chunk_legal_text(text, max_tokens=120)

    assert len(chunks) > 1
    assert all(c.token_count <= 200 for c in chunks)
    assert all("part" in (c.label or "") for c in chunks)


def test_windows_overlap_so_boundary_text_stays_retrievable() -> None:
    text = "9. Provision. " + " ".join([f"Clause {i} states an obligation." for i in range(120)])
    chunks = chunk_legal_text(text, max_tokens=100, overlap_tokens=40)

    first_words = set(chunks[0].content.split())
    second_words = set(chunks[1].content.split())
    assert first_words & second_words


def test_hyphenation_across_line_breaks_is_repaired() -> None:
    """PDF extraction splits words; unrepaired, they never match a search."""
    chunks = chunk_legal_text("4. Notice. The informa-\ntion shall be provided.", min_tokens=1)
    assert "information" in chunks[0].content


def test_page_numbers_are_stripped() -> None:
    text = "4. Grounds. Consent must be free and specific.\n\n12\n\nand informed."
    chunks = chunk_legal_text(text, min_tokens=1)
    assert "\n12\n" not in chunks[0].content


def test_unstructured_text_still_produces_a_chunk() -> None:
    """Contracts and policies have no section numbering; nothing may be lost."""
    text = "This agreement is made between the parties on the date written above."
    chunks = chunk_legal_text(text, min_tokens=1)

    assert len(chunks) == 1
    assert chunks[0].label is None


def test_empty_input_yields_nothing() -> None:
    assert chunk_legal_text("") == []
    assert chunk_legal_text("   \n\n  ") == []


def test_as_row_matches_repository_shape() -> None:
    row = chunk_legal_text(STATUTE)[0].as_row()
    assert set(row) == {"ordinal", "content", "label", "token_count", "meta"}


def test_token_estimate_is_monotonic() -> None:
    assert estimate_tokens("short") < estimate_tokens("a much longer piece of text here")


def test_table_of_contents_dot_leaders_are_stripped() -> None:
    """'Notice ....... 12' survives PDF extraction as noise.

    The dots read as sentence boundaries to the splitter and match nothing
    useful in full-text search.
    """
    chunks = chunk_legal_text(
        "4. Notice. " + "Consent must be free, specific and informed. " * 4,
        min_tokens=1,
    )
    assert "......" not in chunks[0].content


def test_contents_page_fragments_are_dropped() -> None:
    """A real extraction artefact from the Companies Act PDF.

    These clear the token budget but carry no citable prose, and looked like a
    broken product when opened as the source behind a citation.
    """
    statute = (
        "3. \n. \n. \nTotal\nJoint Ventures\n\n"
        "4. Grounds for processing. A person may process the personal data of a "
        "Data Principal only in accordance with the provisions of this Act and "
        "for a lawful purpose for which consent has been given."
    )
    labels = [c.label for c in chunk_legal_text(statute)]

    assert "Section 4" in labels
    assert "Section 3" not in labels


def test_a_short_document_still_produces_a_chunk() -> None:
    """The substance filter must never empty a document entirely."""
    assert len(chunk_legal_text("Short agreement text.", min_tokens=1)) == 1
