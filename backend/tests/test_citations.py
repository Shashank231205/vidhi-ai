"""Citation extraction feeds the graph, so a wrong identity is a wrong edge.

Fixtures are lifted from the real corpus (ninadn/indian-legal), including its
extraction artefacts — spaced-out reporter names and mid-sentence references
are what the parser actually meets.
"""

from caselens.citations import extract_citations, extract_header


def test_extracts_scr_citation_despite_erratic_spacing() -> None:
    """'[1955] 1 S.C. R. 313' is one citation; the corpus really looks like this."""
    refs = extract_citations("Relying on [1955] 1 S.C. R. 313, the Court held otherwise.")

    assert len(refs) == 1
    assert refs[0].kind == "SCR"
    assert refs[0].target_ref == "SCR-1955-1-313"


def test_extracts_air_citation() -> None:
    refs = extract_citations("See A.I.R. 1954 S.C. 229 for the contrary view here.")
    assert any(r.kind == "AIR" for r in refs)


def test_extracts_scc_citation() -> None:
    refs = extract_citations("This was settled in (1997) 6 SCC 241 conclusively.")
    assert any(r.target_ref == "SCC-1997-6-241" for r in refs)


def test_extracts_case_name_reference() -> None:
    refs = extract_citations("The rule in Balmukund vs Motilal governs this dispute.")

    assert len(refs) == 1
    assert refs[0].target_ref == "CASE-balmukund-v-motilal"


def test_strips_narration_before_the_party_name() -> None:
    """Two references to one case must produce one node, not two.

    'as observed by Lord Selborne in Corbett v. X' previously yielded a
    plaintiff of 'as was observed by lord selborne in corbett'.
    """
    narrated = extract_citations("As was observed by Lord Selborne in Corbett vs Smith, ...")
    plain = extract_citations("In Corbett vs Smith the position was different.")

    assert narrated[0].target_ref == plain[0].target_ref


def test_strips_judicial_titles() -> None:
    refs = extract_citations("Jenkins C.J. in Balmukund vs Motilal said as much.")
    assert refs[0].target_ref == "CASE-balmukund-v-motilal"


def test_ignores_pleading_vocabulary() -> None:
    """'the Appellant vs the Respondent' is not a citation."""
    assert extract_citations("Counsel for the Appellant vs The Respondent argued.") == []


def test_ignores_court_names_as_parties() -> None:
    refs = extract_citations("The Andhra Pradesh High Court vs Something Else here.")
    assert all("high court" not in r.target_ref for r in refs)


def test_corporate_suffixes_normalise_to_one_identity() -> None:
    """A case reported with and without 'Ltd.' is the same case."""
    with_suffix = extract_citations("In Travancore Rubber Ltd. vs Commissioner of Tax.")
    without = extract_citations("In Travancore Rubber vs Commissioner of Tax.")

    assert with_suffix[0].target_ref == without[0].target_ref


def test_repeated_citation_yields_one_edge() -> None:
    """A judgment relying on a precedent five times still cites it once."""
    text = "Balmukund vs Motilal is clear. " * 5
    assert len(extract_citations(text)) == 1


def test_citation_carries_its_context() -> None:
    """Context shows why a case was cited, not merely that it was."""
    refs = extract_citations(
        "The appellant relied on Balmukund vs Motilal to argue that specific "
        "performance was unavailable in these circumstances."
    )
    assert "specific performance" in refs[0].context


def test_max_refs_is_respected() -> None:
    text = " ".join(f"Party{i} vs Other{i}." for i in range(50))
    assert len(extract_citations(text, max_refs=10)) <= 10


def test_header_reads_appeal_number_as_title() -> None:
    """Most judgments in this corpus carry no case name at all."""
    header = extract_header(
        "Appeal No. 73 of 1950.\nAppeal from the Judgment and Decree dated "
        "the 26th January, 1944, of the High Court of Judicature at Patna."
    )

    assert header.appeal_number == "Appeal No. 73 of 1950"
    assert header.year == 1950
    assert "Appeal No. 73" in header.title


def test_header_reads_the_court() -> None:
    header = extract_header(
        "Appeals Nos. 181 to 184 of 1960.\nfrom the judgment and order dated "
        "March 16, 1955, of the Madras High Court in Case Referred No. 43."
    )
    assert header.court == "Madras High Court"


def test_header_prefers_a_real_case_name() -> None:
    header = extract_header("Kesavananda Bharati vs State of Kerala.\nAppeal No. 1 of 1972.")
    assert header.title.startswith("Kesavananda Bharati v.")


def test_header_degrades_gracefully() -> None:
    header = extract_header("Some text with no recognisable header at all.")
    assert header.title
    assert header.appeal_number is None
