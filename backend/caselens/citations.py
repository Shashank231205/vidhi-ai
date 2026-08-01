"""Citation extraction from judgment text.

The patterns here were derived by profiling the actual corpus rather than from
the textbook forms. Two things that survey turned up and that shape the code:

- Reporter citations appear with erratic internal spacing and stray periods —
  "[1955] 1 S.C. R. 313" is one citation, not three tokens. Patterns tolerate
  optional whitespace between every component.
- Case-name references ("Balmukund vs Motilal") are roughly fifty times more
  common than reporter citations in this corpus, so treating reporter formats
  as the primary signal would build an almost empty graph.

Both forms are extracted. A reference is normalised to a stable `target_ref`
so that two spellings of the same case converge on one graph node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: "[1955] 1 S.C. R. 313", "1952 SCR 478" — the volume is optional.
_SCR = re.compile(
    r"\[?(?P<year>1[89]\d{2}|20\d{2})\]?\s*"
    r"(?:(?P<volume>\d{1,2})\s*)?"
    r"S\.?\s*C\.?\s*R\.?\s*(?P<page>\d{1,4})",
    re.IGNORECASE,
)

#: "A.I.R. 1954 S.C. 229"
_AIR = re.compile(
    r"A\.?\s*I\.?\s*R\.?\s*\.?\s*(?P<year>1[89]\d{2}|20\d{2})\s*"
    r"(?P<court>S\.?\s*C\.?|[A-Z][a-z]{2,})\s*\.?\s*(?P<page>\d{1,4})",
    re.IGNORECASE,
)

#: "(1997) 6 SCC 241"
_SCC = re.compile(
    r"\((?P<year>1[89]\d{2}|20\d{2})\)\s*(?P<volume>\d{1,2})\s*"
    r"S\.?\s*C\.?\s*C\.?\s*(?P<page>\d{1,4})",
    re.IGNORECASE,
)

#: "Kesavananda Bharati vs State of Kerala".
#:
#: The defendant side is the hard one. Party names are capitalised and the
#: sentence continuing after them is not, so the capture runs to the end of the
#: capitalised run rather than to a punctuation mark — "Motilal governs this
#: dispute" would otherwise become the defendant's name. Lowercase connectives
#: inside a name ("State of Kerala", "Commissioner of Income Tax") are allowed
#: explicitly, since dropping them would split real parties.
_CASE_NAME = re.compile(
    r"(?P<plaintiff>[A-Z][A-Za-z&.'\-]*(?:\s+(?:of|the|and|for|&)?\s*[A-Z][A-Za-z&.'\-]*)"
    r"{0,6})\s+"
    r"v(?:s\.?|\.)\s+"
    r"(?P<defendant>[A-Z][A-Za-z&.'\-]*(?:\s+(?:of|the|and|for|&)?\s*[A-Z][A-Za-z&.'\-]*)"
    r"{0,6})",
    re.MULTILINE,
)

#: Trailing words that are sentence, not party. Judgments run straight on from
#: a case name into the verb, and the capture cannot tell without a stop list.
_TRAILING_VERBS = re.compile(
    r"\s+(?:argued|governs?|held|said|decided|applies|applied|observed|"
    r"laid|stated|is|was|were|has|had|shows?|makes?|and|in|at|that|which)\b.*$",
    re.IGNORECASE,
)

#: Judgments introduce precedents mid-sentence — "as observed by Lord Selborne
#: in Corbett v. X", "the Court in Y v. Z". Without stripping the lead-in, the
#: plaintiff swallows the narration and two references to the same case get
#: different identities.
_LEAD_IN = re.compile(
    r"^(?:"
    r".*?\b(?:in|by|see|per|of|from|followed|approved|cited|observed|held)\s+"
    r"(?:the\s+)?(?:case\s+(?:of\s+)?)?"
    r"|(?:[A-Z][a-z]*\s+)*?"
    r"(?:C\.?J\.?|J\.?J\.?|J\.?|Lord|Sir|Mr\.?|Justice)\s+"
    r")",
    re.IGNORECASE,
)

#: Court and reporter words that mean the capture is a venue, not a party.
_VENUE_WORDS = re.compile(
    r"\b(?:high court|supreme court|privy council|tribunal|bench|"
    r"pradesh|city|division)\b$",
    re.IGNORECASE,
)

#: Words that mean the "case name" is really a sentence fragment. Judgments say
#: "Counsel for the Appellant vs The Respondent" in ways that are not citations.
_NOT_A_PARTY = frozenset(
    {
        "the appellant",
        "the respondent",
        "the appellants",
        "the respondents",
        "appellant",
        "respondent",
        "petitioner",
        "the petitioner",
        "plaintiff",
        "defendant",
        "the plaintiff",
        "the defendant",
        "the state",
        "the union of india",
    }
)


@dataclass(frozen=True, slots=True)
class CitationRef:
    """One extracted reference to another case."""

    #: Normalised identity: two spellings of the same case share this.
    target_ref: str
    #: The text as it appeared, for display.
    raw: str
    kind: str
    #: Surrounding sentence — shows *why* the case was cited.
    context: str = ""


#: Roles, not parties. A capture containing one of these describes a litigant's
#: position in *this* case rather than naming a precedent.
_PLEADING_WORDS = re.compile(
    r"\b(?:appellants?|respondents?|petitioners?|plaintiffs?|defendants?|"
    r"counsel|applicants?)\b",
    re.IGNORECASE,
)


def _is_pleading_vocabulary(name: str) -> bool:
    return name in _NOT_A_PARTY or bool(_PLEADING_WORDS.search(name))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .,;:'\"-")


def _context_around(text: str, start: int, end: int, width: int = 160) -> str:
    left = max(0, start - width // 2)
    right = min(len(text), end + width // 2)
    return _clean(text[left:right])


def _normalise_party(name: str) -> str:
    name = _clean(name)
    # Drop narration preceding the party name: "as observed by Lord X in Y"
    # must normalise to "y", or the same case gets two graph identities.
    name = _LEAD_IN.sub("", name, count=1)
    # Drop the sentence continuing after it: "Motilal governs this dispute".
    name = _TRAILING_VERBS.sub("", name)
    name = _clean(name).lower()
    # Corporate suffixes vary between reports of the same case.
    name = re.sub(r"\b(?:ltd|pvt|co|inc|corp|company|limited)\b\.?", "", name)
    name = re.sub(r"^(?:the|m/s)\s+", "", name)
    return _clean(name)


def extract_citations(text: str, *, max_refs: int = 200) -> list[CitationRef]:
    """Pull every case reference out of a judgment.

    Deduplicated by `target_ref`: a judgment that relies on one precedent five
    times should produce one graph edge, not five.
    """
    found: dict[str, CitationRef] = {}

    for pattern, kind in ((_SCR, "SCR"), (_AIR, "AIR"), (_SCC, "SCC")):
        for match in pattern.finditer(text):
            groups = match.groupdict()
            volume = groups.get("volume") or ""
            target = f"{kind}-{groups['year']}-{volume}-{groups['page']}".replace("--", "-")
            found.setdefault(
                target,
                CitationRef(
                    target_ref=target,
                    raw=_clean(match.group(0)),
                    kind=kind,
                    context=_context_around(text, match.start(), match.end()),
                ),
            )
            if len(found) >= max_refs:
                return list(found.values())

    for match in _CASE_NAME.finditer(text):
        plaintiff = _normalise_party(match.group("plaintiff"))
        defendant = _normalise_party(match.group("defendant"))

        # Both sides must be real party names, not pleading vocabulary or a
        # court the sentence happened to name. Checked as a substring: the
        # capture is often "counsel for the appellant", not bare "appellant".
        if _is_pleading_vocabulary(plaintiff) or _is_pleading_vocabulary(defendant):
            continue
        if len(plaintiff) < 3 or len(defendant) < 3:
            continue
        if _VENUE_WORDS.search(plaintiff) or _VENUE_WORDS.search(defendant):
            continue

        target = f"CASE-{plaintiff}-v-{defendant}"[:255]
        found.setdefault(
            target,
            CitationRef(
                target_ref=target,
                raw=_clean(match.group(0)),
                kind="CASE",
                context=_context_around(text, match.start(), match.end()),
            ),
        )
        if len(found) >= max_refs:
            break

    return list(found.values())


#: "Appeal No. 73 of 1950", "Appeals Nos. 181 to 184 of 1960", "Petition No..."
_APPEAL_NO = re.compile(
    r"(?P<kind>Appeals?|Petitions?|Writ Petitions?|Civil Appeals?|"
    r"Criminal Appeals?)\s+Nos?\.?\s*(?P<number>[\d]+(?:\s*(?:to|-|and)\s*\d+)?)"
    r"\s+of\s+(?P<year>1[89]\d{2}|20\d{2})",
    re.IGNORECASE,
)

_COURT = re.compile(
    r"(?:of the|of|the)\s+(?P<court>[A-Z][A-Za-z ]{0,25}?High Court|Supreme Court"
    r"|Industrial Tribunal|Privy Council)",
)


@dataclass(frozen=True, slots=True)
class JudgmentHeader:
    """What a judgment's opening lines reveal about it.

    This corpus carries no structured metadata, so identity is reconstructed
    from the header. The appeal number is the only reliably unique field —
    party names are frequently absent from the text entirely.
    """

    title: str
    year: int | None
    court: str | None
    appeal_number: str | None


def extract_header(text: str) -> JudgmentHeader:
    head = text[:2000]

    appeal = _APPEAL_NO.search(head)
    appeal_number = _clean(appeal.group(0)) if appeal else None

    court_match = _COURT.search(head)
    court = _clean(court_match.group("court")) if court_match else None

    year: int | None = int(appeal.group("year")) if appeal else None
    if year is None:
        loose = re.search(r"\b(1[89]\d{2}|20\d{2})\b", head)
        year = int(loose.group(0)) if loose else None

    # Prefer a real case name; fall back to the appeal number, which is what
    # most of these judgments actually carry.
    title: str | None = None
    name = _CASE_NAME.search(head)
    if name:
        plaintiff = _clean(_LEAD_IN.sub("", _clean(name.group("plaintiff")), count=1))
        defendant = _clean(name.group("defendant"))
        if (
            _normalise_party(plaintiff) not in _NOT_A_PARTY
            and _normalise_party(defendant) not in _NOT_A_PARTY
            and len(plaintiff) > 2
        ):
            title = f"{plaintiff} v. {defendant}"

    if not title:
        title = appeal_number or (f"Judgment ({year})" if year else "Untitled judgment")

    return JudgmentHeader(title=title[:250], year=year, court=court, appeal_number=appeal_number)
