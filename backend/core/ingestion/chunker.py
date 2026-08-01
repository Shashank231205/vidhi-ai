"""Legal-aware chunking.

Fixed-width windows are wrong for statutes. A section is the unit a lawyer
cites, so splitting mid-section produces chunks that are individually
meaningless and citations that cannot be verified against a real provision.

This splits on statutory structure — sections, sub-sections, clauses — and only
falls back to sentence packing when a single provision exceeds the size budget.
Every chunk carries the `label` it was found under ("Section 8(3)"), which is
what the citation verifier later checks against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Rough chars-per-token for English legal prose. Used only for budgeting, so
#: an approximation is fine; exact counts would need the model's tokenizer.
CHARS_PER_TOKEN = 4

DEFAULT_MAX_TOKENS = 512
DEFAULT_MIN_TOKENS = 24
DEFAULT_OVERLAP_TOKENS = 48


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


#: A numbered section heading: "5. Grounds for processing" or "Section 5."
#: Anchored to line starts, which is what distinguishes a heading from an
#: in-text cross-reference like "as described in section 5".
_SECTION = re.compile(
    r"^\s*(?:Section\s+)?(\d+[A-Z]?)\.\s+(?=\S)",
    re.MULTILINE | re.IGNORECASE,
)

#: A chapter heading: "CHAPTER II — OBLIGATIONS OF DATA FIDUCIARY".
_CHAPTER = re.compile(
    r"^\s*CHAPTER\s+([IVXLC]+|\d+)\b[^\n]*",
    re.MULTILINE,
)

#: Sub-provisions inside a section: "(1)", "(a)", "(iv)".
_SUBSECTION = re.compile(r"^\s*\((\d+|[a-z]{1,3}|[ivxl]{1,6})\)\s+", re.MULTILINE)

_SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


@dataclass(slots=True)
class Chunk:
    """One retrievable span, ready for the chunks table."""

    ordinal: int
    content: str
    label: str | None = None
    token_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "content": self.content,
            "label": self.label,
            "token_count": self.token_count,
            "meta": self.meta,
        }


@dataclass(slots=True)
class _Span:
    """An intermediate structural unit before size budgeting."""

    label: str | None
    content: str
    chapter: str | None


def _normalise(text: str) -> str:
    """Collapse PDF artefacts that would otherwise break heading detection."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # PDF extraction emits non-breaking spaces, which break word-boundary
    # matching in the heading regexes and in full-text search.
    text = text.replace("\u00a0", " ")
    # Join words hyphenated across a line break: "informa-\ntion" -> "information".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Drop page furniture: a line that is only a number.
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    # Collapse table-of-contents dot leaders. "Notice .......... 12" survives
    # extraction as a run of periods that reads as sentence boundaries to the
    # chunker and as noise to full-text search.
    text = re.sub(r"[.·…]{3,}", " ", text)
    text = re.sub(r"(?:\n\s*\.\s*){2,}\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


#: A chunk needs this many letters to be worth retrieving. Below it, the span
#: is a table row, a page header, or a contents entry — text that matches
#: queries on stray words while establishing nothing a citation could rest on.
MIN_ALPHA_CHARS = 80


def _is_substantive(text: str) -> bool:
    """Whether a chunk carries enough prose to be citable.

    Statute PDFs contain contents pages and financial tables that extract as
    fragments like "3. \\n. \\n. \\nTotal". They pollute retrieval and, worse,
    look like a broken product when a reviewer opens the source behind a
    citation.
    """
    letters = sum(1 for character in text if character.isalpha())
    if letters < MIN_ALPHA_CHARS:
        return False

    # Mostly-punctuation spans clear the letter count only when very long.
    return letters / max(len(text), 1) >= 0.45


def _chapter_at(text: str, position: int) -> str | None:
    """The most recent chapter heading at or before `position`."""
    current: str | None = None
    for match in _CHAPTER.finditer(text):
        if match.start() > position:
            break
        current = match.group(0).strip()
    return current


def _split_sections(text: str) -> list[_Span]:
    """Split on section headings, tagging each with its chapter."""
    matches = list(_SECTION.finditer(text))
    if not matches:
        return [_Span(label=None, content=text, chapter=None)]

    spans: list[_Span] = []

    preamble = text[: matches[0].start()].strip()
    if estimate_tokens(preamble) >= DEFAULT_MIN_TOKENS:
        spans.append(_Span(label="Preamble", content=preamble, chapter=None))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.start() : end].strip()
        if not body:
            continue
        spans.append(
            _Span(
                label=f"Section {match.group(1)}",
                content=body,
                chapter=_chapter_at(text, match.start()),
            )
        )
    return spans


def _split_subsections(span: _Span) -> list[_Span]:
    """Break an oversized section at its sub-provision boundaries.

    Sub-sections are independently citable ("Section 8(3)"), so this keeps
    citations precise rather than splitting arbitrarily.
    """
    matches = list(_SUBSECTION.finditer(span.content))
    if len(matches) < 2:
        return [span]

    parts: list[_Span] = []
    head = span.content[: matches[0].start()].strip()

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(span.content)
        body = span.content[match.start() : end].strip()
        if not body:
            continue
        # Prepend the section's opening words so a sub-section reads standalone
        # — "(3) The Data Fiduciary shall..." loses its subject otherwise.
        content = f"{head}\n{body}" if head else body
        parts.append(
            _Span(
                label=f"{span.label}({match.group(1)})" if span.label else None,
                content=content,
                chapter=span.chapter,
            )
        )
    return parts or [span]


def _pack_sentences(span: _Span, max_tokens: int, overlap_tokens: int) -> list[_Span]:
    """Last resort: window a single provision that is still too large.

    Overlapping windows keep a definition that straddles a boundary retrievable
    from either side.
    """
    sentences = _SENTENCE_END.split(span.content)
    windows: list[_Span] = []
    current: list[str] = []
    current_tokens = 0

    def flush(part: int) -> None:
        if not current:
            return
        windows.append(
            _Span(
                label=f"{span.label} [part {part}]" if span.label else None,
                content=" ".join(current).strip(),
                chapter=span.chapter,
            )
        )

    for sentence in sentences:
        tokens = estimate_tokens(sentence)
        if current and current_tokens + tokens > max_tokens:
            flush(len(windows) + 1)
            # Carry the tail forward as overlap.
            carried: list[str] = []
            carried_tokens = 0
            for previous in reversed(current):
                previous_tokens = estimate_tokens(previous)
                if carried_tokens + previous_tokens > overlap_tokens:
                    break
                carried.insert(0, previous)
                carried_tokens += previous_tokens
            current = [*carried, sentence]
            current_tokens = carried_tokens + tokens
        else:
            current.append(sentence)
            current_tokens += tokens

    flush(len(windows) + 1)
    return windows or [span]


def chunk_legal_text(
    text: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split statutory text into citable, retrievable chunks.

    Structure is preferred over size: a section under budget stays whole even
    if small, because a whole provision retrieves and cites better than a
    fragment of one.
    """
    normalised = _normalise(text)
    if not normalised:
        return []

    spans: list[_Span] = []
    for section in _split_sections(normalised):
        if estimate_tokens(section.content) <= max_tokens:
            spans.append(section)
            continue

        for sub in _split_subsections(section):
            if estimate_tokens(sub.content) <= max_tokens:
                spans.append(sub)
            else:
                spans.extend(_pack_sentences(sub, max_tokens, overlap_tokens))

    chunks: list[Chunk] = []
    for span in spans:
        content = span.content.strip()
        # Drop fragments too small to carry meaning (stray headings, page noise)
        # unless they are the only thing we have.
        if estimate_tokens(content) < min_tokens and len(spans) > 1:
            continue
        # Same reasoning for contents pages and table rows: they clear the
        # token budget but carry no citable prose. Kept when nothing else
        # survives, so a short document never ingests as empty.
        if not _is_substantive(content) and len(spans) > 1:
            continue
        meta: dict[str, Any] = {}
        if span.chapter:
            meta["chapter"] = span.chapter
        chunks.append(
            Chunk(
                ordinal=len(chunks),
                content=content,
                label=span.label,
                token_count=estimate_tokens(content),
                meta=meta,
            )
        )
    return chunks
