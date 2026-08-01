"""Judgment corpus ingestion.

Judgments come from open datasets on the HuggingFace Hub, streamed over the
datasets-server HTTP API. The PRD named Indian Kanoon, which bills per document
with no bulk free tier; these corpora are pre-scraped, redistributable, and
free.

`JudgmentSource` is an interface so Kanoon can be dropped in later without
touching the pipeline, should paying for it ever make sense.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from caselens.citations import CitationRef, extract_citations, extract_header
from core.logging import get_logger

log = get_logger(__name__)

HF_ROWS_API = "https://datasets-server.huggingface.co/rows"

#: Rows per request. The API caps a page at 100.
PAGE_SIZE = 100


@dataclass(slots=True)
class RawJudgment:
    """One judgment, parsed but not yet stored."""

    source_ref: str
    title: str
    text: str
    year: int | None
    court: str | None
    citations: list[CitationRef]

    @property
    def meta(self) -> dict[str, object]:
        return {
            "year": self.year,
            "court": self.court,
            "jurisdiction": "India",
            "citation_count": len(self.citations),
        }


class JudgmentSource(ABC):
    """A source of judgments. Implement this to add a corpus."""

    @abstractmethod
    def stream(self, limit: int, offset: int = 0) -> AsyncIterator[RawJudgment]: ...


class HuggingFaceJudgments(JudgmentSource):
    """Judgments from a HuggingFace dataset.

    Default corpus is `ninadn/indian-legal`: 7,130 Indian Supreme Court
    judgments with full text.
    """

    def __init__(
        self,
        dataset: str = "ninadn/indian-legal",
        text_column: str = "Text",
        config: str = "default",
        split: str = "train",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._dataset = dataset
        self._text_column = text_column
        self._config = config
        self._split = split
        self._client = client or httpx.AsyncClient(timeout=90.0)
        self._owns_client = client is None

    async def _page(self, offset: int, length: int) -> list[dict[str, object]]:
        """One page, retried: the datasets server rate-limits and cold-starts."""
        last_error = "unknown"
        for attempt in range(4):
            try:
                response = await self._client.get(
                    HF_ROWS_API,
                    params={
                        "dataset": self._dataset,
                        "config": self._config,
                        "split": self._split,
                        "offset": offset,
                        "length": length,
                    },
                )
            except httpx.HTTPError as exc:
                last_error = str(exc)
            else:
                if response.status_code == 200:
                    rows = response.json().get("rows", [])
                    return [r["row"] for r in rows]
                last_error = f"HTTP {response.status_code}"
                if response.status_code < 500 and response.status_code != 429:
                    break

            await asyncio.sleep(2**attempt)

        log.warning("judgment_page_failed", offset=offset, error=last_error)
        return []

    async def stream(self, limit: int, offset: int = 0) -> AsyncIterator[RawJudgment]:
        fetched = 0
        while fetched < limit:
            page_size = min(PAGE_SIZE, limit - fetched)
            rows = await self._page(offset + fetched, page_size)
            if not rows:
                return

            for index, row in enumerate(rows):
                text = str(row.get(self._text_column) or "").strip()
                # Very short entries are headnotes or fragments, not judgments;
                # they add noise to retrieval without adding law.
                if len(text) < 1000:
                    continue

                header = extract_header(text)
                yield RawJudgment(
                    # Position-derived so re-ingesting updates in place. The
                    # corpus has no stable identifier of its own.
                    source_ref=f"{self._dataset}#{offset + fetched + index}",
                    title=header.title,
                    text=text,
                    year=header.year,
                    court=header.court,
                    citations=extract_citations(text),
                )

            fetched += len(rows)
            if len(rows) < page_size:
                return

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
