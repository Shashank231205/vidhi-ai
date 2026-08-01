"""Fetching statute text from official sources.

India Code serves DSpace handle pages rather than stable PDF URLs, so the PDF
is discovered from the handle at fetch time. It also rejects requests without a
browser user-agent, which is why one is set explicitly.
"""

from __future__ import annotations

import re
from io import BytesIO

import httpx
from pypdf import PdfReader

from core.ingestion.sources import StatuteSource
from core.logging import get_logger

log = get_logger(__name__)

#: India Code returns 404 to default client user-agents.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_BITSTREAM = re.compile(r'/bitstream/123456789/\d+/\d+/[^"\'\s>]+\.pdf', re.IGNORECASE)

#: Hindi versions sit under the same handle, prefixed H. We ingest English.
_HINDI = re.compile(r"/H\d", re.IGNORECASE)


class FetchError(RuntimeError):
    """Raised when a source cannot be turned into usable text."""


async def resolve_pdf_url(client: httpx.AsyncClient, source: StatuteSource) -> str:
    """The PDF URL for a source, discovering it from the handle if needed."""
    if source.url:
        return source.url

    handle_url = source.handle_url
    if handle_url is None:  # pragma: no cover - guarded by __post_init__
        raise FetchError(f"{source.key}: no handle or url")

    response = await client.get(
        handle_url, headers={"User-Agent": BROWSER_UA}, follow_redirects=True
    )
    response.raise_for_status()

    paths = list(dict.fromkeys(_BITSTREAM.findall(response.text)))
    english = [p for p in paths if not _HINDI.search(p)]
    if not english:
        raise FetchError(f"{source.key}: no English PDF found at {handle_url}")

    return f"https://www.indiacode.nic.in{english[0]}"


def extract_text(content: bytes, *, label: str) -> str:
    """Pull text out of a PDF, rejecting scans rather than ingesting nothing."""
    reader = PdfReader(BytesIO(content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text:
        raise FetchError(f"{label}: no extractable text (scanned image?)")
    return text


async def fetch_statute_text(client: httpx.AsyncClient, source: StatuteSource) -> tuple[str, str]:
    """Return `(text, resolved_url)` for a statute source."""
    url = await resolve_pdf_url(client, source)
    response = await client.get(url, headers={"User-Agent": BROWSER_UA}, follow_redirects=True)
    response.raise_for_status()

    text = extract_text(response.content, label=source.key)
    log.info("fetched_statute", key=source.key, characters=len(text), bytes=len(response.content))
    return text, url
