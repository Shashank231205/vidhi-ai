"""User document uploads.

Uploads are untrusted input, so this module is deliberately strict: it checks
the magic bytes rather than the filename, caps the size before reading the
whole body into memory, and rejects encrypted or unextractable PDFs with a
message that says what to do about it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from core.db.models import DocumentKind

#: 20MB. Comfortably above a long commercial contract, well below anything
#: that would threaten a small container's memory.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

PDF_MAGIC = b"%PDF"

#: Uploaded text must contain at least this much extractable content to be
#: worth ingesting; below it, the file is almost certainly a scan.
MIN_EXTRACTED_CHARS = 200


class UploadError(ValueError):
    """Rejected upload. The message is safe to show the user."""


@dataclass(slots=True)
class ExtractedDocument:
    title: str
    text: str
    kind: DocumentKind
    source_ref: str
    pages: int
    characters: int


def _safe_title(filename: str) -> str:
    """A display title from an untrusted filename.

    Strips any path component and control characters — the value ends up in
    responses and the UI, so it must not carry a traversal path or markup.
    """
    base = filename.replace("\\", "/").split("/")[-1]
    base = re.sub(r"\.pdf$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    base = re.sub(r"\s+", " ", base)
    return base[:200] or "Untitled document"


def extract_upload(
    content: bytes,
    filename: str,
    *,
    kind: DocumentKind = DocumentKind.CONTRACT,
) -> ExtractedDocument:
    """Validate an uploaded PDF and extract its text.

    Raises UploadError with a user-facing reason for anything unusable.
    """
    if not content:
        raise UploadError("The file is empty.")

    if len(content) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise UploadError(f"File is larger than the {limit_mb}MB limit.")

    # Trust the content, not the extension.
    if not content.startswith(PDF_MAGIC):
        raise UploadError("Only PDF files are supported.")

    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            # An empty-password decrypt covers PDFs that are merely
            # permission-locked, which is common for legal documents.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise UploadError(
                    "This PDF is password-protected. Remove the password and retry."
                ) from exc

        pages = [page.extract_text() or "" for page in reader.pages]
    except UploadError:
        raise
    except (PdfReadError, Exception) as exc:
        raise UploadError("The PDF could not be read; it may be corrupt.") from exc

    text = "\n".join(pages).strip()
    if len(text) < MIN_EXTRACTED_CHARS:
        raise UploadError("No readable text found. Scanned documents need OCR before upload.")

    # Content-addressed ref: re-uploading the same file updates in place rather
    # than creating a duplicate document.
    digest = hashlib.sha256(content).hexdigest()[:16]

    return ExtractedDocument(
        title=_safe_title(filename),
        text=text,
        kind=kind,
        source_ref=f"upload-{digest}",
        pages=len(reader.pages),
        characters=len(text),
    )
