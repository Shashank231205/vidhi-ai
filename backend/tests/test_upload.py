"""Upload validation. Uploads are untrusted input, so the rejections matter."""

import io

import pytest
from pypdf import PdfWriter

from core.db.models import DocumentKind
from core.ingestion.upload import MAX_UPLOAD_BYTES, UploadError, extract_upload


def make_pdf(text: str = "", pages: int = 1) -> bytes:
    """A minimal valid PDF. pypdf writes no text layer, so callers that need
    extractable text should use the real fixture instead."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_rejects_empty_file() -> None:
    with pytest.raises(UploadError, match="empty"):
        extract_upload(b"", "contract.pdf")


def test_rejects_non_pdf_by_content_not_extension() -> None:
    """A .pdf name on a ZIP must not get through."""
    with pytest.raises(UploadError, match="Only PDF"):
        extract_upload(b"PK\x03\x04 not a pdf", "malicious.pdf")


def test_rejects_oversized_file() -> None:
    oversized = b"%PDF" + b"\x00" * MAX_UPLOAD_BYTES
    with pytest.raises(UploadError, match="larger than"):
        extract_upload(oversized, "huge.pdf")


def test_rejects_pdf_without_extractable_text() -> None:
    """Scans need OCR; ingesting them would create an empty document."""
    with pytest.raises(UploadError, match="No readable text"):
        extract_upload(make_pdf(), "scanned.pdf")


def test_rejects_corrupt_pdf() -> None:
    with pytest.raises(UploadError, match="could not be read"):
        extract_upload(b"%PDF-1.4 truncated garbage", "broken.pdf")


def test_title_strips_path_components() -> None:
    """Filenames are attacker-controlled and end up in responses."""
    from core.ingestion.upload import _safe_title

    assert _safe_title("../../etc/passwd.pdf") == "passwd"
    assert _safe_title("C:\\Users\\x\\Vendor Agreement.pdf") == "Vendor Agreement"


def test_title_strips_control_characters() -> None:
    from core.ingestion.upload import _safe_title

    assert _safe_title("contract\x00\x1b[31m.pdf") == "contract[31m"


def test_title_falls_back_when_empty() -> None:
    from core.ingestion.upload import _safe_title

    assert _safe_title(".pdf") == "Untitled document"


def test_source_ref_is_content_addressed(dpdp_pdf: bytes) -> None:
    """Re-uploading the same file must update in place, not duplicate."""
    first = extract_upload(dpdp_pdf, "a.pdf")
    second = extract_upload(dpdp_pdf, "differently-named.pdf")

    assert first.source_ref == second.source_ref
    assert first.source_ref.startswith("upload-")


def test_extracts_real_document(dpdp_pdf: bytes) -> None:
    extracted = extract_upload(dpdp_pdf, "DPDP Act.pdf", kind=DocumentKind.CONTRACT)

    assert extracted.title == "DPDP Act"
    assert extracted.kind is DocumentKind.CONTRACT
    assert extracted.pages > 1
    assert "personal data" in extracted.text.lower()
