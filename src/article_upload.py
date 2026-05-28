from __future__ import annotations

import io
from typing import Optional

from pypdf import PdfReader

from .models import EvidenceRecord


def _read_pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts).strip()


def _read_txt_text(content: bytes) -> str:
    return content.decode("utf-8", errors="replace").strip()


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def article_to_record(
    *,
    filename: str,
    content: bytes,
    max_chars: int = 20_000,
) -> Optional[EvidenceRecord]:
    """
    Convert a user-uploaded paper (PDF/TXT) into an EvidenceRecord so it can be
    retrieved/cited alongside the uploaded sheet (and only when explicitly provided).
    """
    name = (filename or "").strip()
    lower = name.lower()

    if lower.endswith(".pdf"):
        text = _read_pdf_text(content)
    elif lower.endswith(".txt"):
        text = _read_txt_text(content)
    else:
        return None

    text = _clip(text, max_chars=max_chars).strip()
    if not text:
        return None

    display_title = name.rsplit(".", 1)[0] if "." in name else name
    return EvidenceRecord(
        source_id=f"uploaded:{name}",
        title=display_title or "Uploaded article",
        year="uploaded",
        journal="user_upload",
        peer_reviewed=True,
        approved=True,
        nanoparticle_system="",
        ligand_class="",
        conjugation_chemistry="",
        payload_context="",
        protocol_conditions="(Extracted from uploaded article)\n" + _clip(text, 6_000),
        outcomes=_clip(text, 8_000),
        limitations="Uploaded document text may include methods/results outside the target context.",
        doi_or_pmid=None,
        quality_score=0.0,
    )

