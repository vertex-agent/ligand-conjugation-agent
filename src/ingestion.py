from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from .models import EvidenceRecord


REQUIRED_COLUMNS = [
    "source_id",
    "title",
    "year",
    "journal",
    "peer_reviewed",
    "approved",
    "nanoparticle_system",
    "ligand_class",
    "conjugation_chemistry",
    "payload_context",
    "protocol_conditions",
    "outcomes",
    "limitations",
]


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "approved"}


def _to_float_or_zero(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def validate_columns(df: pd.DataFrame) -> list[str]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing


def load_records(df: pd.DataFrame) -> List[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for _, row in df.iterrows():
        evidence_type = str(row.get("evidence_type", "peer_reviewed_protocol")).strip()
        if not evidence_type:
            evidence_type = "peer_reviewed_protocol"
        records.append(
            EvidenceRecord(
                source_id=str(row.get("source_id", "")).strip(),
                title=str(row.get("title", "")).strip(),
                year=str(row.get("year", "")).strip(),
                journal=str(row.get("journal", "")).strip(),
                peer_reviewed=_to_bool(row.get("peer_reviewed")),
                approved=_to_bool(row.get("approved")),
                nanoparticle_system=str(row.get("nanoparticle_system", "")).strip(),
                ligand_class=str(row.get("ligand_class", "")).strip(),
                conjugation_chemistry=str(row.get("conjugation_chemistry", "")).strip(),
                payload_context=str(row.get("payload_context", "")).strip(),
                protocol_conditions=str(row.get("protocol_conditions", "")).strip(),
                outcomes=str(row.get("outcomes", "")).strip(),
                limitations=str(row.get("limitations", "")).strip(),
                doi_or_pmid=str(row.get("doi_or_pmid", "")).strip() or None,
                quality_score=_to_float_or_zero(row.get("quality_score", 0.0)),
                evidence_type=evidence_type,
            )
        )
    return records


def approved_peer_reviewed(records: Iterable[EvidenceRecord]) -> List[EvidenceRecord]:
    return [r for r in records if r.approved and r.peer_reviewed and r.title and r.source_id]
