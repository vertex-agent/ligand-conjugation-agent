from dataclasses import dataclass
from typing import Optional


EVIDENCE_TIER_LABELS = {
    "peer_reviewed_protocol": "Peer-reviewed protocol",
    "decision_matrix": "Chemistry decision matrix",
    "user_upload": "User-uploaded literature",
}


@dataclass
class EvidenceRecord:
    source_id: str
    title: str
    year: str
    journal: str
    peer_reviewed: bool
    approved: bool
    nanoparticle_system: str
    ligand_class: str
    conjugation_chemistry: str
    payload_context: str
    protocol_conditions: str
    outcomes: str
    limitations: str
    doi_or_pmid: Optional[str] = None
    quality_score: float = 0.0
    evidence_type: str = "peer_reviewed_protocol"

    @property
    def evidence_tier_label(self) -> str:
        return EVIDENCE_TIER_LABELS.get(self.evidence_type, self.evidence_type)

    @property
    def retrieval_text(self) -> str:
        return " | ".join(
            [
                self.title,
                self.nanoparticle_system,
                self.ligand_class,
                self.conjugation_chemistry,
                self.payload_context,
                self.protocol_conditions,
                self.outcomes,
                self.limitations,
            ]
        )
