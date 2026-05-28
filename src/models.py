from dataclasses import dataclass
from typing import Optional


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
