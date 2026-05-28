from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .orchestrator import PromptOrchestrationTrace
from .retrieval import RetrievedEvidence


@dataclass
class AgentResponse:
    status: str
    interpreted_constraints: str
    protocol_options: List[str]
    evidence_mapping: List[str]
    risk_flags: List[str]
    confidence: float
    additional_data_needed: List[str]
    citations: List[str]
    orchestration_trace: PromptOrchestrationTrace


def _confidence(retrieved: List[RetrievedEvidence]) -> float:
    if not retrieved:
        return 0.0
    relevance = [r.relevance for r in retrieved]
    avg_rel = sum(relevance) / len(relevance)
    quality_bonus = min(sum(r.record.quality_score for r in retrieved) / (10 * len(retrieved)), 0.2)
    return min(max(avg_rel + quality_bonus, 0.0), 1.0)


def _citation_label(item: RetrievedEvidence) -> str:
    rec = item.record
    id_part = rec.doi_or_pmid or rec.source_id
    return f"{rec.title} ({rec.year}, {rec.journal}) [{id_part}]"


def generate_response(user_query: str, retrieved: List[RetrievedEvidence], trace: PromptOrchestrationTrace) -> AgentResponse:
    score = _confidence(retrieved)
    citations = [_citation_label(item) for item in retrieved]

    if len(retrieved) < 2 or score < 0.22:
        return AgentResponse(
            status="insufficient_evidence",
            interpreted_constraints=f"Query parsed: {user_query}",
            protocol_options=[],
            evidence_mapping=[],
            risk_flags=[
                "Evidence coverage is too sparse to provide reliable protocol guidance.",
                "Any uncited recommendation is blocked by policy.",
            ],
            confidence=score,
            additional_data_needed=[
                "Add approved studies matching nanoparticle platform and ligand class.",
                "Provide protocol-condition details (ratios, pH, buffers, reaction time).",
                "Include outcome metrics for conjugation efficiency and stability.",
            ],
            citations=citations,
            orchestration_trace=trace,
        )

    top = retrieved[:3]
    protocol_options = []
    evidence_mapping = []
    for idx, hit in enumerate(top, start=1):
        rec = hit.record
        protocol_options.append(
            f"Option {idx}: For {rec.nanoparticle_system} with {rec.ligand_class}, evaluate {rec.conjugation_chemistry} under conditions similar to: {rec.protocol_conditions}"
        )
        evidence_mapping.append(
            f"Option {idx} supported by {rec.title} (relevance={hit.relevance:.2f}, quality={rec.quality_score:.2f}); outcomes: {rec.outcomes}"
        )

    return AgentResponse(
        status="ok",
        interpreted_constraints=f"Query parsed: {user_query}",
        protocol_options=protocol_options,
        evidence_mapping=evidence_mapping,
        risk_flags=[
            "Check transferability if model system differs from intended biological context.",
            "Validate ligand density and surface charge impact before scale-up.",
            "Treat low-sample studies as directional evidence only.",
        ],
        confidence=score,
        additional_data_needed=[
            "Negative-result studies for failure mode coverage.",
            "Manufacturing constraints (scale, purification limits).",
        ],
        citations=citations,
        orchestration_trace=trace,
    )
