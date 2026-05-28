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


OUT_OF_SCOPE_REPLY = (
    "I'm specialized in ligand-nanoparticle conjugation protocols for targeted drug delivery. "
    "I can't help with that topic. If you have a conjugation or nanoparticle formulation question, I'd be happy to help."
)

LIGAND_KEYWORDS = ("transferrin", "antibody", "antibodies", "igg", "peptide", "rgd", "crgd")
NP_KEYWORDS = ("silica", "msn", "liposome", "liposomes", "plga", "nanoparticle", "nanoparticles")
CHEM_KEYWORDS = ("edc", "nhs", "maleimide", "thiol", "pegylation", "peg")
DOMAIN_KEYWORDS = LIGAND_KEYWORDS + NP_KEYWORDS + CHEM_KEYWORDS + ("conjugation", "targeting")


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


def _confidence_label(score: float, source_count: int) -> str:
    if source_count >= 3 and score >= 0.45:
        return "High"
    if source_count >= 1 and score >= 0.22:
        return "Moderate"
    return "Low"


def _detect_ligand(query: str) -> str:
    q = query.lower()
    if "transferrin" in q:
        return "transferrin"
    if "antibody" in q or "antibodies" in q or "igg" in q:
        return "antibodies"
    if "rgd" in q or "crgd" in q or "peptide" in q:
        return "RGD peptide"
    return ""


def _detect_np(query: str) -> str:
    q = query.lower()
    if "silica" in q or "msn" in q:
        return "silica nanoparticles"
    if "liposome" in q:
        return "liposomes"
    if "plga" in q:
        return "PLGA nanoparticles"
    return ""


def _detect_chemistry(query: str) -> str:
    q = query.lower()
    if "maleimide" in q or "thiol" in q:
        return "maleimide-thiol"
    if "edc" in q or "nhs" in q:
        return "EDC/NHS"
    if "pegylation" in q or "peg" in q:
        return "PEGylation strategy"
    return ""


def _recommended_chemistry(ligand: str, np_type: str) -> str:
    if "liposome" in np_type:
        return "maleimide-thiol (DSPE-PEG-Mal post-insertion preferred)"
    if ligand in ("antibodies", "RGD peptide"):
        return "maleimide-thiol for better orientation/activity retention"
    if ligand == "transferrin" and ("silica" in np_type or "plga" in np_type):
        return "EDC/NHS as the most established literature path"
    return "EDC/NHS as a robust baseline, with maleimide-thiol as orientation-preserving alternative"


def _is_out_of_scope(query: str) -> bool:
    q = query.lower()
    return not any(k in q for k in DOMAIN_KEYWORDS)


def generate_response(user_query: str, retrieved: List[RetrievedEvidence], trace: PromptOrchestrationTrace) -> AgentResponse:
    score = _confidence(retrieved)
    citations = [_citation_label(item) for item in retrieved]
    ligand = _detect_ligand(user_query)
    np_type = _detect_np(user_query)
    chemistry = _detect_chemistry(user_query)

    if _is_out_of_scope(user_query):
        return AgentResponse(
            status="out_of_scope",
            interpreted_constraints=OUT_OF_SCOPE_REPLY,
            protocol_options=[],
            evidence_mapping=[],
            risk_flags=[],
            confidence=0.0,
            additional_data_needed=[],
            citations=[],
            orchestration_trace=trace,
        )

    missing = []
    if not ligand:
        missing.append("ligand type (transferrin, antibody, or peptide like RGD/cRGD)")
    if not np_type:
        missing.append("nanoparticle platform (silica, liposome, or PLGA)")

    if missing:
        ask_items = missing[:2]
        question_text = "To build a reliable protocol, I need: " + " and ".join(ask_items) + "."
        suggestion = (
            "If you're targeting BBB delivery with transferrin, a well-characterized default is "
            "EDC/NHS conjugation on silica nanoparticles."
        )
        return AgentResponse(
            status="needs_clarification",
            interpreted_constraints=f"Let me analyze your request... {question_text} {suggestion}",
            protocol_options=[],
            evidence_mapping=[],
            risk_flags=[],
            confidence=0.0,
            additional_data_needed=ask_items,
            citations=[],
            orchestration_trace=trace,
        )

    if len(retrieved) < 2 or score < 0.22:
        return AgentResponse(
            status="insufficient_evidence",
            interpreted_constraints=(
                "My knowledge base does not contain enough verified protocols for this exact combination. "
                "I can provide a constrained starting approach adapted from the closest match, but experimental validation is required."
            ),
            protocol_options=[],
            evidence_mapping=[],
            risk_flags=[
                "Low evidence coverage for this ligand-NP-chemistry combination.",
                "Use pilot-scale optimization before committing critical samples.",
            ],
            confidence=score,
            additional_data_needed=[],
            citations=citations,
            orchestration_trace=trace,
        )

    top = retrieved[:3]
    recommended_chemistry = chemistry or _recommended_chemistry(ligand, np_type)
    confidence_label = _confidence_label(score, len(top))
    protocol_options = []
    evidence_mapping = []
    for idx, hit in enumerate(top, start=1):
        rec = hit.record
        protocol_options.append(
            f"Step {idx} (literature template): {rec.protocol_conditions}"
        )
        evidence_mapping.append(
            f"[{idx}] {rec.title} | Chemistry: {rec.conjugation_chemistry} | Relevance {hit.relevance:.2f} | Outcome: {rec.outcomes}"
        )

    interpreted = (
        f"METHODOLOGY: {ligand} conjugation to {np_type} via {recommended_chemistry}\n"
        f"Confidence Level: {confidence_label}\n"
        f"Knowledge Base Sources Used: {len(top)} papers\n\n"
        "My reasoning for this chemistry selection:\n"
        f"- Ligand analysis: {ligand} and {np_type} are compatible with amide or thiol-reactive coupling.\n"
        f"- Surface chemistry logic: recommended route is {recommended_chemistry} based on dominant literature patterns in the embedded KB.\n"
        "- Tradeoff: EDC/NHS is simpler but less orientation-controlled; maleimide-thiol can preserve activity better when thiolation is feasible."
    )

    return AgentResponse(
        status="ok",
        interpreted_constraints=interpreted,
        protocol_options=protocol_options,
        evidence_mapping=evidence_mapping,
        risk_flags=[
            "Use ranges from multiple sources; avoid locking into single-point conditions on first run.",
            "Watch for overfunctionalization (excess ligand density can reduce receptor engagement).",
            "Verify ligand stability at selected pH and during activation/coupling windows.",
            *[f"Source limitation: {hit.record.limitations}" for hit in top if hit.record.limitations],
        ],
        confidence=score,
        additional_data_needed=[
            "Target tissue/model and administration route",
            "Available surface functionality on your nanoparticle batch (-COOH, -NH2, Maleimide, PEG)",
        ],
        citations=citations,
        orchestration_trace=trace,
    )
