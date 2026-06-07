from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .retrieval import RetrievedEvidence


@dataclass
class PromptOrchestrationTrace:
    persona_assignment: str
    consequence_prompting: str
    chain_of_knowledge_steps: List[str]
    react_steps: List[str]
    hallucination_controls: List[str]


def build_orchestration_trace(
    user_query: str,
    evidence: List[RetrievedEvidence],
    executed_steps: Optional[List[str]] = None,
    decomposition: Optional[List[str]] = None,
) -> PromptOrchestrationTrace:
    chain_steps = decomposition or [
        "Decompose request into target system, ligand constraints, chemistry options, and evaluation outcomes.",
        "Map each sub-question to strongest available approved evidence.",
        "Reject sub-answers without citation support.",
        "Synthesize only supported recommendations with confidence and caveats.",
    ]
    react_steps = executed_steps or [
        f"Reason: parse user request -> {user_query}",
        "Act: retrieve top approved/peer-reviewed evidence.",
        f"Observe: {len(evidence)} relevant records found.",
        "Reason: assess coverage gaps and whether abstention is required.",
        "Act: construct final recommendation blocks with citations and uncertainty.",
    ]
    return PromptOrchestrationTrace(
        persona_assignment=(
            "You are a conservative scientific protocol analyst for ligand-nanoparticle conjugation. "
            "Never generate uncited protocol claims."
        ),
        consequence_prompting=(
            "Consequence framing: uncited or speculative guidance can trigger failed experiments and safety risk; "
            "abstain when support is incomplete."
        ),
        chain_of_knowledge_steps=chain_steps,
        react_steps=react_steps,
        hallucination_controls=[
            "Every recommendation must cite at least one approved source.",
            "If evidence count < 2 or relevance is weak, return constrained/abstention response.",
            "No external facts beyond ingested corpus.",
        ],
    )
