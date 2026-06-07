from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import DEFAULT_CONFIG, AgentConfig
from .retrieval import EvidenceRetriever, RetrievedEvidence


def _format_hit(hit: RetrievedEvidence, idx: int) -> str:
    rec = hit.record
    return (
        f"[{idx}] {rec.title} | tier={rec.evidence_tier_label} | "
        f"ligand={rec.ligand_class} | NP={rec.nanoparticle_system} | "
        f"chemistry={rec.conjugation_chemistry} | relevance={hit.relevance:.3f} | "
        f"conditions={rec.protocol_conditions[:240]}"
    )


@dataclass
class AgentToolkit:
    retriever: EvidenceRetriever
    config: AgentConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    last_retrieved: List[RetrievedEvidence] = field(default_factory=list)
    executed_steps: List[str] = field(default_factory=list)
    decomposition: List[str] = field(default_factory=list)

    def search_evidence(self, query: str, top_k: int = 8) -> str:
        hits = self.retriever.retrieve(query, top_k=top_k)
        self.last_retrieved = hits
        self.executed_steps.append(f"Act: search_evidence(query={query!r}, top_k={top_k})")
        if not hits:
            self.executed_steps.append("Observe: no matching evidence found.")
            return "No evidence matched the query."
        lines = [_format_hit(hit, idx) for idx, hit in enumerate(hits, start=1)]
        self.executed_steps.append(f"Observe: {len(hits)} records retrieved.")
        return "\n".join(lines)

    def filter_by_compatibility(self, ligand: str, nanoparticle: str) -> str:
        hits = self.retriever.filter_by_compatibility(ligand, nanoparticle)
        self.last_retrieved = hits
        self.executed_steps.append(
            f"Act: filter_by_compatibility(ligand={ligand!r}, nanoparticle={nanoparticle!r})"
        )
        if not hits:
            self.executed_steps.append("Observe: no compatible records found.")
            return f"No compatible records for {ligand} on {nanoparticle}."
        lines = [_format_hit(hit, idx) for idx, hit in enumerate(hits, start=1)]
        self.executed_steps.append(f"Observe: {len(hits)} compatible records found.")
        return "\n".join(lines)

    def get_protocol_parameters(self, chemistry: str) -> str:
        hits = self.retriever.get_protocol_parameters(chemistry)
        self.last_retrieved = hits
        self.executed_steps.append(f"Act: get_protocol_parameters(chemistry={chemistry!r})")
        if not hits:
            self.executed_steps.append("Observe: no protocol parameter records found.")
            return f"No protocol parameters found for {chemistry}."
        lines = []
        for idx, hit in enumerate(hits, start=1):
            rec = hit.record
            lines.append(
                f"[{idx}] chemistry={rec.conjugation_chemistry} | "
                f"pH/conditions={rec.protocol_conditions[:300]} | outcomes={rec.outcomes[:180]}"
            )
        self.executed_steps.append(f"Observe: {len(hits)} parameter records found.")
        return "\n".join(lines)

    def check_evidence_sufficiency(self) -> str:
        count = len(self.last_retrieved)
        avg_rel = (
            sum(hit.relevance for hit in self.last_retrieved) / count if count else 0.0
        )
        sufficient = (
            count >= self.config.min_evidence_count
            and avg_rel >= self.config.min_confidence_score
        )
        self.executed_steps.append("Act: check_evidence_sufficiency()")
        payload = {
            "record_count": count,
            "average_relevance": round(avg_rel, 4),
            "min_required_records": self.config.min_evidence_count,
            "min_required_relevance": self.config.min_confidence_score,
            "sufficient": sufficient,
            "recommendation": (
                "Proceed with grounded synthesis."
                if sufficient
                else "Abstain or ask clarifying questions; evidence is insufficient."
            ),
        }
        self.executed_steps.append(f"Observe: {json.dumps(payload)}")
        return json.dumps(payload)

    def decompose_query(self, user_query: str) -> str:
        steps = [
            f"Target system: identify ligand and nanoparticle in '{user_query}'",
            "Ligand constraints: surface chemistry, orientation, activity retention",
            "Chemistry options: compare EDC/NHS, maleimide-thiol, PEGylation routes",
            "Evaluation outcomes: efficiency, stability, and characterization needs",
        ]
        self.decomposition = steps
        self.executed_steps.append("Reason: decompose request into sub-questions.")
        return json.dumps({"decomposition": steps})

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> str:
        if name == "search_evidence":
            return self.search_evidence(
                query=str(arguments.get("query", "")),
                top_k=int(arguments.get("top_k", 8)),
            )
        if name == "filter_by_compatibility":
            return self.filter_by_compatibility(
                ligand=str(arguments.get("ligand", "")),
                nanoparticle=str(arguments.get("nanoparticle", "")),
            )
        if name == "get_protocol_parameters":
            return self.get_protocol_parameters(chemistry=str(arguments.get("chemistry", "")))
        if name == "check_evidence_sufficiency":
            return self.check_evidence_sufficiency()
        if name == "decompose_query":
            return self.decompose_query(user_query=str(arguments.get("user_query", "")))
        return f"Unknown tool: {name}"


TOOL_DEFINITIONS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "decompose_query",
            "description": "Break the user request into target system, constraints, chemistry options, and outcomes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {"type": "string", "description": "The user's research question."},
                },
                "required": ["user_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_evidence",
            "description": "Search the knowledge base for relevant protocol evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "top_k": {"type": "integer", "description": "Maximum records to return.", "default": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_compatibility",
            "description": "Filter evidence by ligand and nanoparticle compatibility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ligand": {"type": "string"},
                    "nanoparticle": {"type": "string"},
                },
                "required": ["ligand", "nanoparticle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_protocol_parameters",
            "description": "Retrieve protocol parameter ranges for a chemistry route.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chemistry": {"type": "string"},
                },
                "required": ["chemistry"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_evidence_sufficiency",
            "description": "Check whether retrieved evidence meets minimum count and relevance thresholds.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
