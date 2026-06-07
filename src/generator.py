from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from .config import DEFAULT_CONFIG, AgentConfig
from .orchestrator import PromptOrchestrationTrace, build_orchestration_trace
from .retrieval import EvidenceRetriever, RetrievedEvidence
from .tools import TOOL_DEFINITIONS, AgentToolkit

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


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
    clarifying_question: str = ""


OUT_OF_SCOPE_REPLY = (
    "I'm specialized in ligand-nanoparticle conjugation protocols for targeted drug delivery. "
    "I can't help with that topic. If you have a conjugation or nanoparticle formulation question, I'd be happy to help."
)

LIGAND_KEYWORDS = ("transferrin", "antibody", "antibodies", "igg", "peptide", "rgd", "crgd", "folate")
NP_KEYWORDS = ("silica", "msn", "liposome", "liposomes", "plga", "nanoparticle", "nanoparticles", "lnp")
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
    return (
        f"[{rec.evidence_tier_label}] {rec.title} ({rec.year}, {rec.journal}) [{id_part}]"
    )


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
    if "folate" in q:
        return "folate"
    if "antibody" in q or "antibodies" in q or "igg" in q:
        return "antibodies"
    if "rgd" in q or "crgd" in q or "peptide" in q:
        return "RGD peptide"
    return ""


def _detect_np(query: str) -> str:
    q = query.lower()
    if "silica" in q or "msn" in q:
        return "silica nanoparticles"
    if "liposome" in q or "lnp" in q:
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


def _format_chat_history(chat_history: Optional[List[dict]]) -> str:
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history[-8:]:
        role = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_json_payload(text: str) -> dict:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def _self_critique(
    payload: dict,
    retrieved: List[RetrievedEvidence],
    config: AgentConfig,
) -> tuple[dict, List[str]]:
    critique_notes: list[str] = []
    count = len(retrieved)
    score = _confidence(retrieved)

    if count < config.min_evidence_count:
        critique_notes.append(
            f"Evidence count {count} below minimum {config.min_evidence_count}; downgrade confidence."
        )
        payload["risk_flags"] = list(payload.get("risk_flags", [])) + [
            f"Only {count} supporting record(s) retrieved; experimental validation required."
        ]
    if score < config.min_confidence_score:
        critique_notes.append(
            f"Average relevance {score:.2f} below threshold {config.min_confidence_score}."
        )

    citations_required = any(
        str(item).strip() for item in payload.get("protocol_options", []) if str(item).strip()
    )
    if citations_required and not retrieved:
        critique_notes.append("Protocol options present without retrieved evidence; clearing options.")
        payload["protocol_options"] = []

    payload["risk_flags"] = list(dict.fromkeys(payload.get("risk_flags", [])))
    return payload, critique_notes


def _generate_with_react_agent(
    user_query: str,
    retrieved: List[RetrievedEvidence],
    trace: PromptOrchestrationTrace,
    score: float,
    retriever: EvidenceRetriever,
    chat_history: Optional[List[dict]] = None,
    config: Optional[AgentConfig] = None,
) -> Optional[AgentResponse]:
    cfg = config or DEFAULT_CONFIG
    if not cfg.openai_api_key or OpenAI is None:
        return None

    toolkit = AgentToolkit(retriever=retriever, config=cfg)
    toolkit.last_retrieved = retrieved
    toolkit.executed_steps.append(f"Reason: parse user request -> {user_query}")
    toolkit.executed_steps.append(f"Observe: {len(retrieved)} records pre-retrieved.")

    history_text = _format_chat_history(chat_history)
    system_prompt = (
        f"{trace.persona_assignment}\n"
        f"{trace.consequence_prompting}\n"
        "You must use the provided tools to gather and verify evidence before answering.\n"
        "Workflow: decompose_query -> search_evidence/filter_by_compatibility -> "
        "get_protocol_parameters -> check_evidence_sufficiency -> final JSON answer.\n"
        "Never invent citations. Only use evidence returned by tools.\n"
        "When finished, respond with JSON only using keys:\n"
        "interpreted_constraints (string),\n"
        "protocol_options (array of strings),\n"
        "evidence_mapping (array of strings),\n"
        "risk_flags (array of strings),\n"
        "additional_data_needed (array of strings),\n"
        "clarifying_question (string, empty if not needed).\n"
        "If evidence is insufficient, keep protocol_options empty and explain in interpreted_constraints."
    )
    user_prompt = (
        f"User query:\n{user_query}\n\n"
        f"Conversation history:\n{history_text or '(none)'}\n\n"
        "Begin by decomposing the query, then retrieve and verify evidence with tools."
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        client = OpenAI(api_key=cfg.openai_api_key)
        for _ in range(cfg.react_max_steps):
            completion = client.chat.completions.create(
                model=cfg.openai_model,
                temperature=0.2,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
            message = completion.choices[0].message
            tool_calls = message.tool_calls or []

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in tool_calls
                        ],
                    }
                )
                for call in tool_calls:
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = toolkit.dispatch(call.function.name, args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        }
                    )
                continue

            payload = _extract_json_payload(message.content or "")
            if payload:
                break
            messages.append(
                {
                    "role": "user",
                    "content": "Provide the final answer as JSON only with the required keys.",
                }
            )
        else:
            final = client.chat.completions.create(
                model=cfg.openai_model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "Summarize your findings as JSON with keys: interpreted_constraints, "
                            "protocol_options, evidence_mapping, risk_flags, additional_data_needed, "
                            "clarifying_question."
                        ),
                    }
                ],
            )
            payload = _extract_json_payload(final.choices[0].message.content or "{}")
    except Exception:
        return None

    active_retrieved = toolkit.last_retrieved or retrieved
    payload, critique_notes = _self_critique(payload, active_retrieved, cfg)
    final_score = _confidence(active_retrieved)
    citations = [_citation_label(item) for item in active_retrieved[:5]]

    updated_trace = build_orchestration_trace(
        user_query,
        active_retrieved,
        executed_steps=toolkit.executed_steps + [f"Critique: {'; '.join(critique_notes)}" if critique_notes else "Critique: passed"],
        decomposition=toolkit.decomposition or trace.chain_of_knowledge_steps,
    )

    clarifying = str(payload.get("clarifying_question", "")).strip()
    if clarifying:
        return AgentResponse(
            status="needs_clarification",
            interpreted_constraints=clarifying,
            protocol_options=[],
            evidence_mapping=[],
            risk_flags=[],
            confidence=0.0,
            additional_data_needed=[clarifying],
            citations=[],
            orchestration_trace=updated_trace,
            clarifying_question=clarifying,
        )

    if len(active_retrieved) < cfg.min_evidence_count or final_score < cfg.min_confidence_score:
        return AgentResponse(
            status="insufficient_evidence",
            interpreted_constraints=str(
                payload.get(
                    "interpreted_constraints",
                    "Evidence count or relevance is below the abstention threshold.",
                )
            ).strip(),
            protocol_options=[],
            evidence_mapping=[str(x) for x in payload.get("evidence_mapping", []) if str(x).strip()],
            risk_flags=list(payload.get("risk_flags", []))
            + critique_notes
            + ["Abstention enforced: fewer than two strong evidence records."],
            confidence=final_score,
            additional_data_needed=[
                str(x) for x in payload.get("additional_data_needed", []) if str(x).strip()
            ],
            citations=citations,
            orchestration_trace=updated_trace,
        )

    return AgentResponse(
        status="ok",
        interpreted_constraints=str(payload.get("interpreted_constraints", "")).strip()
        or "Generated methodology from embedded evidence.",
        protocol_options=[str(x) for x in payload.get("protocol_options", []) if str(x).strip()],
        evidence_mapping=[str(x) for x in payload.get("evidence_mapping", []) if str(x).strip()],
        risk_flags=[str(x) for x in payload.get("risk_flags", []) if str(x).strip()] + critique_notes,
        confidence=final_score,
        additional_data_needed=[
            str(x) for x in payload.get("additional_data_needed", []) if str(x).strip()
        ],
        citations=citations,
        orchestration_trace=updated_trace,
    )


def generate_response(
    user_query: str,
    retrieved: List[RetrievedEvidence],
    trace: PromptOrchestrationTrace,
    chat_history: Optional[List[dict]] = None,
    retriever: Optional[EvidenceRetriever] = None,
    eligible_records: Optional[list] = None,
) -> AgentResponse:
    cfg = DEFAULT_CONFIG
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
        missing.append("ligand type (transferrin, antibody, peptide like RGD/cRGD, or folate)")
    if not np_type:
        missing.append("nanoparticle platform (silica, liposome/LNP, or PLGA)")

    if missing:
        ask_items = missing[:2]
        question_text = "To build a reliable protocol, I need: " + " and ".join(ask_items) + "."
        suggestion = (
            "If you're targeting BBB delivery with transferrin, a well-characterized default is "
            "EDC/NHS conjugation on silica nanoparticles."
        )
        return AgentResponse(
            status="needs_clarification",
            interpreted_constraints=f"{question_text} {suggestion}",
            protocol_options=[],
            evidence_mapping=[],
            risk_flags=[],
            confidence=0.0,
            additional_data_needed=ask_items,
            citations=[],
            orchestration_trace=trace,
            clarifying_question=question_text,
        )

    if len(retrieved) < cfg.min_evidence_count or score < cfg.min_confidence_score:
        if retriever is not None:
            llm_response = _generate_with_react_agent(
                user_query,
                retrieved,
                trace,
                score,
                retriever,
                chat_history=chat_history,
                config=cfg,
            )
            if llm_response is not None and llm_response.status == "needs_clarification":
                return llm_response

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
                f"Retrieved {len(retrieved)} record(s); minimum required is {cfg.min_evidence_count}.",
                "Use pilot-scale optimization before committing critical samples.",
            ],
            confidence=score,
            additional_data_needed=[],
            citations=citations,
            orchestration_trace=trace,
        )

    if retriever is not None:
        llm_response = _generate_with_react_agent(
            user_query,
            retrieved,
            trace,
            score,
            retriever,
            chat_history=chat_history,
            config=cfg,
        )
        if llm_response is not None:
            return llm_response

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
            f"[{idx}] {rec.title} | Chemistry: {rec.conjugation_chemistry} | "
            f"Tier: {rec.evidence_tier_label} | Relevance {hit.relevance:.2f} | Outcome: {rec.outcomes}"
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
