from __future__ import annotations

import io
from typing import List, Optional, Tuple

import pandas as pd

from .generator import AgentResponse, generate_response
from .ingestion import approved_peer_reviewed, load_records, validate_columns
from .models import EvidenceRecord
from .orchestrator import build_orchestration_trace
from .retrieval import EvidenceRetriever


def _normalize_decision_matrix_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map chemistry decision matrix columns into the native evidence schema."""
    matrix_columns = {
        "ligand_type": "Ligand Type",
        "np_type": "NP Type",
        "recommended_chemistry": "Recommended Chemistry",
        "why": "Why This Chemistry",
        "alternative": "Alternative Chemistry",
        "key_constraint": "Key Constraint",
        "ph_range": "Critical pH Range",
        "efficiency": "Typical Efficiency Range (%)",
        "supporting": "Supporting Entries",
        "references": "Key References (author, year)",
        "notes": "Key Finding / Protocol Notes",
    }
    if not all(col in df.columns for col in matrix_columns.values()):
        return pd.DataFrame()

    normalized = pd.DataFrame()
    normalized["source_id"] = [
        f"matrix:{idx + 1}" for idx in range(len(df))
    ]
    ligand = df[matrix_columns["ligand_type"]].fillna("").astype(str).str.strip()
    np_type = df[matrix_columns["np_type"]].fillna("").astype(str).str.strip()
    chemistry = df[matrix_columns["recommended_chemistry"]].fillna("").astype(str).str.strip()
    normalized["title"] = (
        ligand + " on " + np_type + " via " + chemistry
    ).str.strip()
    normalized["year"] = "matrix"
    normalized["journal"] = "Chemistry Decision Matrix"
    normalized["peer_reviewed"] = True
    normalized["approved"] = True
    normalized["nanoparticle_system"] = np_type
    normalized["ligand_class"] = ligand
    normalized["conjugation_chemistry"] = chemistry
    normalized["payload_context"] = df[matrix_columns["supporting"]].fillna("").astype(str).str.strip()

    why = df[matrix_columns["why"]].fillna("").astype(str).str.strip()
    alt = df[matrix_columns["alternative"]].fillna("").astype(str).str.strip()
    constraint = df[matrix_columns["key_constraint"]].fillna("").astype(str).str.strip()
    ph = df[matrix_columns["ph_range"]].fillna("").astype(str).str.strip()
    eff = df[matrix_columns["efficiency"]].fillna("").astype(str).str.strip()
    notes = df[matrix_columns["notes"]].fillna("").astype(str).str.strip()
    refs = df[matrix_columns["references"]].fillna("").astype(str).str.strip()

    normalized["protocol_conditions"] = (
        "Critical pH: "
        + ph
        + " | Typical efficiency: "
        + eff
        + " | Key constraint: "
        + constraint
        + " | Rationale: "
        + why
        + " | Alternative: "
        + alt
        + " | Notes: "
        + notes
    )
    normalized["outcomes"] = notes
    normalized["limitations"] = (
        "Decision-matrix guidance; validate experimentally. References: " + refs
    ).str.strip()
    normalized["doi_or_pmid"] = refs
    normalized["quality_score"] = 0.55
    normalized["evidence_type"] = "decision_matrix"
    normalized.attrs["schema_detected"] = "decision_matrix"
    return normalized


def _normalize_protocol_database_df(df: pd.DataFrame) -> pd.DataFrame:
    source_columns = {
        "entry": "Entry #",
        "ligand_type": "Ligand Type",
        "nanoparticle_type": "Nanoparticle Type",
        "conjugation_chemistry": "Conjugation Chemistry",
        "notes": "Key Finding / Notes",
        "first_author": "First Author",
        "year": "Year",
        "journal": "Journal",
        "doi": "DOI",
    }
    if not all(col in df.columns for col in source_columns.values()):
        return pd.DataFrame()

    normalized = pd.DataFrame()
    normalized["source_id"] = df[source_columns["entry"]].apply(lambda x: f"kb:{str(x).strip()}")
    normalized["title"] = (
        df[source_columns["first_author"]].fillna("").astype(str).str.strip()
        + " "
        + df[source_columns["year"]].fillna("").astype(str).str.strip()
        + " "
        + df[source_columns["journal"]].fillna("").astype(str).str.strip()
    ).str.strip()
    normalized["year"] = df[source_columns["year"]].fillna("").astype(str).str.strip()
    normalized["journal"] = df[source_columns["journal"]].fillna("").astype(str).str.strip()
    normalized["peer_reviewed"] = True
    normalized["approved"] = True
    normalized["nanoparticle_system"] = df[source_columns["nanoparticle_type"]].fillna("").astype(str).str.strip()
    normalized["ligand_class"] = df[source_columns["ligand_type"]].fillna("").astype(str).str.strip()
    normalized["conjugation_chemistry"] = (
        df[source_columns["conjugation_chemistry"]].fillna("").astype(str).str.strip()
    )
    normalized["payload_context"] = ""
    ratio_col = "Molar Ratio (Ligand:NP)"
    ph_col = "pH"
    buffer_col = "Buffer"
    temp_col = "Temperature (\u00b0C)"
    time_col = "Incubation Time (h)"
    eff_col = "Conjugation Efficiency (%)"
    char_col = "Characterization Method"

    def _opt(col_name: str) -> pd.Series:
        if col_name in df.columns:
            return df[col_name].fillna("").astype(str).str.strip()
        return pd.Series([""] * len(df))

    ratio = _opt(ratio_col)
    ph = _opt(ph_col)
    buffer = _opt(buffer_col)
    temp = _opt(temp_col)
    inc_time = _opt(time_col)
    eff = _opt(eff_col)
    char = _opt(char_col)
    notes = df[source_columns["notes"]].fillna("").astype(str).str.strip()

    normalized["protocol_conditions"] = (
        "Molar ratio: "
        + ratio
        + " | pH: "
        + ph
        + " | Buffer: "
        + buffer
        + " | Temperature: "
        + temp
        + " | Incubation: "
        + inc_time
        + " | Conjugation efficiency: "
        + eff
        + " | Notes: "
        + notes
    )
    normalized["outcomes"] = df[source_columns["notes"]].fillna("").astype(str).str.strip()
    normalized["limitations"] = ("Characterization used: " + char).str.strip()
    normalized["doi_or_pmid"] = df[source_columns["doi"]].fillna("").astype(str).str.strip()
    normalized["quality_score"] = 0.75
    normalized["evidence_type"] = "peer_reviewed_protocol"
    normalized.attrs["schema_detected"] = "protocol_database"
    return normalized


def _normalize_knowledge_base_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept alternate spreadsheet schemas and map them into the app's required columns.
    """
    required = set(validate_columns(pd.DataFrame(columns=[])))
    if required.issubset(set(df.columns)):
        if "evidence_type" not in df.columns:
            df = df.copy()
            df["evidence_type"] = "peer_reviewed_protocol"
        df.attrs["schema_detected"] = "native"
        return df

    matrix_df = _normalize_decision_matrix_df(df)
    if not matrix_df.empty:
        return matrix_df

    protocol_df = _normalize_protocol_database_df(df)
    if not protocol_df.empty:
        return protocol_df

    df.attrs["schema_detected"] = "unknown"
    return df


def load_dataframe_from_upload(filename: str, content: bytes) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
        return _normalize_knowledge_base_df(df)
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
        return _normalize_knowledge_base_df(df)
    raise ValueError("Unsupported file format. Use CSV or XLSX.")


def build_effective_query(user_query: str, chat_history: Optional[List[dict]] = None) -> str:
    """Combine recent user turns so follow-up clarifications inherit prior context."""
    if not chat_history:
        return user_query.strip()
    user_turns = [
        str(msg.get("content", "")).strip()
        for msg in chat_history
        if msg.get("role") == "user" and str(msg.get("content", "")).strip()
    ]
    if not user_turns:
        return user_query.strip()
    if user_query.strip() in user_turns:
        user_turns = user_turns[:-1]
    combined = " ".join(user_turns[-3:] + [user_query.strip()])
    return combined.strip()


def run_agent(
    user_query: str,
    df: pd.DataFrame,
    chat_history: Optional[List[dict]] = None,
    extra_records: Optional[List[EvidenceRecord]] = None,
) -> Tuple[AgentResponse, dict]:
    missing = validate_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    all_records = load_records(df)
    if extra_records:
        all_records = all_records + list(extra_records)
    eligible = approved_peer_reviewed(all_records)

    effective_query = build_effective_query(user_query, chat_history)
    retriever = EvidenceRetriever(eligible)
    retrieved = retriever.retrieve(effective_query, top_k=8)
    trace = build_orchestration_trace(effective_query, retrieved)
    response = generate_response(
        effective_query,
        retrieved,
        trace,
        chat_history=chat_history,
        retriever=retriever,
        eligible_records=eligible,
    )

    stats = {
        "total_records": len(all_records),
        "eligible_records": len(eligible),
        "retrieved_records": len(retrieved),
    }
    return response, stats
