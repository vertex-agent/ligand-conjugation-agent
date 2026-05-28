from __future__ import annotations

import io
from typing import Tuple

import pandas as pd

from .generator import AgentResponse, generate_response
from .ingestion import approved_peer_reviewed, load_records, validate_columns
from .orchestrator import build_orchestration_trace
from .retrieval import EvidenceRetriever


def _normalize_knowledge_base_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept alternate spreadsheet schemas and map them into the app's required columns.
    """
    required = set(validate_columns(pd.DataFrame(columns=[])))
    if required.issubset(set(df.columns)):
        df.attrs["schema_detected"] = "native"
        return df

    # Mapping for user-provided protocol database sheet.
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
        df.attrs["schema_detected"] = "unknown"
        return df

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
    normalized.attrs["schema_detected"] = "protocol_database"
    return normalized


def load_dataframe_from_upload(filename: str, content: bytes) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
        return _normalize_knowledge_base_df(df)
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
        return _normalize_knowledge_base_df(df)
    raise ValueError("Unsupported file format. Use CSV or XLSX.")


def run_agent(
    user_query: str,
    df: pd.DataFrame,
) -> Tuple[AgentResponse, dict]:
    missing = validate_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    all_records = load_records(df)
    eligible = approved_peer_reviewed(all_records)

    retriever = EvidenceRetriever(eligible)
    retrieved = retriever.retrieve(user_query, top_k=8)
    trace = build_orchestration_trace(user_query, retrieved)
    response = generate_response(user_query, retrieved, trace)

    stats = {
        "total_records": len(all_records),
        "eligible_records": len(eligible),
        "retrieved_records": len(retrieved),
    }
    return response, stats
