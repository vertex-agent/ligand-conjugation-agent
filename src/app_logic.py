from __future__ import annotations

import io
from typing import Optional, Tuple

import pandas as pd

from .article_upload import article_to_record
from .generator import AgentResponse, generate_response
from .ingestion import approved_peer_reviewed, load_records, validate_columns
from .orchestrator import build_orchestration_trace
from .retrieval import EvidenceRetriever


def load_dataframe_from_upload(filename: str, content: bytes) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    raise ValueError("Unsupported file format. Use CSV or XLSX.")


def run_agent(
    user_query: str,
    df: pd.DataFrame,
    *,
    uploaded_article_filename: Optional[str] = None,
    uploaded_article_content: Optional[bytes] = None,
) -> Tuple[AgentResponse, dict]:
    missing = validate_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    all_records = load_records(df)
    eligible = approved_peer_reviewed(all_records)

    if uploaded_article_filename and uploaded_article_content:
        extra = article_to_record(filename=uploaded_article_filename, content=uploaded_article_content)
        if extra is not None:
            eligible = [*eligible, extra]

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
