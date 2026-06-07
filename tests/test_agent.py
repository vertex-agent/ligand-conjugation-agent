from __future__ import annotations

from pathlib import Path

import pytest

from src.app_logic import build_effective_query, load_dataframe_from_upload, run_agent
from src.generator import _detect_chemistry, _detect_ligand, _detect_np
from src.ingestion import approved_peer_reviewed, load_records
from src.retrieval import EvidenceRetriever


ROOT = Path(__file__).resolve().parents[1]


def _load_all_kb() -> tuple:
    dfs = []
    for path in sorted(ROOT.glob("knowledge_base*.csv")):
        dfs.append(load_dataframe_from_upload(path.name, path.read_bytes()))
    import pandas as pd

    merged = pd.concat(dfs, ignore_index=True)
    return merged


def test_decision_matrix_schema_is_normalized():
    path = ROOT / "knowledge_base_chemistry_decision_matrix.csv"
    df = load_dataframe_from_upload(path.name, path.read_bytes())
    assert df.attrs.get("schema_detected") == "decision_matrix"
    assert "evidence_type" in df.columns
    assert (df["evidence_type"] == "decision_matrix").all()
    assert len(df) >= 15


def test_merged_kb_has_many_eligible_records():
    merged = _load_all_kb()
    records = load_records(merged)
    eligible = approved_peer_reviewed(records)
    assert len(eligible) >= 70


def test_retrieval_finds_transferrin_silica():
    merged = _load_all_kb()
    records = approved_peer_reviewed(load_records(merged))
    retriever = EvidenceRetriever(records)
    hits = retriever.retrieve(
        "conjugate transferrin to silica nanoparticles for brain targeting",
        top_k=5,
    )
    assert len(hits) >= 2
    joined = " ".join(hit.record.ligand_class.lower() for hit in hits)
    assert "transferrin" in joined


@pytest.mark.parametrize(
    "query,expected_ligand,expected_np,expected_chem",
    [
        (
            "I want to conjugate transferrin to silica nanoparticles for brain targeting",
            "transferrin",
            "silica nanoparticles",
            "",
        ),
        (
            "What's the best chemistry for attaching antibodies to liposomes?",
            "antibodies",
            "liposomes",
            "",
        ),
        (
            "Help me design an RGD peptide conjugation protocol for PLGA nanoparticles using EDC/NHS",
            "RGD peptide",
            "PLGA nanoparticles",
            "EDC/NHS",
        ),
    ],
)
def test_gold_query_entity_detection(query, expected_ligand, expected_np, expected_chem):
    assert _detect_ligand(query) == expected_ligand
    assert _detect_np(query) == expected_np
    assert _detect_chemistry(query) == expected_chem


def test_effective_query_uses_chat_history():
    history = [
        {"role": "user", "content": "I want brain targeting with transferrin"},
        {"role": "assistant", "content": "Which nanoparticle platform?"},
    ]
    effective = build_effective_query("silica nanoparticles", history)
    assert "transferrin" in effective.lower()
    assert "silica" in effective.lower()


def test_run_agent_abstains_without_entities():
    merged = _load_all_kb()
    response, _ = run_agent("Tell me about quantum computing", merged)
    assert response.status == "out_of_scope"


def test_run_agent_needs_clarification_on_vague_query():
    merged = _load_all_kb()
    response, _ = run_agent("Help me with conjugation", merged)
    assert response.status == "needs_clarification"


def test_run_agent_ok_for_transferrin_silica():
    merged = _load_all_kb()
    response, stats = run_agent(
        "I want to conjugate transferrin to silica nanoparticles for brain targeting",
        merged,
    )
    assert stats["eligible_records"] >= 70
    assert stats["retrieved_records"] >= 2
    assert response.status in {"ok", "insufficient_evidence"}
