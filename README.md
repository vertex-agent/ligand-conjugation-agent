# VERTEX — Ligand-Nanoparticle Conjugation Agent

VERTEX is a retrieval-grounded, tool-using scientific assistant for ligand-functionalized nanoparticle conjugation protocols. It combines an embedded knowledge base, hybrid semantic retrieval, and an OpenAI ReAct agent with citation and abstention gates.

## What VERTEX does

- Loads embedded CSV knowledge bases (protocol database + chemistry decision matrix).
- Normalizes alternate spreadsheet schemas into a unified evidence model.
- Retrieves evidence with **hybrid dense embeddings + TF-IDF** (embeddings optional; TF-IDF fallback when no API key).
- Runs a **ReAct tool loop** (`search_evidence`, `filter_by_compatibility`, `get_protocol_parameters`, `check_evidence_sufficiency`, `decompose_query`).
- Generates grounded protocol recommendations with **evidence tier labels** in citations.
- Supports **multi-turn clarifying dialogue** via chat history.
- Accepts **PDF/TXT uploads** in-session for additional retrievable literature.
- Abstains when evidence count or relevance falls below configured thresholds.

## Knowledge base sources

| Source | Tier | Notes |
|--------|------|-------|
| Protocol database CSV | Peer-reviewed protocol | Literature-backed conditions |
| Chemistry decision matrix CSV | Chemistry decision matrix | Rule-based compatibility guidance |
| User PDF/TXT upload | User-uploaded literature | Session-only, lower default quality score |

## Required native columns

`source_id`, `title`, `year`, `journal`, `peer_reviewed`, `approved`, `nanoparticle_system`, `ligand_class`, `conjugation_chemistry`, `payload_context`, `protocol_conditions`, `outcomes`, `limitations`

Optional: `doi_or_pmid`, `quality_score`, `evidence_type`

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY for ReAct + embeddings
streamlit run streamlit_app.py
```

## Configuration

See [.env.example](.env.example) for:

- `OPENAI_API_KEY` / `OPENAI_MODEL` — ReAct agent synthesis
- `OPENAI_EMBEDDING_MODEL` — hybrid retrieval embeddings
- `VERTEX_EMBEDDINGS` — `auto` (default), `on`, or `off`
- `VERTEX_MIN_EVIDENCE` — abstention threshold (default `2`)
- `VERTEX_REACT_MAX_STEPS` — max tool-calling iterations

## Tests

```bash
pip install pytest
pytest tests/ -q
```

## Architecture

```
streamlit_app.py
  └─ run_agent (app_logic.py)
       ├─ normalize KB schemas (protocol DB + decision matrix)
       ├─ EvidenceRetriever (hybrid TF-IDF + embeddings)
       ├─ AgentToolkit + ReAct loop (generator.py + tools.py)
       └─ AgentResponse with executed orchestration trace
```

Outputs are research-assistive and should be validated experimentally before wet-lab decisions.
