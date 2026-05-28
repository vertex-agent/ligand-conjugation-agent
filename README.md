# Ligand-Nanoparticle Conjugation Agent (MVP)

This app implements a domain-specific AI assistant for protocol optimization using **only approved, peer-reviewed literature records** supplied by the user.

## What this MVP does

- Ingests CSV/XLSX literature sheets.
- Enforces a strict evidence gate (`approved == true` and `peer_reviewed == true`).
- Optionally ingests a user-uploaded article (PDF/TXT) **only when explicitly provided**.
- Retrieves relevant records using TF-IDF semantic matching.
- Generates evidence-constrained protocol options with citations.
- Applies built-in orchestration inspired by:
  - Persona assignment + consequence-based prompting
  - Chain-of-Knowledge decomposition
  - RAG retrieval flow
  - ReAct-style retrieve/reason/synthesize loop
  - Hallucination reduction via citation and abstention checks
- Abstains when evidence is insufficient.

## Required input columns

- `source_id`
- `title`
- `year`
- `journal`
- `peer_reviewed`
- `approved`
- `nanoparticle_system`
- `ligand_class`
- `conjugation_chemistry`
- `payload_context`
- `protocol_conditions`
- `outcomes`
- `limitations`

Optional columns:
- `doi_or_pmid`
- `quality_score` (numeric, default `0`)

## Quickstart

1. Create and activate a virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the app:
   - `streamlit run streamlit_app.py`

## Notes

- This MVP is retrieval-grounded and deterministic; it does not call external LLM APIs.
- The agent only retrieves from (a) your uploaded sheet, and (b) an optional uploaded paper if you provide one.
- You can later add an LLM synthesis layer, but keep the same citation and abstention gates.
- Outputs are research-assistive and should be reviewed before wet-lab decisions.
