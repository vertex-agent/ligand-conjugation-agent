from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.app_logic import load_dataframe_from_upload, run_agent
from src.article_upload import article_to_record


st.set_page_config(page_title="VERTEX", layout="wide")
load_dotenv()

KB_DIR = Path(__file__).resolve().parent
EMBEDDED_KB_GLOB = "knowledge_base*.csv"


def load_embedded_kb_dataframe() -> pd.DataFrame:
    kb_paths = sorted(KB_DIR.glob(EMBEDDED_KB_GLOB))
    if not kb_paths:
        raise FileNotFoundError(
            f"No embedded knowledge base CSV files found (expected pattern: {EMBEDDED_KB_GLOB})"
        )

    dfs: list[pd.DataFrame] = []
    schemas: set[str] = set()
    for kb_path in kb_paths:
        df = load_dataframe_from_upload(kb_path.name, kb_path.read_bytes())
        schemas.add(str(df.attrs.get("schema_detected", "unknown")))
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged.attrs["schema_detected"] = ", ".join(sorted(schemas))
    merged.attrs["kb_file_count"] = len(kb_paths)
    return merged

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[dict[str, str]] with keys: role, content
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "uploaded_records" not in st.session_state:
    st.session_state.uploaded_records = []

st.title("VERTEX")
st.markdown(
    """
Hi! I'm VERTEX, your conjugation protocol advisor for targeted drug delivery systems.  
Tell me what you're working on and I'll generate a complete, literature-backed methodology with reagent ranges, step-by-step protocols, and citations.

I work with:  
Ligands: transferrin, antibodies, peptides (RGD, cRGD)  
Nanoparticles: silica, liposomes, PLGA  
Chemistries: EDC/NHS, maleimide-thiol, PEGylation  
You don't need to be specific — just describe what you want to do and I'll ask the right follow-up questions to build your protocol.
"""
)

st.markdown(
    """
<style>
/* --- Professional scientific styling (light, publication-like) --- */

/* --- Typography overrides from browser preview --- */
/* Keep dark text in main area only; sidebar has its own light text theme */
section[data-testid="stMain"] div[data-testid="stMarkdownContainer"] {
  color: rgba(0, 0, 0, 1) !important;
}

/* Change 1: stHeadingWithActionElements (h1 title) */
div[data-testid="stHeadingWithActionElements"] h1,
h1#vertex,
#vertex {
  font-family: "Material Symbols Rounded" !important;
  color: rgba(255, 255, 255, 0.92) !important;
}

/* Change 2: Markdown paragraphs */
div[data-testid="stMarkdownContainer"] p,
.stMarkdown p {
  font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: rgba(200, 192, 192, 0.92) !important;
}

/* Change 3: Markdown bold/strong */
div[data-testid="stMarkdownContainer"] strong,
.stMarkdown strong {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Change 4: Query textarea font */
textarea#text_area_1,
#text_area_1 {
  font-family: "SF Compact Display", sans-serif !important;
  background-color: rgba(230, 230, 230, 0.92) !important;
  color: rgba(99, 99, 99, 0.92) !important;
  background-clip: unset !important;
  -webkit-background-clip: unset !important;
  box-shadow: none !important;
}

/* Persisted preview change: textarea root opacity */
div[data-testid="stTextAreaRootElement"] {
  opacity: 0.6 !important;
}

/* Remove dark wrapper panel/shadow around query textarea */
div[data-testid="stTextArea"] > div,
div[data-testid="stTextArea"] > div:focus-within {
  background-color: transparent !important;
  box-shadow: none !important;
}

/* Change 5: HeadingWithActionElements (h2) */
div[data-testid="stHeadingWithActionElements"] h2 {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: rgba(255, 255, 255, 1) !important;
}
section[data-testid="stSidebar"] div[data-testid="stHeadingWithActionElements"] h2 {
  color: rgba(255, 255, 255, 1) !important;
}

/* Slightly tighter, cleaner typography */
.stApp {
  color: rgba(15, 23, 42, 0.92);
  background: linear-gradient(180deg, #f6f9ff 0%, #f7fafc 40%, #f3f7fb 100%);
}
.stApp p, .stApp li, .stApp label, .stApp div {
  line-height: 1.45;
}

/* Change 6 & 7: stMain section styling */
section[data-testid="stMain"] {
  color: rgba(0, 0, 0, 0.92) !important;
  background-color: rgba(0, 0, 0, 1) !important;
}

/* Title treatment */
div[data-testid="stAppViewContainer"] h1 {
  letter-spacing: 0.12em;
  font-weight: 750;
  margin-bottom: 0.25rem;
}

/* Subtle section headers */
div[data-testid="stAppViewContainer"] h2, 
div[data-testid="stAppViewContainer"] h3 {
  font-weight: 650;
}

/* Result subheaders on dark main panel */
section[data-testid="stMain"] h3#interpreted-constraints,
section[data-testid="stMain"] h3#protocol-options,
section[data-testid="stMain"] h3#evidence-mapping,
section[data-testid="stMain"] h3#risk-flags,
section[data-testid="stMain"] h3#additional-data-needed,
section[data-testid="stMain"] h3#citations {
  color: rgba(255, 255, 255, 1) !important;
}

section[data-testid="stMain"] div[data-testid="stMarkdownContainer"] li {
  color: rgba(249, 246, 246, 1) !important;
}

/* Main content block width + spacing */
section[data-testid="stMain"] > div.block-container {
  padding-top: 4.5rem;
  padding-bottom: 2.25rem;
  max-width: 1100px;
}

/* Give the main container a subtle "panel" look */
section[data-testid="stMain"] > div.block-container > div[data-testid="stVerticalBlock"] {
  background-color: rgba(48, 48, 48, 0.72);
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 16px;
  padding: 18px 18px 10px 18px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
  backdrop-filter: blur(8px);
}

/* Chat message cards (works for both main + sidebar chat_message) */
div[data-testid="stChatMessage"] {
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(248, 250, 252, 0.92);
}

/* Sidebar: cleaner, lab-notebook feel */
section[data-testid="stSidebar"] {
  border-right: 1px solid rgba(148, 163, 184, 0.35);
  background: linear-gradient(180deg, rgba(241, 246, 255, 0.95) 0%, rgba(245, 247, 250, 0.95) 100%);
}
div[data-testid="stSidebarContent"] {
  background-color: rgba(0, 0, 0, 0.95) !important;
  color: rgba(255, 255, 255, 0.92) !important;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
  letter-spacing: 0.04em;
}

/* Persisted preview change: info alert styling in sidebar/main */
div.stAlertContainer {
  color: rgba(255, 255, 255, 1) !important;
  background-color: rgba(255, 255, 255, 0.1) !important;
}

/* Buttons: science-y accent */
div.stButton > button {
  border-radius: 12px !important;
  border: 1px solid rgba(59, 130, 246, 0.35) !important;
}

/* Primary action button: "→" in the query row */
button[data-testid="stBaseButton-primary"] {
  background-color: rgba(233, 234, 241, 0.6) !important;
  border-color: #0b1f3b !important;
  color: rgba(0, 0, 0, 1) !important;
  opacity: 0.6 !important;
}

/* Ensure the arrow text (rendered via Markdown <p>) isn't overridden */
button[data-testid="stBaseButton-primary"] div[data-testid="stMarkdownContainer"] p {
  color: rgba(0, 0, 0, 1) !important;
}

button[data-testid="stBaseButton-primary"]:hover {
  background-color: #09172d !important; /* slightly darker navy */
  border-color: #09172d !important;
}

/* Make Streamlit file uploader look like "Browse files" only (hide dropzone chrome/text) */
section[data-testid="stFileUploaderDropzone"] {
  border: none !important;
  padding: 0 !important;
  background: transparent !important;
}
section[data-testid="stFileUploaderDropzone"] p,
section[data-testid="stFileUploaderDropzone"] small,
section[data-testid="stFileUploaderDropzone"] svg {
  display: none !important;
}
section[data-testid="stFileUploaderDropzone"] button {
  margin: 0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Chat history")
    st.caption("Use the sidebar collapse icon to expand/collapse.")
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            with st.chat_message(role):
                st.markdown(content)
    else:
        st.info("No messages yet.")

    st.divider()
    st.subheader("Knowledge base")
    st.caption("Embedded CSV sources plus optional PDF/TXT uploads for this session.")
    st.info("Answers are generated from the embedded protocol CSV knowledge base and any uploaded papers.")

    uploaded_articles = st.file_uploader(
        "Add literature (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="article_uploader",
    )
    if uploaded_articles:
        new_records = []
        for upload in uploaded_articles:
            record = article_to_record(filename=upload.name, content=upload.getvalue())
            if record is not None:
                new_records.append(record)
        if new_records:
            existing_ids = {r.source_id for r in st.session_state.uploaded_records}
            for record in new_records:
                if record.source_id not in existing_ids:
                    st.session_state.uploaded_records.append(record)
                    existing_ids.add(record.source_id)
            st.success(f"{len(st.session_state.uploaded_records)} uploaded paper(s) indexed for this session.")
    if st.session_state.uploaded_records:
        if st.button("Clear uploaded papers", use_container_width=True):
            st.session_state.uploaded_records = []
            st.rerun()

st.caption("Try a preset question:")
preset_col_1, preset_col_2, preset_col_3 = st.columns(3)
with preset_col_1:
    if st.button(
        "I want to conjugate transferrin to silica nanoparticles for brain targeting",
        use_container_width=True,
        key="preset_q1",
    ):
        st.session_state.query_input = (
            "I want to conjugate transferrin to silica nanoparticles for brain targeting"
        )
with preset_col_2:
    if st.button(
        "Help me design an RGD peptide conjugation protocol for PLGA nanoparticles",
        use_container_width=True,
        key="preset_q2",
    ):
        st.session_state.query_input = (
            "Help me design an RGD peptide conjugation protocol for PLGA nanoparticles"
        )
with preset_col_3:
    if st.button(
        "What's the best chemistry for attaching antibodies to liposomes?",
        use_container_width=True,
        key="preset_q3",
    ):
        st.session_state.query_input = (
            "What's the best chemistry for attaching antibodies to liposomes?"
        )

query_col, button_col = st.columns([0.92, 0.08], vertical_alignment="bottom")
with query_col:
    st.markdown("**How can I help you?**")
    query = st.text_area(
        "How can I help you?",
        key="query_input",
        placeholder="Example: Optimize folate-conjugated LNP protocol for targeted delivery in solid tumor models with improved stability.",
        height=120,
        label_visibility="collapsed",
    )

with button_col:
    run = st.button("→", type="primary", use_container_width=True)

if run:
    if query.strip():
        st.session_state.chat_history.append({"role": "user", "content": query.strip()})
    if not query.strip():
        st.error("Enter a research question.")
    else:
        try:
            try:
                df = load_embedded_kb_dataframe()
            except FileNotFoundError as kb_err:
                msg = str(kb_err)
                st.error(msg)
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
                st.stop()
            schema_detected = df.attrs.get("schema_detected")
            if schema_detected == "protocol_database":
                st.info(
                    "Schema detected: Protocol Database format. "
                    "Columns were auto-mapped to the agent knowledge base schema."
                )
            elif schema_detected == "native":
                st.info("Schema detected: Native evidence schema.")
            else:
                st.info(f"Loaded embedded knowledge base files: {df.attrs.get('kb_file_count', 1)}")
            response, stats = run_agent(
                query.strip(),
                df,
                chat_history=st.session_state.chat_history,
                extra_records=st.session_state.uploaded_records,
            )
            assistant_content = response.interpreted_constraints
            if response.clarifying_question:
                assistant_content = response.clarifying_question
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_content}
            )

            st.caption(
                f"KB coverage: {stats['eligible_records']} eligible records "
                f"({stats['retrieved_records']} retrieved for this query)."
            )

            st.subheader("Interpreted Constraints")
            st.write(response.interpreted_constraints)

            st.subheader("Protocol Options")
            if response.protocol_options:
                for option in response.protocol_options:
                    st.markdown(f"- {option}")
            else:
                st.info("No protocol options generated due to insufficient literature support.")

            st.subheader("Evidence Mapping")
            if response.evidence_mapping:
                for row in response.evidence_mapping:
                    st.markdown(f"- {row}")
            else:
                st.info("No claim-level mapping available because evidence threshold was not met.")

            st.subheader("Risk Flags")
            for item in response.risk_flags:
                st.markdown(f"- {item}")

            st.subheader("Additional Data Needed")
            for item in response.additional_data_needed:
                st.markdown(f"- {item}")

            st.subheader("Citations")
            if response.citations:
                for citation in response.citations:
                    st.markdown(f"- {citation}")
            else:
                st.warning("No eligible citations found in provided dataset.")

            if response.status != "ok":
                if response.status == "needs_clarification":
                    st.warning("VERTEX needs a bit more detail — reply in the text box to continue.")
                elif response.status == "out_of_scope":
                    st.warning("This question is outside VERTEX's conjugation domain.")
                else:
                    st.error("Insufficient evidence in the knowledge base for this combination.")

            with st.expander("Agent trace (ReAct + chain-of-knowledge)"):
                trace = response.orchestration_trace
                st.write("**Persona**")
                st.write(trace.persona_assignment)
                st.write("**Chain-of-knowledge**")
                for step in trace.chain_of_knowledge_steps:
                    st.markdown(f"- {step}")
                st.write("**Executed ReAct steps**")
                for step in trace.react_steps:
                    st.markdown(f"- {step}")
                st.write("**Hallucination controls**")
                for step in trace.hallucination_controls:
                    st.markdown(f"- {step}")
        except Exception as exc:
            st.exception(exc)
