from __future__ import annotations

import streamlit as st

from src.app_logic import load_dataframe_from_upload, run_agent


st.set_page_config(page_title="VERTEX", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[dict[str, str]] with keys: role, content

st.title("VERTEX")
st.markdown(
    "Vertex is a specialized scientific AI agent for **literature-grounded protocol reasoning** in ligand–nanoparticle "
    "conjugation workflows. It surfaces actionable options, maps each option back to supporting evidence, flags risks and "
    "limitations, and abstains when evidence is insufficient."
)

st.markdown(
    """
<style>
/* --- Professional scientific styling (light, publication-like) --- */

/* --- Typography overrides from browser preview --- */
/* Change 1: stMarkdownContainer text color */
div[data-testid="stMarkdownContainer"] {
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

/* Main content block width + spacing */
section[data-testid="stMain"] > div.block-container {
  padding-top: 3.0rem;
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

upload = None
article = None

query_col, button_col = st.columns([0.92, 0.08], vertical_alignment="bottom")
with query_col:
    st.markdown("**How can I help you?**")
    query = st.text_area(
        "How can I help you?",
        placeholder="Example: Optimize folate-conjugated LNP protocol for targeted delivery in solid tumor models with improved stability.",
        height=120,
        label_visibility="collapsed",
    )

with button_col:
    run = st.button("→", type="primary", use_container_width=True)

if run:
    if query.strip():
        st.session_state.chat_history.append({"role": "user", "content": query.strip()})
    if not upload:
        msg = "No knowledge base is connected (Input Data section removed)."
        st.error(msg)
        st.session_state.chat_history.append({"role": "assistant", "content": msg})
    elif not query.strip():
        st.error("Enter a research question.")
    else:
        try:
            df = load_dataframe_from_upload(upload.name, upload.getvalue())
            response, stats = run_agent(
                query.strip(),
                df,
                uploaded_article_filename=(article.name if article else None),
                uploaded_article_content=(article.getvalue() if article else None),
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response.interpreted_constraints}
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Total records", stats["total_records"])
            c2.metric("Eligible records", stats["eligible_records"])
            c3.metric("Retrieved records", stats["retrieved_records"])

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

            st.subheader("Confidence")
            st.progress(float(response.confidence))
            st.write(f"Confidence score: **{response.confidence:.2f}**")

            with st.expander("Prompt Engineering Orchestration (Internal Trace)"):
                st.write("**Persona assignment**")
                st.write(response.orchestration_trace.persona_assignment)
                st.write("**Consequence-based prompting**")
                st.write(response.orchestration_trace.consequence_prompting)
                st.write("**Chain-of-Knowledge steps**")
                for step in response.orchestration_trace.chain_of_knowledge_steps:
                    st.markdown(f"- {step}")
                st.write("**ReAct loop**")
                for step in response.orchestration_trace.react_steps:
                    st.markdown(f"- {step}")
                st.write("**Hallucination controls**")
                for step in response.orchestration_trace.hallucination_controls:
                    st.markdown(f"- {step}")

            if response.status != "ok":
                st.error(
                    "Insufficient evidence for reliable protocol recommendation. "
                    "Upload additional approved, peer-reviewed records."
                )
        except Exception as exc:
            st.exception(exc)
