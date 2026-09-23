"""Minimal Streamlit UI for the live RAG API (ingest + ask).

The FastAPI service is the source of truth. This page only POSTs to /ingest and /ask.

Run:
  cd ai-engineering-bootcamp-v2/week-1
  source .venv/bin/activate
  streamlit run rag_demo.py
"""

import os

import httpx
import streamlit as st

DEFAULT_API = os.getenv("API_BASE_URL", "https://ai-internship-hjip.onrender.com")

st.set_page_config(page_title="Northwind RAG demo", layout="centered")
st.title("Northwind RAG")
st.caption("Calls your FastAPI /ingest and /ask. Retrieval and generation stay on the server.")

base_url = st.sidebar.text_input("API base URL", DEFAULT_API).rstrip("/")
st.sidebar.markdown("Set `API_BASE_URL` in the environment to change the default. No API keys in this app.")
st.sidebar.code(
    "cd ai-engineering-bootcamp-v2/week-1\n"
    "source .venv/bin/activate\n"
    "streamlit run rag_demo.py",
    language="bash",
)


def post_json(path: str, payload: dict) -> tuple[int, dict | str]:
    try:
        response = httpx.post(f"{base_url}{path}", json=payload, timeout=120.0)
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, response.text
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


ingest_tab, ask_tab = st.tabs(["Ingest document", "Ask"])

with ingest_tab:
    st.subheader("POST /ingest")
    document_id = st.text_input("document_id", "handbook")
    source = st.text_input("source (optional)", "pasted.txt")
    text = st.text_area("text", height=200, placeholder="Paste document text here")
    if st.button("Ingest", type="primary"):
        status, data = post_json(
            "/ingest",
            {
                "document_id": document_id,
                "source": source,
                "text": text,
            },
        )
        st.markdown(f"**HTTP {status}**" if status else "**Request failed**")
        st.json(data)

with ask_tab:
    st.subheader("POST /ask")
    question = st.text_area(
        "question",
        "Where is Northwind headquarters?",
        height=80,
    )
    model = st.selectbox("model", ["gpt-4o-mini", "gpt-4o"], index=0)
    if st.button("Ask", type="primary"):
        status, data = post_json(
            "/ask",
            {"question": question, "model": model},
        )
        st.markdown(f"**HTTP {status}**" if status else "**Request failed**")
        if not isinstance(data, dict) or "answer" not in data:
            st.json(data)
        else:
            inner = data.get("answer") or {}
            answer_text = inner.get("answer", "")
            sources_needed = inner.get("sources_needed")
            chunk_ids = data.get("retrieved_chunk_ids") or []

            if sources_needed:
                st.warning("Refusal: the API could not answer from the retrieved documents.")
            else:
                st.success("Answer grounded in retrieved documents.")

            st.markdown("### Answer")
            st.write(answer_text)

            st.markdown("### Citations (retrieved chunk IDs)")
            if chunk_ids:
                st.code("\n".join(str(cid) for cid in chunk_ids))
            else:
                st.write("None")

            cols = st.columns(4)
            cols[0].metric("Confidence", str(inner.get("confidence", "-")))
            cols[1].metric("Tokens", str(data.get("tokens_used", "-")))
            cols[2].metric("Cost USD", str(data.get("cost_usd", "-")))
            cols[3].metric("Model", str(data.get("model", "-")))

            with st.expander("Raw JSON"):
                st.json(data)
