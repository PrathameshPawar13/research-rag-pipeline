import os
import sys

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
SAMPLE_IDS = ["2302.00093", "2310.06825", "2401.09056", "2307.06435", "2404.07123"]


st.set_page_config(
    page_title="Research RAG Pipeline",
    page_icon="📄",
    layout="wide",
)

st.title("Research RAG Pipeline")
st.write("Ask questions about academic papers. Ingest arXiv papers, then query them using RAG.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Ingest Papers")
    paper_ids = st.text_area(
        "arXiv paper IDs (one per line)",
        value="\n".join(SAMPLE_IDS),
        help="Enter arXiv IDs like '2302.00093' (one per line)",
    )

    if st.button("Ingest Papers", type="primary"):
        ids = [pid.strip() for pid in paper_ids.strip().split("\n") if pid.strip()]
        with st.spinner(f"Ingesting {len(ids)} papers..."):
            try:
                resp = httpx.post(f"{API_BASE}/ingest", json={"arxiv_ids": ids}, timeout=300)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Ingested {data['ingested']} papers: {', '.join(data['papers'])}")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

with col2:
    st.subheader("ℹ️ Status")
    if st.button("Check Status"):
        try:
            resp = httpx.get(f"{API_BASE}/status", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            st.metric("Chunks in Vector DB", data["chunks_count"])
            st.metric("Papers Ingested", data["papers_count"])
            if data["papers"]:
                st.write("**Papers:**")
                for p in data["papers"]:
                    st.write(f"- {p['id']}: {p['title'][:80]}...")
        except Exception as e:
            st.error(f"Status check failed: {e}")

st.divider()

st.subheader("🔍 Query")
query = st.text_input("Ask a question about the ingested papers")

if query:
    with st.spinner("Searching and generating answer..."):
        try:
            resp = httpx.post(
                f"{API_BASE}/query",
                json={"query": query, "top_k": 10},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            st.markdown("### Answer")
            st.write(data["answer"])

            with st.expander("📚 Sources & Chunks"):
                for i, chunk in enumerate(data["chunks"], 1):
                    score = chunk.get("rerank_score", chunk.get("score", 0))
                    st.markdown(f"**Chunk {i}** (Score: {score:.4f})")
                    st.caption(f"Source: {chunk['source']}")
                    st.text(chunk["text"][:500])
                    st.divider()

        except Exception as e:
            st.error(f"Query failed: {e}")
