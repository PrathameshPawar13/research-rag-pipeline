import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import RAGPipeline

SAMPLE_IDS = ["2302.00093", "2310.06825", "2401.09056", "2307.06435", "2404.07123"]


@st.cache_resource
def get_pipeline():
    return RAGPipeline()


st.set_page_config(
    page_title="Research RAG Pipeline",
    page_icon="📄",
    layout="wide",
)

st.title("Research RAG Pipeline")
st.write("Ask questions about academic papers. Ingest arXiv papers, then query them using RAG.")

pipeline = get_pipeline()

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
                ingested = pipeline.ingest(ids)
                st.success(f"Ingested {len(ingested)} papers: {', '.join(ingested)}")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

with col2:
    st.subheader("ℹ️ Status")
    if st.button("Check Status"):
        try:
            status = pipeline.status()
            st.metric("Chunks in Vector DB", status["chunks_count"])
            st.metric("Papers Ingested", status["papers_count"])
            if status["papers"]:
                st.write("**Papers:**")
                for p in status["papers"]:
                    st.write(f"- {p['id']}: {p['title'][:80]}...")
        except Exception as e:
            st.error(f"Status check failed: {e}")

st.divider()

st.subheader("🔍 Query")
query = st.text_input("Ask a question about the ingested papers")

if query:
    with st.spinner("Searching and generating answer..."):
        try:
            result = pipeline.query(query, top_k=10)

            st.markdown("### Answer")
            st.write(result["answer"])

            with st.expander("📚 Sources & Chunks"):
                for i, chunk in enumerate(result["chunks"], 1):
                    score = chunk.get("rerank_score", chunk.get("score", 0))
                    st.markdown(f"**Chunk {i}** (Score: {score:.4f})")
                    st.caption(f"Source: {chunk['source']}")
                    st.text(chunk["text"][:500])
                    st.divider()

        except Exception as e:
            st.error(f"Query failed: {e}")
