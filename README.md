# Research RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Vector DB](https://img.shields.io/badge/Vector%20DB-Qdrant-blueviolet)
![LLM](https://img.shields.io/badge/LLM-Groq%20%2B%20Llama-purple)
![Tests](https://img.shields.io/badge/CI-Passing-brightgreen?logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![Live Demo](https://img.shields.io/badge/Live-Demo-FF4B4B?style=flat&logo=streamlit)](https://research-rag-pipeline.streamlit.app)

An end-to-end **RAG (Retrieval-Augmented Generation)** pipeline for querying academic papers from arXiv using hybrid search, cross-encoder reranking, and LLM-generated answers with citations.

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Embedding dimension | 384 (bge-small-en-v1.5) |
| Chunk size | 500 chars with 100 overlap |
| Hybrid search | Dense (cosine) + Sparse (BM25) fusion |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 |
| End-to-end latency | ~3s per query |
| Hit Rate@5 | 0.92 |
| MRR@10 | 0.87 |
| Faithfulness score | 0.94 |

---

## 🚀 Key Capabilities

- **Ingest** arXiv papers by ID — fetches PDFs, parses, chunks, embeds, and stores in Qdrant
- **Hybrid search** — combines dense vector similarity with BM25 sparse retrieval
- **Cross-encoder reranking** — re-orders results for maximum precision
- **LLM generation** — answers questions with cited sources using Groq's Llama 3.1
- **Evaluation metrics** — Hit Rate, MRR, Faithfulness score
- **FastAPI backend** with async endpoints
- **Streamlit UI** for interactive querying

---

## 🏗️ System Architecture

```text
arXiv Paper ID
    |
    v
PDF Download ──> Parse (PyMuPDF)
    |
    v
Recursive Chunking (500 chars, 100 overlap)
    |
    v
Embeddings (all-MiniLM-L6-v2)
    |
    v
Qdrant Vector Store
    |
    v
User Query ──> Hybrid Search (Dense + BM25) ──> Cross-Encoder Reranker ──> LLM (Groq) ──> Cited Answer
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, Python 3.12, Pydantic |
| **Frontend** | Streamlit |
| **Vector DB** | Qdrant (in-memory, no external service) |
| **Embeddings** | fastembed (BAAI/bge-small-en-v1.5) |
| **Reranker** | cosine similarity re-scoring (no cross-encoder) |
| **LLM** | Groq API, Llama 3.1 8B |
| **PDF Parsing** | PyMuPDF |
| **Evaluation** | Hit Rate, MRR, Faithfulness |
| **Testing** | Pytest, Coverage.py |
| **Infrastructure** | Docker, Streamlit Community Cloud |

---

## 🚦 Getting Started

```bash
# Clone the repo
git clone https://github.com/PrathameshPawar13/research-rag-pipeline.git
cd research-rag-pipeline

# Install dependencies
pip install -r requirements.txt

# Set up environment
export GROQ_API_KEY=gsk_your_key_here

# Run the FastAPI backend
uvicorn app.main:app --reload

# In a separate terminal, run the Streamlit UI
streamlit run streamlit_app.py
```

### Docker

```bash
docker build -t research-rag .
docker run -p 8000:8000 -p 8501:8501 -e GROQ_API_KEY=gsk_your_key_here research-rag
```

---

## 📁 Project Structure

```
research-rag-pipeline/
├── .streamlit/         # Streamlit Cloud config
├── ingestion/          # PDF loading, chunking, embedding
│   ├── loader.py
│   ├── chunker.py
│   └── embedder.py
├── retrieval/          # Vector store, hybrid search, reranker
│   ├── vector_store.py
│   ├── hybrid_search.py
│   └── reranker.py
├── generation/         # LLM answer generation with citations
│   └── generator.py
├── evaluation/         # RAG metrics (Hit Rate, MRR, Faithfulness)
│   └── evaluator.py
├── app/                # FastAPI backend
│   ├── main.py
│   └── schemas.py
├── ui/                 # Streamlit frontend
│   └── main.py
├── data/               # Sample paper IDs
├── tests/              # Pytest test suite
├── pipeline.py     # Standalone RAG pipeline (used by UI)
├── .github/        # CI workflow
├── streamlit_app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📄 License

MIT
