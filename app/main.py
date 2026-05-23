import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ingestion.loader import fetch_arxiv_papers, download_arxiv_pdf, extract_text_from_pdf
from ingestion.chunker import chunk_document
from ingestion.embedder import embed_texts
from retrieval.vector_store import get_client, create_collection, upsert_chunks, search as vector_search
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank
from generation.generator import generate_answer
from app.schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse, StatusResponse, ChunkResult, PaperInfo


COLLECTION_NAME = "papers"
_papers_metadata: dict[str, PaperInfo] = {}
_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _papers_metadata
    _client = get_client()
    create_collection(_client, COLLECTION_NAME)
    yield
    _client.close()


app = FastAPI(
    title="Research RAG Pipeline",
    description="Ask questions about academic papers using RAG",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_client():
    if _client is None:
        raise RuntimeError("Client not initialized")
    return _client


@app.get("/")
def root():
    return {"message": "Research RAG Pipeline API is running", "docs": "/docs"}


@app.get("/status", response_model=StatusResponse)
def status():
    client = _get_client()
    collection_info = client.get_collection(COLLECTION_NAME)
    chunks_count = collection_info.points_count or 0
    papers = list(_papers_metadata.values())
    return StatusResponse(
        collection=COLLECTION_NAME,
        chunks_count=chunks_count,
        papers_count=len(papers),
        papers=papers,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    global _papers_metadata
    client = _get_client()

    papers = fetch_arxiv_papers(request.arxiv_ids)
    all_chunks = []
    ingested_ids = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for paper in papers:
            pdf_path = download_arxiv_pdf(paper["id"], tmpdir)
            if not pdf_path:
                continue

            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_document(text, paper["id"])
            all_chunks.extend(chunks)
            ingested_ids.append(paper["id"])

            _papers_metadata[paper["id"]] = PaperInfo(
                id=paper["id"],
                title=paper["title"],
                authors=paper["authors"],
                published=paper["published"],
                summary=paper["summary"],
            )

    if all_chunks:
        texts = [c["text"] for c in all_chunks]
        embeddings = embed_texts(texts)
        upsert_chunks(client, COLLECTION_NAME, all_chunks, embeddings)

    return IngestResponse(ingested=len(ingested_ids), papers=ingested_ids)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    client = _get_client()

    query_embedding = embed_texts([request.query])[0]
    dense_results = vector_search(client, COLLECTION_NAME, query_embedding, top_k=request.top_k)

    if not dense_results:
        return QueryResponse(answer="No relevant documents found. Ingest papers first.", sources=[], chunks=[])

    documents = [r["text"] for r in dense_results]
    fused = hybrid_search(request.query, dense_results, documents, alpha=0.5)

    reranked = rerank(request.query, fused, top_k=5)

    generation = generate_answer(request.query, reranked)

    chunks = [
        ChunkResult(
            text=c["text"],
            source=c["source"],
            score=c.get("score", 0),
            rerank_score=c.get("rerank_score"),
            chunk_index=c.get("chunk_index"),
        )
        for c in reranked
    ]

    return QueryResponse(
        answer=generation["answer"],
        sources=generation["sources"],
        chunks=chunks,
    )
