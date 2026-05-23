import tempfile
from qdrant_client import QdrantClient

from ingestion.loader import fetch_arxiv_papers, download_arxiv_pdf, extract_text_from_pdf
from ingestion.chunker import chunk_document
from ingestion.embedder import embed_texts
from retrieval.vector_store import get_client, create_collection, upsert_chunks, search as vector_search
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank
from generation.generator import generate_answer


COLLECTION_NAME = "papers"


class RAGPipeline:
    def __init__(self):
        self.client: QdrantClient = get_client()
        create_collection(self.client, COLLECTION_NAME)
        self._papers: dict[str, dict] = {}

    def ingest(self, arxiv_ids: list[str]) -> dict:
        papers = fetch_arxiv_papers(arxiv_ids)
        all_chunks = []
        ingested_ids = []
        errors: dict[str, str] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            for paper in papers:
                pid = paper["id"]
                try:
                    pdf_path = download_arxiv_pdf(pid, tmpdir)
                    if not pdf_path:
                        errors[pid] = "PDF download failed (arXiv returned non-200)"
                        continue
                    text = extract_text_from_pdf(pdf_path)
                    if not text.strip():
                        errors[pid] = "PDF extracted no text"
                        continue
                    chunks = chunk_document(text, pid)
                    all_chunks.extend(chunks)
                    ingested_ids.append(pid)
                    self._papers[pid] = paper
                except Exception as e:
                    errors[pid] = str(e)

        if all_chunks:
            texts = [c["text"] for c in all_chunks]
            embeddings = embed_texts(texts)
            upsert_chunks(self.client, COLLECTION_NAME, all_chunks, embeddings)

        return {"ingested_ids": ingested_ids, "errors": errors}

    def query(self, query_text: str, top_k: int = 10) -> dict:
        query_embedding = embed_texts([query_text])[0]
        dense_results = vector_search(self.client, COLLECTION_NAME, query_embedding, top_k=top_k)

        if not dense_results:
            return {"answer": "No relevant documents found. Ingest papers first.", "sources": [], "chunks": []}

        documents = [r["text"] for r in dense_results]
        fused = hybrid_search(query_text, dense_results, documents, alpha=0.5)
        reranked = rerank(query_text, fused, top_k=3)
        generation = generate_answer(query_text, reranked)

        return {
            "answer": generation["answer"],
            "sources": generation["sources"],
            "chunks": reranked,
        }

    def status(self) -> dict:
        collection_info = self.client.get_collection(COLLECTION_NAME)
        return {
            "chunks_count": collection_info.points_count or 0,
            "papers_count": len(self._papers),
            "papers": list(self._papers.values()),
        }
