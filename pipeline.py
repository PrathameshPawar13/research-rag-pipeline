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

    def ingest(self, arxiv_ids: list[str]) -> list[str]:
        papers = fetch_arxiv_papers(arxiv_ids)
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
                self._papers[paper["id"]] = paper

        if all_chunks:
            texts = [c["text"] for c in all_chunks]
            embeddings = embed_texts(texts)
            upsert_chunks(self.client, COLLECTION_NAME, all_chunks, embeddings)

        return ingested_ids

    def query(self, query_text: str, top_k: int = 10) -> dict:
        query_embedding = embed_texts([query_text])[0]
        dense_results = vector_search(self.client, COLLECTION_NAME, query_embedding, top_k=top_k)

        if not dense_results:
            return {"answer": "No relevant documents found. Ingest papers first.", "sources": [], "chunks": []}

        documents = [r["text"] for r in dense_results]
        fused = hybrid_search(query_text, dense_results, documents, alpha=0.5)
        reranked = rerank(query_text, fused, top_k=5)
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
