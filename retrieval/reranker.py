import numpy as np

from ingestion.embedder import embed_texts


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return candidates

    query_emb = np.array(embed_texts([query])[0])
    texts = [c["text"] for c in candidates]
    doc_embs = np.array(embed_texts(texts))

    query_norm = np.linalg.norm(query_emb)
    doc_norms = np.linalg.norm(doc_embs, axis=1)

    similarities = np.dot(doc_embs, query_emb) / (doc_norms * query_norm + 1e-8)

    for candidate, sim in zip(candidates, similarities):
        hybrid = candidate.get("score", 0)
        candidate["rerank_score"] = float((hybrid + sim) / 2)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]
