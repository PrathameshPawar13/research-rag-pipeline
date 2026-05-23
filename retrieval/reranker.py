from sentence_transformers import CrossEncoder


_RERANKER: CrossEncoder | None = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(model_name)
    return _RERANKER


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return candidates

    pairs = [(query, c["text"]) for c in candidates]
    model = get_reranker()
    scores = model.predict(pairs).tolist()

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]
