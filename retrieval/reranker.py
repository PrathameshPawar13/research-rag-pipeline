from fastembed.rerank import Reranker


_RERANKER: Reranker | None = None


def get_reranker(model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> Reranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker(model_name=model_name)
    return _RERANKER


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return candidates

    docs = [c["text"] for c in candidates]
    model = get_reranker()
    results = list(model.rerank(query, docs))

    for candidate, result in zip(candidates, results):
        candidate["rerank_score"] = float(result.score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]
