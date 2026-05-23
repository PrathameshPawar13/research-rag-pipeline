import math
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _compute_bm25_scores(query_tokens: list[str], documents: list[str], avg_doc_len: float, k1: float = 1.5, b: float = 0.75) -> list[float]:
    vocab = Counter()
    doc_lens = []
    for doc in documents:
        tokens = _tokenize(doc)
        vocab.update(set(tokens))
        doc_lens.append(len(tokens))

    n_docs = len(documents)
    idf = {}
    for term in set(query_tokens):
        doc_count = sum(1 for doc in documents if term in _tokenize(doc))
        idf[term] = math.log((n_docs - doc_count + 0.5) / (doc_count + 0.5) + 1.0)

    scores = []
    for i, doc in enumerate(documents):
        doc_tokens = _tokenize(doc)
        doc_len = doc_lens[i]
        score = 0.0
        for term in query_tokens:
            if term in doc_tokens:
                tf = doc_tokens.count(term)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
                score += idf.get(term, 0) * numerator / denominator
        scores.append(score)

    return scores


def _normalize(scores: list[float]) -> list[float]:
    if not scores or max(scores) == 0:
        return [0.0] * len(scores)
    max_s = max(scores)
    return [s / max_s for s in scores]


def hybrid_search(
    query: str,
    dense_results: list[dict],
    documents: list[str],
    alpha: float = 0.5,
) -> list[dict]:
    query_tokens = _tokenize(query)
    avg_doc_len = sum(len(_tokenize(d)) for d in documents) / max(len(documents), 1)

    bm25_scores = _compute_bm25_scores(query_tokens, documents, avg_doc_len)
    dense_scores = [r["score"] for r in dense_results]

    bm25_norm = _normalize(bm25_scores)
    dense_norm = _normalize(dense_scores)

    for i, result in enumerate(dense_results):
        combined = alpha * dense_norm[i] + (1 - alpha) * bm25_norm[i]
        result["score"] = combined
        result["bm25_score"] = bm25_scores[i]
        result["dense_score"] = dense_scores[i]

    dense_results.sort(key=lambda x: x["score"], reverse=True)
    return dense_results
