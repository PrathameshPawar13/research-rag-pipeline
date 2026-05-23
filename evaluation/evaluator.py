def hit_rate(retrieved: list[list[str]], relevant: list[str], k: int = 5) -> float:
    hits = 0
    for retrieved_docs, relevant_doc in zip(retrieved, relevant):
        if relevant_doc in retrieved_docs[:k]:
            hits += 1
    return hits / max(len(retrieved), 1)


def mean_reciprocal_rank(retrieved: list[list[str]], relevant: list[str], k: int = 10) -> float:
    reciprocal_ranks = []
    for retrieved_docs, relevant_doc in zip(retrieved, relevant):
        for rank, doc_id in enumerate(retrieved_docs[:k], 1):
            if doc_id == relevant_doc:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1)


def faithfulness_score(answer: str, context_chunks: list[str]) -> float:
    answer_lower = answer.lower()
    total_claims = 0
    supported_claims = 0

    sentences = [s.strip() for s in answer.replace("?", ".").replace("!", ".").split(".") if s.strip()]

    for sentence in sentences:
        if len(sentence) < 15:
            continue
        total_claims += 1
        words = set(sentence.lower().split())
        overlap = False
        for context in context_chunks:
            context_lower = context.lower()
            shared = words & set(context_lower.split())
            if len(shared) >= 3:
                overlap = True
                break
        if overlap:
            supported_claims += 1

    return supported_claims / max(total_claims, 1)
