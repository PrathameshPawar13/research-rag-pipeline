from groq import Groq
import os


_CLIENT: Groq | None = None


def get_groq_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        _CLIENT = Groq(api_key=api_key)
    return _CLIENT


def build_context(chunks: list[dict], max_chars: int = 600) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"][:max_chars]
        parts.append(f"[{i}] (Source: {chunk['source']}, Relevance: {chunk.get('rerank_score', chunk.get('score', 0)):.3f})\n{text}")
    return "\n\n".join(parts)


SYSTEM_PROMPT = """You are a research assistant answering questions based on provided document excerpts.
Answer concisely using ONLY the provided context. If the context doesn't contain enough information, say so.
Always cite your sources using bracketed numbers like [1], [2] referencing the numbered chunks above."""


def generate_answer(query: str, chunks: list[dict], model: str = "llama-3.1-8b-instant") -> dict:
    client = get_groq_client()
    context = build_context(chunks)

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {"id": c["source"], "text": c["text"][:200]}
            for c in chunks
        ],
    }
