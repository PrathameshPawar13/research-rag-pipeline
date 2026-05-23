from fastembed import TextEmbedding


_EMBEDDER: TextEmbedding | None = None


def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> TextEmbedding:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = TextEmbedding(model_name=model_name)
    return _EMBEDDER


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()
    return list(model.embed(text))[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return [e.tolist() for e in model.embed(texts)]
