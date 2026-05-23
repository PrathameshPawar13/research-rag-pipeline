from sentence_transformers import SentenceTransformer


_MODEL: SentenceTransformer | None = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    model = get_embedding_model(model_name)
    return model.encode(text).tolist()


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    model = get_embedding_model(model_name)
    return model.encode(texts).tolist()
