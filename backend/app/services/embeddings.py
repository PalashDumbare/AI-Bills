from sentence_transformers import SentenceTransformer

_model = None
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
