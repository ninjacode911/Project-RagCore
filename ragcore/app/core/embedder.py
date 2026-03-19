import logging
import time

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbedderService:
    EMBEDDING_DIM = 384

    def __init__(self, model_name: str):
        start = time.perf_counter()
        self.model = SentenceTransformer(model_name, device="cpu")
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Loaded embedding model '{model_name}' in {elapsed:.0f}ms")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


_embedder: EmbedderService | None = None


def get_embedder() -> EmbedderService:
    global _embedder
    if _embedder is None:
        settings = get_settings()
        _embedder = EmbedderService(settings.embedding_model)
    return _embedder
