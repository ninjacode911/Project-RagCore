import logging
import re
import time

from rank_bm25 import BM25Okapi

from app.models.document import Chunk

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "not", "no", "if", "then",
    "than", "that", "this", "it", "its", "he", "she", "they", "we", "you",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


class BM25Index:
    def __init__(self):
        self.documents: list[dict] = []
        self.index: BM25Okapi | None = None

    def build_index(self, chunks: list[Chunk]) -> None:
        self.documents = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "tokens": tokenize(chunk.text),
                "metadata": chunk.metadata.model_dump() if chunk.metadata else {},
            }
            for chunk in chunks
        ]
        if self.documents:
            corpus = [doc["tokens"] for doc in self.documents]
            self.index = BM25Okapi(corpus)
        logger.info(f"Built BM25 index with {len(self.documents)} documents")

    def add_documents(self, chunks: list[Chunk]) -> None:
        new_docs = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "tokens": tokenize(chunk.text),
                "metadata": chunk.metadata.model_dump() if chunk.metadata else {},
            }
            for chunk in chunks
        ]
        self.documents.extend(new_docs)
        if self.documents:
            corpus = [doc["tokens"] for doc in self.documents]
            self.index = BM25Okapi(corpus)
        logger.info(f"BM25 index updated: {len(self.documents)} total documents")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.index or not self.documents:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.index.get_scores(tokens)
        scored_docs = [
            (score, doc) for score, doc in zip(scores, self.documents) if score > 0
        ]
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "chunk_id": doc["chunk_id"],
                "document_id": doc["document_id"],
                "text": doc["text"],
                "score": float(score),
                "metadata": doc["metadata"],
            }
            for score, doc in scored_docs[:top_k]
        ]

    def rebuild_from_vectorstore(self, vectorstore) -> None:
        start = time.perf_counter()
        all_points = vectorstore.scroll_all()
        self.documents = [
            {
                "chunk_id": p["chunk_id"],
                "document_id": p["document_id"],
                "text": p["text"],
                "tokens": tokenize(p["text"]),
                "metadata": p["metadata"],
            }
            for p in all_points
            if p.get("text")
        ]
        if self.documents:
            corpus = [doc["tokens"] for doc in self.documents]
            self.index = BM25Okapi(corpus)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"Rebuilt BM25 index from vectorstore: {len(self.documents)} docs in {elapsed:.0f}ms"
        )

    @property
    def doc_count(self) -> int:
        return len(self.documents)


_bm25: BM25Index | None = None


def get_bm25() -> BM25Index:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Index()
    return _bm25
