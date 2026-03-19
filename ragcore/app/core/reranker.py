import logging
import time

from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self):
        start = time.perf_counter()
        from flashrank import Ranker

        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Loaded FlashRank reranker in {elapsed:.0f}ms")

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        from flashrank import RerankRequest

        passages = [{"id": chunk.chunk_id, "text": chunk.text} for chunk in chunks]
        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        # Map reranked scores back to chunks
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        reranked = []
        for i, result in enumerate(results[:top_k]):
            chunk_id = result["id"]
            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id].model_copy()
                chunk.score = float(result["score"])
                chunk.rank = i
                reranked.append(chunk)

        logger.info(f"Reranked {len(chunks)} → top {len(reranked)} chunks")
        return reranked


_reranker: RerankerService | None = None


def get_reranker() -> RerankerService:
    global _reranker
    if _reranker is None:
        _reranker = RerankerService()
    return _reranker
