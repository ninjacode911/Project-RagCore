import logging
import time
from collections import defaultdict

from app.core.bm25 import BM25Index
from app.core.embedder import EmbedderService
from app.core.vectorstore import VectorStoreService
from app.models.document import DocumentMetadata
from app.models.schemas import RetrievedChunk, SearchFilters

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        vectorstore: VectorStoreService,
        bm25: BM25Index,
        embedder: EmbedderService,
    ):
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> list[RetrievedChunk]:
        start = time.perf_counter()

        query_vector = self.embedder.embed_query(query)

        # Dense search via Qdrant (over-fetch 2x)
        dense_results = self.vectorstore.search(
            query_vector=query_vector,
            limit=top_k * 2,
            filters=filters,
        )

        # Sparse search via BM25
        sparse_results = self.bm25.search(query, top_k=top_k * 2)

        # Post-filter BM25 results if filters are provided
        if filters and filters.has_filters():
            sparse_results = self._apply_filters(sparse_results, filters)

        # RRF fusion
        fused = self.rrf_fuse(
            [dense_results, sparse_results],
            weights=[dense_weight, sparse_weight],
        )

        # Deduplicate by chunk_id and take top_k
        seen = set()
        unique = []
        for item in fused:
            if item["chunk_id"] not in seen:
                seen.add(item["chunk_id"])
                unique.append(item)
            if len(unique) >= top_k:
                break

        # Convert to RetrievedChunk models
        results = [
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                document_id=item.get("document_id", ""),
                text=item["text"],
                score=item["fused_score"],
                metadata=DocumentMetadata(**item.get("metadata", {})),
                rank=i,
            )
            for i, item in enumerate(unique)
        ]

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"Hybrid retrieval: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(results)} results in {elapsed:.0f}ms"
        )
        return results

    @staticmethod
    def rrf_fuse(
        result_lists: list[list[dict]],
        k: int = 60,
        weights: list[float] | None = None,
    ) -> list[dict]:
        if weights is None:
            weights = [1.0] * len(result_lists)

        scores: dict[str, float] = defaultdict(float)
        docs: dict[str, dict] = {}

        for result_list, weight in zip(result_lists, weights):
            for rank, item in enumerate(result_list):
                chunk_id = item["chunk_id"]
                scores[chunk_id] += weight * (1.0 / (k + rank))
                if chunk_id not in docs:
                    docs[chunk_id] = item

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {**docs[chunk_id], "fused_score": score}
            for chunk_id, score in ranked
        ]

    @staticmethod
    def _apply_filters(results: list[dict], filters: SearchFilters) -> list[dict]:
        filtered = []
        for r in results:
            meta = r.get("metadata", {})
            if filters.source and meta.get("source") != filters.source:
                continue
            if filters.doc_type and meta.get("doc_type") != filters.doc_type:
                continue
            if filters.tags:
                doc_tags = meta.get("tags", [])
                if not any(t in doc_tags for t in filters.tags):
                    continue
            filtered.append(r)
        return filtered
