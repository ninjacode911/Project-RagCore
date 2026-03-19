import logging

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import get_settings
from app.models.document import Chunk
from app.models.schemas import SearchFilters

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self, url: str, api_key: str, collection_name: str):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        logger.info(f"Connected to Qdrant at {url}")

    def ensure_collection(self, vector_size: int = 384) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created collection '{self.collection_name}' (dim={vector_size})")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

        # Ensure payload indexes exist for filterable fields
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Create payload indexes for fields used in filtering."""
        index_fields = {
            "document_id": PayloadSchemaType.KEYWORD,
            "source": PayloadSchemaType.KEYWORD,
            "doc_type": PayloadSchemaType.KEYWORD,
            "tags": PayloadSchemaType.KEYWORD,
            "created_date": PayloadSchemaType.KEYWORD,
        }
        try:
            collection_info = self.client.get_collection(self.collection_name)
            existing_indexes = set(collection_info.payload_schema.keys()) if collection_info.payload_schema else set()
        except Exception:
            existing_indexes = set()

        for field_name, field_type in index_fields.items():
            if field_name not in existing_indexes:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
                    logger.info(f"Created payload index: {field_name} ({field_type})")
                except Exception as e:
                    logger.warning(f"Could not create index for '{field_name}': {e}")

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            points = [
                PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding,
                    payload={
                        "text": chunk.text,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "source": chunk.metadata.source,
                        "doc_type": chunk.metadata.doc_type,
                        "title": chunk.metadata.title,
                        "created_date": chunk.metadata.created_date.isoformat()
                        if chunk.metadata.created_date
                        else None,
                        "tags": chunk.metadata.tags,
                        "page_count": chunk.metadata.page_count,
                    },
                )
                for chunk, embedding in zip(batch_chunks, batch_embeddings)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Upserted {len(chunks)} chunks to '{self.collection_name}'")

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[dict]:
        qdrant_filter = self._build_filter(filters) if filters and filters.has_filters() else None
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
        ).points
        return [
            {
                "chunk_id": str(r.id),
                "text": r.payload.get("text", ""),
                "score": r.score,
                "document_id": r.payload.get("document_id", ""),
                "metadata": {
                    "source": r.payload.get("source", ""),
                    "doc_type": r.payload.get("doc_type", ""),
                    "title": r.payload.get("title"),
                    "created_date": r.payload.get("created_date"),
                    "tags": r.payload.get("tags", []),
                    "page_count": r.payload.get("page_count"),
                },
            }
            for r in results
        ]

    def delete_document(self, document_id: str) -> int:
        # First, find all point IDs belonging to this document
        doc_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )
        point_ids = []
        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=doc_filter,
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.extend([r.id for r in results])
            if next_offset is None:
                break
            offset = next_offset

        if not point_ids:
            logger.warning(f"No points found for document '{document_id}'")
            return 0

        # Delete by point IDs (requires only write permission, not manage)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids),
        )
        logger.info(f"Deleted {len(point_ids)} points for document '{document_id}'")
        return len(point_ids)

    def scroll_all(self, batch_size: int = 100) -> list[dict]:
        all_points = []
        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in results:
                all_points.append({
                    "chunk_id": str(r.id),
                    "text": r.payload.get("text", ""),
                    "document_id": r.payload.get("document_id", ""),
                    "metadata": {
                        "source": r.payload.get("source", ""),
                        "doc_type": r.payload.get("doc_type", ""),
                        "title": r.payload.get("title"),
                        "tags": r.payload.get("tags", []),
                    },
                })
            if next_offset is None:
                break
            offset = next_offset
        return all_points

    def get_document_ids(self) -> list[dict]:
        all_points = self.scroll_all()
        docs: dict[str, dict] = {}
        for p in all_points:
            doc_id = p["document_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "source": p["metadata"]["source"],
                    "title": p["metadata"].get("title"),
                    "doc_type": p["metadata"]["doc_type"],
                    "num_chunks": 0,
                }
            docs[doc_id]["num_chunks"] += 1
        return list(docs.values())

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    @staticmethod
    def _build_filter(filters: SearchFilters) -> Filter | None:
        conditions = []
        if filters.source:
            conditions.append(FieldCondition(key="source", match=MatchValue(value=filters.source)))
        if filters.doc_type:
            conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=filters.doc_type)))
        if filters.tags:
            conditions.append(FieldCondition(key="tags", match=MatchAny(any=filters.tags)))
        if filters.date_from or filters.date_to:
            range_params = {}
            if filters.date_from:
                range_params["gte"] = filters.date_from.isoformat()
            if filters.date_to:
                range_params["lte"] = filters.date_to.isoformat()
            conditions.append(FieldCondition(key="created_date", range=Range(**range_params)))
        return Filter(must=conditions) if conditions else None


_vectorstore: VectorStoreService | None = None


def get_vectorstore() -> VectorStoreService:
    global _vectorstore
    if _vectorstore is None:
        settings = get_settings()
        _vectorstore = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection,
        )
        _vectorstore.ensure_collection(vector_size=settings.embedding_dim)
    return _vectorstore
