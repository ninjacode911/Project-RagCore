from datetime import datetime

from pydantic import BaseModel, Field

from app.models.document import DocumentMetadata


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    message: str


class SearchFilters(BaseModel):
    source: str | None = None
    doc_type: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    tags: list[str] | None = None

    def has_filters(self) -> bool:
        """Return True only if at least one filter field is set."""
        return any([self.source, self.doc_type, self.date_from, self.date_to, self.tags])


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: DocumentMetadata
    rank: int = 0


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: SearchFilters | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    total_results: int
    search_time_ms: float


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    rerank_top_k: int = 5
    filters: SearchFilters | None = None
    stream: bool = False


class GeneratedAnswer(BaseModel):
    query: str
    answer: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    generation_time_ms: float = 0.0
    model: str = ""


class AnalyzedQuery(BaseModel):
    original_query: str
    clean_query: str
    intent: str = "factual"
    extracted_filters: SearchFilters = Field(default_factory=SearchFilters)
    confidence: float = 0.5
