from datetime import datetime

from pydantic import BaseModel, Field

from app.utils.helpers import generate_id


class DocumentMetadata(BaseModel):
    source: str = ""
    doc_type: str = ""
    title: str | None = None
    created_date: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    page_count: int | None = None


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=generate_id)
    document_id: str = ""
    text: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0


class Document(BaseModel):
    document_id: str = Field(default_factory=generate_id)
    filename: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    chunks: list[Chunk] = Field(default_factory=list)
    raw_text: str = ""
