import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.deps import dep_bm25, dep_embedder, dep_vectorstore
from app.config import get_settings
from app.core.bm25 import BM25Index
from app.core.chunker import chunk_text
from app.core.embedder import EmbedderService
from app.core.metadata import extract_metadata
from app.core.vectorstore import VectorStoreService
from app.models.document import Chunk, Document, DocumentMetadata
from app.models.schemas import IngestResponse
from app.utils.helpers import generate_id
from app.utils.parsers import SUPPORTED_EXTENSIONS, get_page_count, parse_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile,
    vectorstore: VectorStoreService = Depends(dep_vectorstore),
    embedder: EmbedderService = Depends(dep_embedder),
    bm25: BM25Index = Depends(dep_bm25),
):
    settings = get_settings()

    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Read file
    file_bytes = await file.read()

    # Validate file size
    max_size = settings.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB",
        )

    # Check for duplicate document (same filename already indexed)
    existing_docs = vectorstore.get_document_ids()
    for doc in existing_docs:
        if doc.get("source") == file.filename:
            raise HTTPException(
                status_code=409,
                detail=f"Document '{file.filename}' is already indexed (ID: {doc['document_id'][:12]}...). "
                       f"Delete it first if you want to re-upload.",
            )

    # Parse document
    try:
        raw_text = parse_document(file_bytes, file.filename)
    except Exception as e:
        logger.error(f"Failed to parse '{file.filename}': {e}")
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}")

    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from file")

    # Extract metadata
    page_count = get_page_count(file_bytes, file.filename)
    metadata = extract_metadata(raw_text, file.filename, page_count=page_count)

    # Create document
    document_id = generate_id()

    # Chunk text
    chunk_dicts = chunk_text(
        raw_text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunk_dicts:
        raise HTTPException(status_code=422, detail="Document produced no text chunks")

    chunks = [
        Chunk(
            chunk_id=generate_id(),
            document_id=document_id,
            text=c["text"],
            metadata=metadata,
            chunk_index=c["chunk_index"],
            start_char=c["start_char"],
            end_char=c["end_char"],
        )
        for c in chunk_dicts
    ]

    # Embed chunks
    try:
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)
    except Exception as e:
        logger.error(f"Embedding failed for '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # Store in Qdrant
    try:
        vectorstore.upsert_chunks(chunks, embeddings)
    except Exception as e:
        logger.error(f"Vector store upsert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store document: {e}")

    # Add to BM25 index
    bm25.add_documents(chunks)

    logger.info(f"Ingested '{file.filename}': {len(chunks)} chunks")

    return IngestResponse(
        document_id=document_id,
        filename=file.filename,
        num_chunks=len(chunks),
        message=f"Successfully ingested '{file.filename}' with {len(chunks)} chunks",
    )


@router.get("/documents")
async def list_documents(
    vectorstore: VectorStoreService = Depends(dep_vectorstore),
):
    try:
        docs = vectorstore.get_document_ids()
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    vectorstore: VectorStoreService = Depends(dep_vectorstore),
    bm25: BM25Index = Depends(dep_bm25),
):
    try:
        vectorstore.delete_document(document_id)
        bm25.rebuild_from_vectorstore(vectorstore)
        return {"message": f"Document '{document_id}' deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete document '{document_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")
