import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import dep_generator, dep_query_analyzer, dep_retriever
from app.core.generator import AnswerGenerator
from app.core.query_analyzer import QueryAnalyzer
from app.core.retriever import HybridRetriever
from app.models.schemas import (
    GeneratedAnswer,
    QueryRequest,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


def _resolve_filters(request_filters, analyzed_filters):
    """Use explicit request filters if provided, otherwise use analyzed filters only if they contain values."""
    if request_filters and request_filters.has_filters():
        return request_filters
    if analyzed_filters and analyzed_filters.has_filters():
        return analyzed_filters
    return None


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retriever: HybridRetriever = Depends(dep_retriever),
    analyzer: QueryAnalyzer = Depends(dep_query_analyzer),
):
    try:
        start = time.perf_counter()

        analyzed = analyzer.analyze(request.query)
        filters = _resolve_filters(request.filters, analyzed.extracted_filters)

        results = retriever.retrieve(
            query=analyzed.clean_query,
            top_k=request.top_k,
            filters=filters,
        )

        elapsed = (time.perf_counter() - start) * 1000

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=elapsed,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@router.post("/ask")
async def ask(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(dep_retriever),
    generator: AnswerGenerator = Depends(dep_generator),
    analyzer: QueryAnalyzer = Depends(dep_query_analyzer),
):
    try:
        analyzed = analyzer.analyze(request.query)
        filters = _resolve_filters(request.filters, analyzed.extracted_filters)

        chunks = retriever.retrieve(
            query=analyzed.clean_query,
            top_k=request.top_k,
            filters=filters,
        )

        if request.stream:
            return StreamingResponse(
                _stream_response(request.query, chunks, generator, request.rerank_top_k, analyzed.intent),
                media_type="text/event-stream",
            )

        answer = generator.generate_answer(
            query=request.query,
            chunks=chunks,
            rerank_top_k=request.rerank_top_k,
            intent=analyzed.intent,
        )
        return answer
    except Exception as e:
        logger.error(f"Ask failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


async def _stream_response(
    query: str,
    chunks,
    generator: AnswerGenerator,
    rerank_top_k: int,
    intent: str,
):
    try:
        async for item in generator.generate_answer_stream(
            query=query,
            chunks=chunks,
            rerank_top_k=rerank_top_k,
            intent=intent,
        ):
            if isinstance(item, str):
                yield f"data: {json.dumps({'text': item})}\n\n"
            elif isinstance(item, GeneratedAnswer):
                sources = [
                    {
                        "chunk_id": s.chunk_id,
                        "text": s.text[:200],
                        "source": s.metadata.source,
                        "score": s.score,
                    }
                    for s in item.sources
                ]
                yield f"data: {json.dumps({'done': True, 'sources': sources, 'model': item.model, 'time_ms': item.generation_time_ms})}\n\n"
    except Exception as e:
        logger.error(f"Streaming failed: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
