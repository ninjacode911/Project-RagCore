import logging
from functools import lru_cache

from app.core.bm25 import BM25Index, get_bm25
from app.core.embedder import EmbedderService, get_embedder
from app.core.generator import AnswerGenerator
from app.core.llm import GeminiService, get_llm
from app.core.query_analyzer import QueryAnalyzer
from app.core.reranker import RerankerService, get_reranker
from app.core.retriever import HybridRetriever
from app.core.vectorstore import VectorStoreService, get_vectorstore

logger = logging.getLogger(__name__)


def dep_embedder() -> EmbedderService:
    return get_embedder()


def dep_vectorstore() -> VectorStoreService:
    return get_vectorstore()


def dep_bm25() -> BM25Index:
    return get_bm25()


def dep_reranker() -> RerankerService:
    return get_reranker()


def dep_llm() -> GeminiService:
    return get_llm()


@lru_cache
def dep_query_analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


def dep_retriever() -> HybridRetriever:
    return HybridRetriever(
        vectorstore=get_vectorstore(),
        bm25=get_bm25(),
        embedder=get_embedder(),
    )


def dep_generator() -> AnswerGenerator:
    return AnswerGenerator(
        llm=get_llm(),
        reranker=get_reranker(),
    )
