import logging
import time
from collections.abc import AsyncGenerator

from app.core.llm import GeminiService
from app.core.reranker import RerankerService
from app.models.schemas import GeneratedAnswer, RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant answering questions based on the provided context.

CONTEXT:
{context}

RULES:
- Answer based ONLY on the provided context.
- Cite sources using [1], [2], etc. inline after the relevant information.
- If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer this question."
- Be concise but thorough.
- Use markdown formatting for readability.

QUESTION: {query}

ANSWER:"""

SUMMARY_PROMPT = """You are a helpful assistant. Summarize the following context.

CONTEXT:
{context}

RULES:
- Provide a structured summary using markdown.
- Cite sources using [1], [2], etc.
- Cover the key points from all provided sources.

QUESTION: {query}

SUMMARY:"""


class AnswerGenerator:
    def __init__(self, llm: GeminiService, reranker: RerankerService):
        self.llm = llm
        self.reranker = reranker

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.source or "unknown"
            header = f"[{i}] (Source: {source})"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n".join(parts)

    def _build_prompt(self, query: str, chunks: list[RetrievedChunk], intent: str = "factual") -> str:
        context = self._build_context(chunks)
        template = SUMMARY_PROMPT if intent == "summarize" else SYSTEM_PROMPT
        return template.format(context=context, query=query)

    def generate_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        rerank_top_k: int = 5,
        intent: str = "factual",
    ) -> GeneratedAnswer:
        start = time.perf_counter()

        # Rerank
        reranked = self.reranker.rerank(query, chunks, top_k=rerank_top_k)
        if not reranked:
            return GeneratedAnswer(
                query=query,
                answer="No relevant documents found to answer your question.",
                sources=[],
                generation_time_ms=0,
                model=self.llm.model_name,
            )

        prompt = self._build_prompt(query, reranked, intent)
        answer = self.llm.generate(prompt)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Generated answer in {elapsed:.0f}ms")

        return GeneratedAnswer(
            query=query,
            answer=answer,
            sources=reranked,
            generation_time_ms=elapsed,
            model=self.llm.model_name,
        )

    async def generate_answer_stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        rerank_top_k: int = 5,
        intent: str = "factual",
    ) -> AsyncGenerator[str | GeneratedAnswer, None]:
        # Rerank
        reranked = self.reranker.rerank(query, chunks, top_k=rerank_top_k)
        if not reranked:
            yield GeneratedAnswer(
                query=query,
                answer="No relevant documents found to answer your question.",
                sources=[],
                generation_time_ms=0,
                model=self.llm.model_name,
            )
            return

        prompt = self._build_prompt(query, reranked, intent)
        start = time.perf_counter()

        async for text_chunk in self.llm.generate_stream(prompt):
            yield text_chunk

        elapsed = (time.perf_counter() - start) * 1000

        # Final message with sources
        yield GeneratedAnswer(
            query=query,
            answer="",  # Full answer was streamed
            sources=reranked,
            generation_time_ms=elapsed,
            model=self.llm.model_name,
        )
