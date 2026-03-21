<div align="center">

# RagCore

**Production-ready Retrieval-Augmented Generation with hybrid search, cross-encoder reranking, and streaming responses**

[![Live Demo](https://img.shields.io/badge/Live_Demo-HuggingFace_Spaces-8b5cf6?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/NinjainPJs/RagCore)
[![License](https://img.shields.io/badge/License-Source_Available-f59e0b?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python)](ragcore/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async_SSE-009688?style=for-the-badge&logo=fastapi)](ragcore/app/)

Hybrid BM25 + dense retrieval. FlashRank reranking. Gemini 2.5 Flash streaming.

</div>

---

## Overview

RagCore is a production-grade RAG system for document Q&A. It combines BM25 keyword search with Qdrant dense vector retrieval, merges the results via Reciprocal Rank Fusion, re-scores with a FlashRank cross-encoder, and streams the final answer from Gemini 2.5 Flash with inline source citations.

The FastAPI backend mounts a Gradio UI for interactive document upload and querying, while exposing a clean REST API with Server-Sent Events (SSE) streaming for programmatic access.

---

## Pipeline

```
Document Upload
        |
        v
+--------------------------------+
|  Parser (PDF, TXT, HTML)       |
|  Sentence-aware chunking       |
|  Metadata extraction           |
+----------------+---------------+
                 |
     +-----------v-----------+
     |    Dual Indexing       |
     |  Qdrant (384-dim dense)|
     |  BM25 (in-memory)      |
     +-----------+-----------+

                 | Query Time

     +-----------v-------------------------------+
     |  Query Analyzer (intent + filters)        |
     +-----------+-------------------------------+
                 |
     +-----------v-----------+
     |  Hybrid Retrieval      |
     |  Qdrant -> top-K dense |
     |  BM25 -> top-K sparse  |
     |  RRF merge             |
     +-----------+-----------+
                 |
     +-----------v-----------+
     |  FlashRank Reranker    |
     |  ms-marco cross-encoder|
     +-----------+-----------+
                 |
     +-----------v-----------+
     |  Gemini 2.5 Flash      |
     |  Streaming SSE         |
     |  [1][2] source cites   |
     +-----------------------+
```

---

## Features

| Feature | Details |
|---------|---------|
| **Hybrid search** | Dense (Qdrant) + sparse (BM25) merged via Reciprocal Rank Fusion |
| **Cross-encoder reranking** | FlashRank ms-marco-MiniLM-L-12-v2 for precision re-scoring |
| **Streaming responses** | Server-Sent Events (SSE) for real-time answer generation |
| **Source citations** | Inline [1][2] references with document source metadata |
| **Multi-format ingestion** | PDF, TXT, and HTML document parsing |
| **Sentence-aware chunking** | Configurable overlap without mid-sentence breaks |
| **Query analysis** | Intent detection and filter extraction before retrieval |
| **Metadata extraction** | Title, date, and tags automatically extracted from documents |
| **Gradio UI** | Interactive Q&A interface mounted inside FastAPI |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.10+, Uvicorn (async + SSE) |
| Frontend | Gradio (mounted inside FastAPI) |
| Vector DB | Qdrant Cloud (dense, 384-dim embeddings) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Keyword Search | BM25 (rank_bm25, in-memory) |
| Reranker | FlashRank (ms-marco-MiniLM-L-12-v2) |
| LLM | Google Gemini 2.5 Flash (streaming) |
| Parsers | pypdf, BeautifulSoup |
| Deployment | Docker, HuggingFace Spaces |

---

## Project Structure

```
ragcore/app/
├── main.py                    # FastAPI app + Gradio mount
├── config.py                  # Pydantic settings (env vars)
├── api/routes/
│   ├── health.py              # GET /health
│   ├── ingest.py              # POST /api/ingest, GET/DELETE /api/documents
│   └── query.py               # POST /api/search, POST /api/ask (SSE)
├── core/
│   ├── embedder.py            # sentence-transformers wrapper
│   ├── vectorstore.py         # Qdrant client
│   ├── bm25.py                # BM25 keyword index
│   ├── retriever.py           # Hybrid search + RRF fusion
│   ├── reranker.py            # FlashRank cross-encoder
│   ├── query_analyzer.py      # Intent + filter extraction
│   ├── generator.py           # Prompt builder
│   ├── llm.py                 # Gemini API with rate limiting
│   ├── chunker.py             # Sentence-aware text chunking
│   └── metadata.py            # Title/date/tag extraction
├── models/
│   ├── document.py            # Chunk, Document, DocumentMetadata
│   └── schemas.py             # API request/response schemas
├── ui/gradio_app.py           # Gradio web interface
└── utils/
    ├── parsers.py             # PDF/HTML/TXT parsing
    └── helpers.py             # ID generation, text cleaning
```

---

## Quick Start

```bash
git clone https://github.com/ninjacode911/Project-RagCore.git
cd Project-RagCore
pip install -r requirements.txt
```

Create a `.env` file:

```
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key
GEMINI_API_KEY=your_gemini_key
```

```bash
uvicorn ragcore.app.main:app --reload

# API:    http://localhost:8000
# Gradio: http://localhost:8000/ui
# Docs:   http://localhost:8000/docs
```

---

## License

Source Available — All Rights Reserved. See [LICENSE](LICENSE) for details.
