<div align="center">

# RagCore

**Production-Grade Retrieval-Augmented Generation — Hybrid Search, Reranking, and Streaming**

*Upload documents. Ask questions. Get streamed answers with cited sources.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async%20SSE-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Source%20Available-blue.svg)](LICENSE)
[![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-FFD21E)](https://huggingface.co/spaces/NinjainPJs/RagCore)

[**Live Demo →**](https://huggingface.co/spaces/NinjainPJs/RagCore)

</div>

---

## Overview

RagCore is a production-grade RAG system for document Q&A. It combines BM25 keyword search with Qdrant dense vector retrieval, merges the results via Reciprocal Rank Fusion, re-scores with a FlashRank cross-encoder, and streams the final answer from Gemini 2.5 Flash with inline source citations.

The FastAPI backend mounts a Gradio UI for interactive document upload and querying, while exposing a clean REST API with Server-Sent Events (SSE) streaming for programmatic access.

**What makes this different from typical RAG demos:**
- **Hybrid retrieval** — BM25 + Qdrant dense search merged with Reciprocal Rank Fusion. Most tutorials use only one retrieval method.
- **Cross-encoder reranking** — FlashRank ms-marco-MiniLM-L-12-v2 re-scores the merged candidates for precision before the LLM call.
- **SSE streaming** — answers stream token-by-token via Server-Sent Events; no waiting for the full generation to complete.
- **Source citations** — every answer includes inline `[1][2]` references traceable to the source document and chunk.
- **Sentence-aware chunking** — chunks never break mid-sentence; configurable overlap prevents context loss at boundaries.

---

## Architecture

```
INGESTION PATH (one-time per document set)
----------------------------------------------------------
  User uploads PDF / TXT / HTML
      |
      v
  Parser                 ->  text + metadata per section
      |                      (pypdf, BeautifulSoup)
      v
  Sentence-Aware Chunker ->  sentence-boundary chunks
      |                      with configurable overlap
      v
  Dual Indexing
      |-- Qdrant          ->  384-dim dense embeddings
      |                       (all-MiniLM-L6-v2)
      +-- BM25 (in-memory)->  keyword index synced from Qdrant

QUERY PATH (real-time, per question)
----------------------------------------------------------
  User question
      |
      v
  Query Analyzer         ->  intent detection + filter extraction
      |
      v
  Hybrid Retrieval
      |-- Qdrant top-K    ->  dense semantic candidates
      |-- BM25 top-K      ->  sparse keyword candidates
      +-- RRF merge       ->  Reciprocal Rank Fusion
      |
      v
  FlashRank Reranker      ->  cross-encoder re-scoring
      |                       (ms-marco-MiniLM-L-12-v2)
      v
  Gemini 2.5 Flash        ->  streaming generation
      |                       with [Source:N] markers
      v
  Citation Resolver       ->  [1][2] inline references
      |                       with source document metadata
      v
  SSE Stream              ->  token-by-token response to client
```

---

## Features

| Feature | Detail |
|---------|--------|
| **Hybrid search** | Dense (Qdrant 384-dim) + sparse (BM25) merged via Reciprocal Rank Fusion |
| **Cross-encoder reranking** | FlashRank ms-marco-MiniLM-L-12-v2 for precision re-scoring before LLM |
| **SSE streaming** | Server-Sent Events deliver token-by-token streaming responses |
| **Source citations** | Inline `[1][2]` references with source document and chunk metadata |
| **Multi-format ingestion** | PDF, TXT, and HTML document parsing |
| **Sentence-aware chunking** | Configurable overlap without mid-sentence breaks |
| **Query analysis** | Intent detection and filter extraction before retrieval |
| **Metadata extraction** | Title, date, and tags automatically extracted from documents |
| **Gradio UI** | Interactive Q&A interface mounted inside FastAPI at `/ui` |
| **REST API** | Clean JSON endpoints for programmatic access with full OpenAPI docs |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI, Uvicorn | Async REST backend with SSE streaming |
| **Frontend** | Gradio (mounted inside FastAPI) | Interactive document upload and Q&A UI |
| **Vector DB** | Qdrant Cloud | Dense vector storage and ANN search |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | 384-dim dense vector representations |
| **Keyword Search** | BM25 (rank_bm25, in-memory) | Lexical keyword matching |
| **Reranking** | FlashRank (ms-marco-MiniLM-L-12-v2) | Cross-encoder semantic relevance scoring |
| **LLM** | Google Gemini 2.5 Flash | Streaming answer generation |
| **Parsers** | pypdf, BeautifulSoup | PDF and HTML document extraction |
| **Config** | Pydantic-settings | Type-safe environment variable configuration |
| **Deployment** | Docker, HuggingFace Spaces | Container-based cloud hosting |

---

## Project Structure

```
ragcore/app/
├── main.py                    # FastAPI app entry point + Gradio mount
├── config.py                  # Pydantic settings (env vars)
├── api/routes/
│   ├── health.py              # GET /health
│   ├── ingest.py              # POST /api/ingest, GET/DELETE /api/documents
│   └── query.py               # POST /api/search, POST /api/ask (SSE stream)
├── core/
│   ├── embedder.py            # sentence-transformers wrapper
│   ├── vectorstore.py         # Qdrant client wrapper
│   ├── bm25.py                # BM25 keyword index (in-memory)
│   ├── retriever.py           # Hybrid search + RRF fusion
│   ├── reranker.py            # FlashRank cross-encoder
│   ├── query_analyzer.py      # Intent classification + filter extraction
│   ├── generator.py           # Prompt builder with [Source:N] markers
│   ├── llm.py                 # Gemini API with rate limiting and streaming
│   ├── chunker.py             # Sentence-aware text chunking
│   └── metadata.py            # Title, date, and tag extraction
├── models/
│   ├── document.py            # Chunk, Document, DocumentMetadata
│   └── schemas.py             # API request and response schemas
├── ui/
│   └── gradio_app.py          # Gradio web interface
└── utils/
    ├── parsers.py             # PDF, HTML, TXT parsing logic
    └── helpers.py             # ID generation, text cleaning
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A Qdrant Cloud account and cluster ([free tier at cloud.qdrant.io](https://cloud.qdrant.io))
- A Google Gemini API key ([free at aistudio.google.com](https://aistudio.google.com))

### 1. Clone and install

```bash
git clone https://github.com/ninjacode911/Project-RagCore.git
cd Project-RagCore
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and add:
# QDRANT_URL=https://your-cluster.qdrant.io
# QDRANT_API_KEY=your_qdrant_key
# GEMINI_API_KEY=your_gemini_key
```

### 3. Run

```bash
uvicorn ragcore.app.main:app --reload

# API:    http://localhost:8000
# Gradio: http://localhost:8000/ui
# Docs:   http://localhost:8000/docs
```

### 4. Use it

1. Open the Gradio UI at `http://localhost:8000/ui`
2. Upload a PDF, TXT, or HTML document
3. Type a question — the answer streams token-by-token with inline `[1][2]` citations

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | — | **Required.** Qdrant cluster URL |
| `QDRANT_API_KEY` | — | **Required.** Qdrant API key |
| `GEMINI_API_KEY` | — | **Required.** Google Gemini API key |
| `COLLECTION_NAME` | `ragcore` | Qdrant collection name |
| `TOP_K_DENSE` | `20` | Number of Qdrant dense candidates |
| `TOP_K_SPARSE` | `20` | Number of BM25 sparse candidates |
| `RERANK_TOP_K` | `5` | Candidates passed to LLM after reranking |
| `CHUNK_SIZE` | `512` | Max tokens per document chunk |
| `CHUNK_OVERLAP` | `64` | Overlap tokens between adjacent chunks |

---

## License

**Source Available — All Rights Reserved.** See [LICENSE](LICENSE) for full terms.

The source code is publicly visible for viewing and educational purposes. Any use in personal, commercial, or academic projects requires explicit written permission from the author.

To request permission: navnitamrutharaj1234@gmail.com

**Author:** Navnit Amrutharaj
