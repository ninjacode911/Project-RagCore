---
title: RagCore
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# RagCore

**A production-ready Retrieval-Augmented Generation system with hybrid search, metadata filtering, and a conversational UI.**

RagCore solves the problem of querying unstructured documents (PDFs, text files, HTML pages) using natural language. It ingests documents, splits them into semantically meaningful chunks, indexes them in both a vector database and a BM25 keyword index, then retrieves and reranks the most relevant passages to generate grounded, citation-backed answers using Google Gemini.

Unlike naive RAG implementations that rely solely on vector similarity, RagCore combines dense (semantic) and sparse (keyword) retrieval using Reciprocal Rank Fusion, applies a cross-encoder reranker to promote the most relevant passages, and uses an intelligent query analyzer that automatically extracts filters (date ranges, document types, sources) from natural language queries.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Core Components Deep Dive](#core-components-deep-dive)
5. [Data Models](#data-models)
6. [API Reference](#api-reference)
7. [UI Guide](#ui-guide)
8. [Setup and Installation](#setup-and-installation)
9. [Deployment](#deployment)
10. [Configuration Reference](#configuration-reference)
11. [How It Works End-to-End](#how-it-works-end-to-end)
12. [Testing](#testing)
13. [CI/CD](#cicd)
14. [Performance and Limits](#performance-and-limits)
15. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

RagCore is built as a FastAPI application with two main pipelines: **Ingestion** and **Query**. A Gradio-based UI is mounted directly onto the FastAPI app at `/ui`.

### Ingestion Pipeline

```
+------------------+     +----------------+     +-------------------+
|   File Upload    | --> |    Parser      | --> |    Text Cleaner   |
| (PDF/TXT/HTML)   |     | (pypdf/bs4)    |     | (regex cleanup)   |
+------------------+     +----------------+     +-------------------+
                                                        |
                                                        v
+------------------+     +----------------+     +-------------------+
|  Qdrant Cloud    | <-- |   Embedder     | <-- |    Chunker        |
|  (vector store)  |     | (MiniLM-L6-v2) |     | (sentence-aware)  |
+------------------+     +----------------+     +-------------------+
        |                                               |
        |                                               v
        |                                      +-------------------+
        +------------------------------------> |  BM25 Index       |
                                               | (in-memory)       |
                                               +-------------------+
                                                        ^
                                                        |
                                               +-------------------+
                                               | Metadata Extractor|
                                               | (title/dates/tags)|
                                               +-------------------+
```

**Step-by-step flow:**

1. User uploads a file via the `/api/ingest` endpoint or the Gradio UI.
2. The **Parser** detects file type by extension and extracts raw text (pypdf for PDFs, BeautifulSoup for HTML, direct decoding for TXT).
3. The **Text Cleaner** normalizes whitespace, collapses blank lines, and trims each line.
4. The **Metadata Extractor** pulls out the document title (first non-empty line), dates (via regex patterns), and tags (frequent capitalized phrases).
5. The **Chunker** splits text into overlapping chunks at sentence boundaries, respecting a configurable word-count limit.
6. The **Embedder** encodes each chunk into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer.
7. Chunks with their vectors and payload metadata are upserted into **Qdrant Cloud** in batches of 100.
8. The same chunks are added to the in-memory **BM25 index** for keyword search.

### Query Pipeline

```
+------------------+     +-------------------+     +------------------+
|   User Query     | --> |  Query Analyzer   | --> |  Hybrid Retriever|
| "What is RAG     |     | (intent, filters, |     |                  |
|  from PDFs?"     |     |  cleaned query)   |     |  +----------+   |
+------------------+     +-------------------+     |  |Dense     |   |
                                                   |  |(Qdrant)  |   |
                                                   |  +----------+   |
                                                   |       |         |
                                                   |  +----------+   |
                                                   |  |Sparse    |   |
                                                   |  |(BM25)    |   |
                                                   |  +----------+   |
                                                   |       |         |
                                                   |  +----------+   |
                                                   |  |RRF Fusion|   |
                                                   |  +----------+   |
                                                   +------------------+
                                                          |
                                                          v
                         +-------------------+     +------------------+
                         |  Answer Generator | <-- |   Reranker       |
                         | (Gemini Flash)    |     | (FlashRank)      |
                         +-------------------+     +------------------+
                                |
                                v
                         +-------------------+
                         |  Cited Answer     |
                         |  with Sources     |
                         +-------------------+
```

**Step-by-step flow:**

1. User submits a natural language query.
2. The **Query Analyzer** classifies intent (factual, summarize, comparative, list, explanatory), extracts inline filters (doc type, date range, source filename), and produces a cleaned query.
3. The **Hybrid Retriever** runs two parallel searches:
   - **Dense search**: encodes the query with the same embedding model, queries Qdrant with cosine similarity, fetching `top_k * 2` results.
   - **Sparse search**: tokenizes the query and scores all chunks via BM25Okapi, also fetching `top_k * 2` results.
4. Results are fused using **Reciprocal Rank Fusion (RRF)** with configurable weights (default: 0.6 dense, 0.4 sparse).
5. The top-K fused results are passed to the **Reranker** (FlashRank cross-encoder), which rescores and selects the best 5 passages.
6. The **Answer Generator** builds a prompt with numbered context passages and sends it to **Google Gemini Flash**, which generates a cited, markdown-formatted answer.
7. The answer is returned with source references (streaming or non-streaming).

---

## Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.12 | Runtime language. Chosen for its ML/NLP ecosystem. |
| **FastAPI** | >=0.110 | Async web framework. High performance, automatic OpenAPI docs, dependency injection. |
| **Uvicorn** | >=0.29 | ASGI server for running FastAPI in production. |
| **Pydantic** | >=2.6 | Data validation and serialization for all request/response models. |
| **pydantic-settings** | >=2.2 | Environment-based configuration with `.env` file support. |
| **sentence-transformers** | >=2.6 | Embedding model loading and inference (`all-MiniLM-L6-v2`). Chosen for fast CPU inference and high quality at 384 dimensions. |
| **qdrant-client** | >=1.8 | Client for Qdrant vector database. Chosen for its generous free tier (1GB), filtering support, and payload storage. |
| **rank-bm25** | >=0.2.2 | BM25Okapi implementation for sparse keyword retrieval. Lightweight, pure-Python, no external dependencies. |
| **FlashRank** | >=0.2 | Ultra-fast cross-encoder reranker (`ms-marco-MiniLM-L-12-v2`). Runs on CPU, no GPU required. |
| **google-generativeai** | >=0.5 | Official Google Gemini SDK. Gemini 2.0 Flash offers a free tier with 15 RPM. |
| **Gradio** | >=4.20 | Web UI framework mounted directly on FastAPI. Two-tab interface for Q&A and document management. |
| **pypdf** | >=4.1 | PDF text extraction. Handles most PDF formats without external system dependencies. |
| **beautifulsoup4** | >=4.12 | HTML parsing with tag stripping (removes scripts, styles, nav, footer, header). |
| **httpx** | >=0.27 | Async/sync HTTP client used by the Gradio UI to call the FastAPI backend. |
| **python-multipart** | >=0.0.9 | Required by FastAPI for file upload support. |
| **python-dateutil** | >=2.9 | Fuzzy date parsing for the query analyzer's absolute date extraction. |
| **Ruff** | >=0.3 | Fast Python linter. Used in CI for code quality checks. |
| **pytest** | >=8.0 | Test framework. Unit tests for chunker, parsers, query analyzer, retrieval, and API. |
| **Docker** | - | Containerization. Pre-downloads ML models in the build step for fast cold starts. |

---

## Project Structure

```
ragcore/
|-- .github/
|   +-- workflows/
|       +-- ci.yml                  # GitHub Actions CI pipeline (lint + test)
|-- app/
|   |-- __init__.py
|   |-- config.py                   # Settings class with all env vars, setup_logging()
|   |-- main.py                     # FastAPI app creation, lifespan, middleware, routing
|   |-- api/
|   |   |-- __init__.py
|   |   |-- deps.py                 # Dependency injection factories for all services
|   |   +-- routes/
|   |       |-- __init__.py
|   |       |-- health.py           # GET /health endpoint
|   |       |-- ingest.py           # POST /api/ingest, GET /api/documents, DELETE /api/documents/{id}
|   |       +-- query.py            # POST /api/search, POST /api/ask (with streaming)
|   |-- core/
|   |   |-- __init__.py
|   |   |-- bm25.py                 # BM25 index: tokenization, search, rebuild from vectorstore
|   |   |-- chunker.py              # Sentence-aware text chunking with overlap
|   |   |-- embedder.py             # SentenceTransformer embedding service
|   |   |-- generator.py            # Answer generation with prompt templates and streaming
|   |   |-- llm.py                  # Gemini API client with rate limiting
|   |   |-- metadata.py             # Metadata extraction (title, dates, tags)
|   |   |-- query_analyzer.py       # Query intent classification and filter extraction
|   |   |-- reranker.py             # FlashRank cross-encoder reranking
|   |   |-- retriever.py            # Hybrid retriever with RRF fusion
|   |   +-- vectorstore.py          # Qdrant client wrapper (CRUD, search, filtering)
|   |-- models/
|   |   |-- __init__.py
|   |   |-- document.py             # DocumentMetadata, Chunk, Document models
|   |   +-- schemas.py              # API request/response schemas (IngestResponse, QueryRequest, etc.)
|   |-- ui/
|   |   |-- __init__.py
|   |   +-- gradio_app.py           # Gradio Blocks UI (Ask tab, Documents tab)
|   +-- utils/
|       |-- __init__.py
|       |-- helpers.py              # generate_id, clean_text, count_words, timer, retry_with_backoff
|       +-- parsers.py              # File parsing (PDF, TXT, HTML) and page count extraction
|-- tests/
|   |-- __init__.py
|   |-- conftest.py                 # Shared fixtures (TestClient, sample_text)
|   |-- test_api.py                 # API integration tests (health, redirect, docs)
|   |-- test_chunker.py             # Chunker unit tests (empty, single, multiple, overlap)
|   |-- test_parsers.py             # Parser unit tests (UTF-8, Latin-1, HTML, unsupported)
|   |-- test_query_analyzer.py      # Query analyzer tests (intents, filters, dates)
|   +-- test_retrieval.py           # RRF fusion tests (basic, empty, weights, filters)
|-- .dockerignore
|-- .env                            # Environment variables (not committed to git)
|-- .gitignore
|-- Dockerfile                      # Python 3.12-slim, pre-downloads ML models
|-- docker-compose.yml              # Single-service compose with env_file
+-- requirements.txt                # All Python dependencies with version constraints
```

---

## Core Components Deep Dive

### Parsers (`app/utils/parsers.py`)

**What it does:** Extracts raw text from uploaded files based on their extension.

**Supported formats:** `.pdf`, `.txt`, `.html`, `.htm`

**How it works internally:**

- `parse_document(file_bytes, filename)` is the main dispatcher. It reads the file extension and calls the appropriate parser.
- **PDF parsing** uses `pypdf.PdfReader` to iterate over all pages, extract text from each, and join them with double newlines.
- **HTML parsing** uses `BeautifulSoup` with the `html.parser` backend. Before extracting text, it decomposes `<script>`, `<style>`, `<nav>`, `<footer>`, and `<header>` tags to remove boilerplate content. Text is extracted with `get_text(separator="\n")`.
- **TXT parsing** attempts UTF-8 decoding first, falling back to Latin-1 for non-UTF-8 files.
- All parsers pass their output through `clean_text()` for normalization.

**Key functions:**

```python
def parse_document(file_bytes: bytes, filename: str) -> str
def parse_pdf(file_bytes: bytes, filename: str) -> str
def parse_text(file_bytes: bytes, filename: str) -> str
def parse_html(file_bytes: bytes, filename: str) -> str
def get_page_count(file_bytes: bytes, filename: str) -> int | None
```

**Configuration:** No direct configuration. File size is validated at the API layer (`max_file_size_mb`).

---

### Chunker (`app/core/chunker.py`)

**What it does:** Splits raw text into overlapping chunks at sentence boundaries, sized by word count.

**How it works internally:**

1. Text is split into sentences using the regex pattern `(?<=[.!?])\s+` (splits after sentence-ending punctuation followed by whitespace).
2. Sentences are accumulated word-by-word into the current chunk.
3. When adding the next sentence would exceed `chunk_size` words, the current chunk is finalized.
4. Overlap is implemented by retaining the last `chunk_overlap` words from the previous chunk as the start of the new chunk.
5. Each chunk records its `text`, `start_char`, `end_char`, and `chunk_index`.

**Key function:**

```python
def chunk_text(
    text: str,
    chunk_size: int = 512,      # Maximum words per chunk
    chunk_overlap: int = 50,    # Number of overlapping words between consecutive chunks
) -> list[dict]
```

**Return format:** Each dict contains `{"text": str, "start_char": int, "end_char": int, "chunk_index": int}`.

**Configuration:**

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 512 | Maximum number of words per chunk |
| `CHUNK_OVERLAP` | 50 | Number of overlapping words between consecutive chunks |

**Design note:** Sentence-aware splitting avoids cutting mid-sentence, which improves both retrieval relevance and answer generation quality compared to fixed-character splitting.

---

### Metadata Extractor (`app/core/metadata.py`)

**What it does:** Automatically extracts structured metadata from raw document text.

**How it works internally:**

- **Title extraction:** Scans lines from the top of the document, returning the first non-empty line with more than 3 characters (truncated to 200 chars).
- **Date extraction:** Searches the first 2000 characters for dates using three regex patterns:
  - `YYYY-MM-DD` (ISO format)
  - `MM/DD/YYYY` (US format)
  - `Month DD, YYYY` (long format, e.g., "January 15, 2024")
- **Tag extraction:** Finds all capitalized phrases (e.g., "Machine Learning", "Neural Network") using regex, counts their occurrences, and returns the top 10 that appear at least twice. Tags are lowercased before returning.
- **Doc type:** Derived from the file extension (e.g., "pdf", "html", "txt").

**Key function:**

```python
def extract_metadata(raw_text: str, filename: str, page_count: int | None = None) -> DocumentMetadata
```

**Supporting functions:**

```python
def extract_title(text: str) -> str | None
def extract_dates(text: str) -> datetime | None
def extract_tags(text: str, max_tags: int = 10) -> list[str]
```

---

### Embedder (`app/core/embedder.py`)

**What it does:** Converts text into dense vector representations using a sentence transformer model.

**How it works internally:**

- Uses `sentence-transformers` to load the `all-MiniLM-L6-v2` model on CPU at startup.
- Encodes text in batches of 64 with L2 normalization enabled (so cosine similarity is equivalent to dot product).
- The model produces 384-dimensional embeddings.
- Singleton pattern via `get_embedder()` ensures the model is loaded only once.

**Key class:** `EmbedderService`

```python
class EmbedderService:
    EMBEDDING_DIM = 384

    def __init__(self, model_name: str)
    def embed_texts(self, texts: list[str]) -> list[list[float]]   # Batch embedding
    def embed_query(self, query: str) -> list[float]                # Single query embedding
```

**Configuration:**

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace sentence-transformers model name |
| `EMBEDDING_DIM` | 384 | Embedding vector dimensionality |

---

### Vector Store -- Qdrant (`app/core/vectorstore.py`)

**What it does:** Manages all interactions with the Qdrant vector database: collection management, upserting chunks, searching, filtering, scrolling, and deleting.

**How it works internally:**

- On initialization, connects to Qdrant Cloud using the provided URL and API key.
- `ensure_collection()` checks if the collection exists; if not, creates it with cosine distance and the configured vector size.
- **Upsert:** Chunks are uploaded in batches of 100 as `PointStruct` objects, with the chunk text and all metadata stored in the payload.
- **Search:** Uses `query_points()` with an optional `Filter` object built from `SearchFilters`. Over-fetches `top_k * 2` results to give the fusion step more candidates.
- **Filtering:** Supports exact match on `source`, `doc_type`, `MatchAny` on `tags`, and `Range` on `created_date`.
- **Scroll:** Iterates through all points in the collection using offset-based pagination (batch size 100). Used to rebuild the BM25 index on startup.
- **Document listing:** Aggregates all points by `document_id` to return a list of unique documents with chunk counts.

**Key class:** `VectorStoreService`

```python
class VectorStoreService:
    def __init__(self, url: str, api_key: str, collection_name: str)
    def ensure_collection(self, vector_size: int = 384) -> None
    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None
    def search(self, query_vector: list[float], limit: int = 10, filters: SearchFilters | None = None) -> list[dict]
    def delete_document(self, document_id: str) -> int
    def scroll_all(self, batch_size: int = 100) -> list[dict]
    def get_document_ids(self) -> list[dict]
    def count(self) -> int
```

**Payload schema stored per point:**

```json
{
    "text": "chunk text content",
    "document_id": "uuid-string",
    "chunk_index": 0,
    "source": "filename.pdf",
    "doc_type": "pdf",
    "title": "Document Title or null",
    "created_date": "2024-01-15T00:00:00 or null",
    "tags": ["machine learning", "neural networks"],
    "page_count": 12
}
```

**Configuration:**

| Setting | Default | Description |
|---|---|---|
| `QDRANT_URL` | (required) | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | (required) | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | `ragcore_docs` | Collection name in Qdrant |

---

### BM25 Index (`app/core/bm25.py`)

**What it does:** Maintains an in-memory BM25 keyword index for sparse retrieval alongside the dense vector search.

**How it works internally:**

- **Tokenization:** Text is lowercased, split into words via `\b\w+\b`, then filtered to remove stop words (58 common English words) and single-character tokens.
- Uses `rank_bm25.BM25Okapi`, which implements the Okapi BM25 scoring formula:
  ```
  score(D, Q) = SUM[ IDF(q) * (f(q,D) * (k1+1)) / (f(q,D) + k1 * (1 - b + b * |D|/avgdl)) ]
  ```
- On startup, the index is rebuilt from all existing points in Qdrant via `rebuild_from_vectorstore()`, which scrolls through all stored chunks.
- When new documents are ingested, `add_documents()` appends them and rebuilds the full BM25 corpus (the index is not incremental -- it rebuilds from the full document list).
- Search returns scored results filtered to only those with `score > 0`.

**Key class:** `BM25Index`

```python
class BM25Index:
    def __init__(self)
    def build_index(self, chunks: list[Chunk]) -> None
    def add_documents(self, chunks: list[Chunk]) -> None
    def search(self, query: str, top_k: int = 10) -> list[dict]
    def rebuild_from_vectorstore(self, vectorstore) -> None
    @property
    def doc_count(self) -> int
```

**Tokenization function:**

```python
def tokenize(text: str) -> list[str]
```

**Design note:** The in-memory approach means the BM25 index is rebuilt on every application restart (from Qdrant data). This is acceptable for small-to-medium collections (thousands of chunks) but would need a persistent store for larger deployments.

---

### Hybrid Retriever with RRF (`app/core/retriever.py`)

**What it does:** Combines dense (vector) and sparse (BM25) retrieval results using Reciprocal Rank Fusion.

**How it works internally:**

1. Embeds the query using the same `EmbedderService`.
2. Runs a dense search via Qdrant, fetching `top_k * 2` candidates (over-fetch to give fusion more options).
3. Runs a BM25 search, also fetching `top_k * 2` candidates.
4. If filters were provided, applies them post-hoc to BM25 results (since BM25 does not natively support metadata filtering).
5. Fuses both result lists using the **RRF formula**:

```
RRF_score(d) = SUM_over_lists[ weight_i * 1 / (k + rank_i(d)) ]
```

Where `k = 60` (smoothing constant), `rank_i(d)` is the rank of document `d` in list `i` (0-indexed), and `weight_i` is the list weight (default: 0.6 for dense, 0.4 for sparse).

6. Deduplicates by `chunk_id` and returns the top-K results as `RetrievedChunk` objects.

**Key class:** `HybridRetriever`

```python
class HybridRetriever:
    def __init__(self, vectorstore: VectorStoreService, bm25: BM25Index, embedder: EmbedderService)
    def retrieve(self, query: str, top_k: int = 10, filters: SearchFilters | None = None,
                 dense_weight: float = 0.6, sparse_weight: float = 0.4) -> list[RetrievedChunk]

    @staticmethod
    def rrf_fuse(result_lists: list[list[dict]], k: int = 60,
                 weights: list[float] | None = None) -> list[dict]

    @staticmethod
    def _apply_filters(results: list[dict], filters: SearchFilters) -> list[dict]
```

**Configuration:**

| Setting | Default | Description |
|---|---|---|
| `TOP_K` | 10 | Number of chunks to return from retrieval |
| `DENSE_WEIGHT` | 0.6 | Weight for dense (vector) search in RRF |
| `SPARSE_WEIGHT` | 0.4 | Weight for sparse (BM25) search in RRF |

**Why RRF?** Reciprocal Rank Fusion is a score-agnostic fusion method. Since BM25 scores and cosine similarity scores are on different scales, RRF uses only rank positions, making it a robust choice for combining heterogeneous retrieval signals.

---

### Reranker (`app/core/reranker.py`)

**What it does:** Rescores retrieved chunks using a cross-encoder model to improve ranking precision.

**How it works internally:**

- Uses FlashRank with the `ms-marco-MiniLM-L-12-v2` model, which is a lightweight cross-encoder trained on the MS MARCO passage ranking dataset.
- Unlike embedding models (which encode query and document independently), cross-encoders process the query-document pair jointly, allowing richer interaction signals.
- Input: the query string and a list of `RetrievedChunk` objects from the hybrid retriever.
- Output: the top `rerank_top_k` chunks reordered by cross-encoder score.
- The reranker model is cached in `./flashrank_cache/` to avoid re-downloading on each startup.

**Key class:** `RerankerService`

```python
class RerankerService:
    def __init__(self)
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]
```

**Configuration:**

| Setting | Default | Description |
|---|---|---|
| `RERANK_TOP_K` | 5 | Number of chunks to keep after reranking |

---

### LLM Client (`app/core/llm.py`)

**What it does:** Manages all communication with the Google Gemini API, including rate limiting and streaming.

**How it works internally:**

- Configures the `google.generativeai` library with the provided API key.
- Instantiates a `GenerativeModel` for the configured model name (default: `gemini-2.0-flash`).
- **Rate limiting:** Enforces a minimum interval between API calls based on `rpm_limit`. For the free tier (15 RPM), the minimum interval is 4 seconds. Uses `time.sleep()` for synchronous calls and `asyncio.sleep()` for async calls.
- **Synchronous generation:** `generate(prompt, temperature, max_tokens)` returns the full response text.
- **Streaming generation:** `generate_stream(prompt, temperature, max_tokens)` is an async generator that yields text chunks as they arrive from the API.

**Key class:** `GeminiService`

```python
class GeminiService:
    def __init__(self, api_key: str, model_name: str, rpm_limit: int = 15)
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> str
    async def generate_stream(self, prompt: str, temperature: float = 0.3,
                               max_tokens: int = 2048) -> AsyncGenerator[str, None]
```

**Configuration:**

| Setting | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model identifier |
| `GEMINI_RPM_LIMIT` | 15 | Requests per minute limit |
| `GEMINI_TEMPERATURE` | 0.3 | Generation temperature (lower = more deterministic) |
| `GEMINI_MAX_TOKENS` | 2048 | Maximum output tokens per generation |

---

### Query Analyzer (`app/core/query_analyzer.py`)

**What it does:** Parses natural language queries to extract intent, metadata filters, and a cleaned query string.

**How it works internally:**

The analyzer performs multiple regex-based extractions in sequence:

1. **Document type extraction:** Matches patterns like "PDFs", "pdf", "HTML", "text files", "txt" and sets the `doc_type` filter.
2. **Relative date extraction:** Matches temporal phrases like "last week", "last month", "this year", "today", "yesterday" and converts them to `date_from`/`date_to` datetime ranges.
3. **Absolute date extraction:** Matches "after {date}" and "before {date}" patterns. Uses `python-dateutil` for fuzzy parsing of the date string.
4. **Source extraction:** Matches "from {filename.ext}" patterns to filter by specific source file.
5. **Query cleaning:** Removes all matched filter phrases from the query, collapses whitespace, and strips dangling prepositions (about, from, in, on).
6. **Intent classification:** Matches the original query against patterns for five intent types:
   - `summarize` -- "summarize", "summary", "overview"
   - `comparative` -- "compare", "difference", "vs", "versus"
   - `list` -- "list", "enumerate", "what are all"
   - `explanatory` -- starts with "why", "how", "explain"
   - `factual` -- starts with "what", "who", "when", "where", "how many/much" (default fallback)
7. **Confidence scoring:** Starts at 0.5, incremented by 0.1 for each filter successfully extracted, capped at 1.0.

**Key class:** `QueryAnalyzer`

```python
class QueryAnalyzer:
    def analyze(self, query: str) -> AnalyzedQuery
```

**Example:**

Input: `"summarize PDFs from last month"`

Output:
```json
{
    "original_query": "summarize PDFs from last month",
    "clean_query": "summarize",
    "intent": "summarize",
    "extracted_filters": {
        "doc_type": "pdf",
        "date_from": "2026-02-17T00:00:00",
        "date_to": "2026-03-17T00:00:00"
    },
    "confidence": 0.7
}
```

---

### Answer Generator (`app/core/generator.py`)

**What it does:** Builds a prompt from retrieved chunks and generates a cited answer using the LLM.

**How it works internally:**

1. **Reranking:** Calls the `RerankerService` to narrow the retrieved chunks to `rerank_top_k`.
2. **Context building:** Formats each reranked chunk as a numbered passage with its source filename:
   ```
   [1] (Source: report.pdf)
   Chunk text content here...

   [2] (Source: notes.txt)
   Another chunk text...
   ```
3. **Prompt selection:** Uses `SYSTEM_PROMPT` for most intents and `SUMMARY_PROMPT` when the intent is "summarize".
4. **Prompt rules instruct the LLM to:**
   - Answer based ONLY on the provided context
   - Cite sources inline using [1], [2], etc.
   - Admit when context is insufficient
   - Use markdown formatting
5. **Streaming:** The `generate_answer_stream()` async generator yields text chunks during generation, then yields a final `GeneratedAnswer` object with source metadata.

**Key class:** `AnswerGenerator`

```python
class AnswerGenerator:
    def __init__(self, llm: GeminiService, reranker: RerankerService)
    def generate_answer(self, query: str, chunks: list[RetrievedChunk],
                        rerank_top_k: int = 5, intent: str = "factual") -> GeneratedAnswer
    async def generate_answer_stream(self, query: str, chunks: list[RetrievedChunk],
                                      rerank_top_k: int = 5, intent: str = "factual") -> AsyncGenerator
```

---

## Data Models

All models are defined using Pydantic v2 and live in `app/models/`.

### Core Document Models (`app/models/document.py`)

#### `DocumentMetadata`

Stores extracted metadata for a document or chunk.

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | `str` | `""` | Original filename |
| `doc_type` | `str` | `""` | File type without dot (e.g., "pdf", "html", "txt") |
| `title` | `str \| None` | `None` | Extracted title (first meaningful line) |
| `created_date` | `datetime \| None` | `None` | Extracted date from document content |
| `tags` | `list[str]` | `[]` | Auto-extracted topic tags |
| `page_count` | `int \| None` | `None` | Number of pages (PDFs only) |

#### `Chunk`

Represents a single text chunk derived from a document.

| Field | Type | Default | Description |
|---|---|---|---|
| `chunk_id` | `str` | `uuid4()` | Unique chunk identifier |
| `document_id` | `str` | `""` | Parent document identifier |
| `text` | `str` | `""` | Chunk text content |
| `metadata` | `DocumentMetadata` | `{}` | Inherited document metadata |
| `chunk_index` | `int` | `0` | Position of this chunk in the document |
| `start_char` | `int` | `0` | Start character offset in original text |
| `end_char` | `int` | `0` | End character offset in original text |

#### `Document`

Represents a full ingested document.

| Field | Type | Default | Description |
|---|---|---|---|
| `document_id` | `str` | `uuid4()` | Unique document identifier |
| `filename` | `str` | `""` | Original filename |
| `metadata` | `DocumentMetadata` | `{}` | Extracted metadata |
| `chunks` | `list[Chunk]` | `[]` | Child chunks (populated during ingestion) |
| `raw_text` | `str` | `""` | Full extracted text |

### API Schemas (`app/models/schemas.py`)

#### `IngestResponse`

Returned after successful document ingestion.

| Field | Type | Description |
|---|---|---|
| `document_id` | `str` | Assigned UUID |
| `filename` | `str` | Original filename |
| `num_chunks` | `int` | Number of chunks created |
| `message` | `str` | Human-readable success message |

#### `SearchFilters`

Used for metadata filtering in search and query operations.

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | `str \| None` | `None` | Filter by exact source filename |
| `doc_type` | `str \| None` | `None` | Filter by document type |
| `date_from` | `datetime \| None` | `None` | Filter documents created on or after this date |
| `date_to` | `datetime \| None` | `None` | Filter documents created on or before this date |
| `tags` | `list[str] \| None` | `None` | Filter by any matching tag |

#### `RetrievedChunk`

A chunk returned from retrieval, with its relevance score and rank.

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `str` | Chunk identifier |
| `document_id` | `str` | Parent document identifier |
| `text` | `str` | Chunk text |
| `score` | `float` | Relevance score (RRF-fused or reranker score) |
| `metadata` | `DocumentMetadata` | Chunk metadata |
| `rank` | `int` | Position in the result list (0-indexed) |

#### `SearchRequest`

Request body for the `/api/search` endpoint.

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | (required) | Natural language search query |
| `top_k` | `int` | `10` | Number of results to return |
| `filters` | `SearchFilters \| None` | `None` | Optional explicit filters (overrides auto-extraction) |

#### `SearchResponse`

Response from the `/api/search` endpoint.

| Field | Type | Description |
|---|---|---|
| `query` | `str` | Original query |
| `results` | `list[RetrievedChunk]` | Retrieved and ranked chunks |
| `total_results` | `int` | Number of results returned |
| `search_time_ms` | `float` | Total search time in milliseconds |

#### `QueryRequest`

Request body for the `/api/ask` endpoint.

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | (required) | Natural language question |
| `top_k` | `int` | `10` | Number of chunks to retrieve |
| `rerank_top_k` | `int` | `5` | Number of chunks to keep after reranking |
| `filters` | `SearchFilters \| None` | `None` | Optional explicit filters |
| `stream` | `bool` | `False` | Enable Server-Sent Events streaming |

#### `GeneratedAnswer`

Response from the `/api/ask` endpoint (non-streaming).

| Field | Type | Description |
|---|---|---|
| `query` | `str` | Original question |
| `answer` | `str` | Generated markdown answer with inline citations |
| `sources` | `list[RetrievedChunk]` | Source chunks used for generation |
| `generation_time_ms` | `float` | Total generation time in milliseconds |
| `model` | `str` | LLM model name used |

#### `AnalyzedQuery`

Internal model from the query analyzer (not directly exposed via API).

| Field | Type | Default | Description |
|---|---|---|---|
| `original_query` | `str` | - | The raw user query |
| `clean_query` | `str` | - | Query with filter phrases removed |
| `intent` | `str` | `"factual"` | Classified intent |
| `extracted_filters` | `SearchFilters` | `{}` | Automatically extracted filters |
| `confidence` | `float` | `0.5` | Confidence in filter extraction |

---

## API Reference

The FastAPI app automatically generates interactive API documentation at `/docs` (Swagger UI) and `/redoc` (ReDoc).

### Health Check

```
GET /health
```

Returns the status of all system components.

**Response:**

```json
{
    "status": "ok",
    "components": {
        "embedder": "loaded",
        "bm25": "142 documents",
        "vectorstore": "connected"
    }
}
```

**curl example:**

```bash
curl http://localhost:7860/health
```

---

### Ingest Document

```
POST /api/ingest
Content-Type: multipart/form-data
```

Uploads and indexes a document. The file is parsed, chunked, embedded, and stored in both the vector database and the BM25 index.

**Request:** Multipart form with a `file` field.

**Constraints:**
- Supported extensions: `.pdf`, `.txt`, `.html`, `.htm`
- Maximum file size: 10 MB (configurable via `MAX_FILE_SIZE_MB`)

**Response (200):**

```json
{
    "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "filename": "report.pdf",
    "num_chunks": 47,
    "message": "Successfully ingested 'report.pdf' with 47 chunks"
}
```

**Error responses:**
- `400` -- Missing filename or unsupported file type
- `413` -- File exceeds maximum size
- `422` -- Could not extract text from file

**curl example:**

```bash
curl -X POST http://localhost:7860/api/ingest \
  -F "file=@/path/to/document.pdf"
```

---

### List Documents

```
GET /api/documents
```

Returns all indexed documents with their metadata and chunk counts.

**Response (200):**

```json
{
    "documents": [
        {
            "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "source": "report.pdf",
            "title": "Annual Report 2024",
            "doc_type": "pdf",
            "num_chunks": 47
        }
    ],
    "total": 1
}
```

**curl example:**

```bash
curl http://localhost:7860/api/documents
```

---

### Delete Document

```
DELETE /api/documents/{document_id}
```

Removes all chunks for the given document from Qdrant and rebuilds the BM25 index.

**Response (200):**

```json
{
    "message": "Document 'a1b2c3d4-e5f6-7890-abcd-ef1234567890' deleted successfully"
}
```

**curl example:**

```bash
curl -X DELETE http://localhost:7860/api/documents/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

---

### Search (Retrieval Only)

```
POST /api/search
Content-Type: application/json
```

Performs hybrid retrieval without LLM generation. Useful for inspecting which chunks would be retrieved for a given query.

**Request body:**

```json
{
    "query": "What is retrieval-augmented generation?",
    "top_k": 10,
    "filters": {
        "doc_type": "pdf",
        "tags": ["machine learning"]
    }
}
```

**Response (200):**

```json
{
    "query": "What is retrieval-augmented generation?",
    "results": [
        {
            "chunk_id": "uuid",
            "document_id": "uuid",
            "text": "Retrieval-Augmented Generation (RAG) is...",
            "score": 0.0234,
            "metadata": {
                "source": "report.pdf",
                "doc_type": "pdf",
                "title": "Annual Report",
                "created_date": null,
                "tags": ["machine learning"],
                "page_count": 12
            },
            "rank": 0
        }
    ],
    "total_results": 10,
    "search_time_ms": 142.5
}
```

**curl example:**

```bash
curl -X POST http://localhost:7860/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 5}'
```

---

### Ask (Full RAG Pipeline)

```
POST /api/ask
Content-Type: application/json
```

Runs the full pipeline: query analysis, hybrid retrieval, reranking, and LLM answer generation.

**Request body:**

```json
{
    "query": "What are the key findings in the report?",
    "top_k": 10,
    "rerank_top_k": 5,
    "filters": null,
    "stream": false
}
```

**Response (200, non-streaming):**

```json
{
    "query": "What are the key findings in the report?",
    "answer": "Based on the provided documents, the key findings are:\n\n1. **Finding one** [1]...\n2. **Finding two** [2]...",
    "sources": [
        {
            "chunk_id": "uuid",
            "document_id": "uuid",
            "text": "chunk text...",
            "score": 0.892,
            "metadata": { "source": "report.pdf", "..." : "..." },
            "rank": 0
        }
    ],
    "generation_time_ms": 3420.5,
    "model": "gemini-2.0-flash"
}
```

**Streaming response (`"stream": true`):**

Returns `text/event-stream` with Server-Sent Events:

```
data: {"text": "Based on"}

data: {"text": " the provided"}

data: {"text": " documents..."}

data: {"done": true, "sources": [...], "model": "gemini-2.0-flash", "time_ms": 3420.5}
```

**curl examples:**

```bash
# Non-streaming
curl -X POST http://localhost:7860/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the report", "stream": false}'

# Streaming
curl -X POST http://localhost:7860/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "stream": true}' \
  --no-buffer
```

---

## UI Guide

RagCore includes a Gradio web interface mounted at `/ui` (the root `/` redirects there automatically).

### Ask Tab

The primary interaction surface for querying your documents.

**Components:**

- **Query input** -- A text box where you type your question in natural language. Supports pressing Enter to submit.
- **Document Type filter** -- Dropdown to restrict results to a specific file type: All, PDF, TXT, or HTML.
- **Stream response toggle** -- Checkbox (default: on) to enable real-time streaming of the answer as it is generated.
- **Ask button** -- Submits the query.
- **Answer area** -- Displays the generated answer with markdown formatting, followed by a "Sources" section listing each referenced chunk with its filename, relevance score, and a text snippet.
- **Example queries** -- Pre-filled example questions you can click to populate the query input.

### Documents Tab

Manages the document collection.

**Components:**

- **File upload zone** -- Drag-and-drop or click to select a file (`.pdf`, `.txt`, `.html`, `.htm`).
- **Upload & Index button** -- Triggers the ingestion pipeline. Shows a status card with filename, chunk count, and document ID on success.
- **Indexed Documents table** -- Displays all ingested documents with their filename, type, chunk count, and truncated document ID. Click "Refresh" to update.
- **Delete section** -- Paste a full document ID and click "Delete" to remove a document and all its chunks.

### Stats Bar

At the top of every tab, a card shows the current count of indexed documents and total chunks.

---

## Setup and Installation

### Prerequisites

- Python 3.12 or later
- A Qdrant Cloud account (free tier)
- A Google AI Studio account (free tier Gemini API key)
- (Optional) Docker and Docker Compose

### Step 1: Get API Keys

**Qdrant Cloud (vector database):**

1. Go to [https://cloud.qdrant.io](https://cloud.qdrant.io) and create a free account.
2. Create a new cluster (the free tier provides 1 GB of storage).
3. Copy the cluster URL (e.g., `https://abc123-xyz.us-east4-0.gcp.cloud.qdrant.io:6333`).
4. Generate an API key from the cluster dashboard.

**Google Gemini (LLM):**

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Click "Create API key" and select or create a Google Cloud project.
3. Copy the generated API key. The free tier allows 15 requests per minute for Gemini 2.0 Flash.

### Step 2: Clone and Configure

```bash
git clone <repository-url>
cd ragcore
```

Create a `.env` file in the `ragcore/` directory:

```env
# Required
GEMINI_API_KEY=your-gemini-api-key-here
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key-here

# Optional (these are the defaults)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384
QDRANT_COLLECTION=ragcore_docs
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=10
RERANK_TOP_K=5
DENSE_WEIGHT=0.6
SPARSE_WEIGHT=0.4
GEMINI_MODEL=gemini-2.0-flash
GEMINI_RPM_LIMIT=15
GEMINI_TEMPERATURE=0.3
GEMINI_MAX_TOKENS=2048
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=10
```

### Step 3: Running Locally

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate        # On Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

The first startup will download two ML models (~90 MB for the embedding model, ~50 MB for the reranker). Subsequent startups use cached models.

Once running:
- Web UI: [http://localhost:7860/ui](http://localhost:7860/ui)
- API docs: [http://localhost:7860/docs](http://localhost:7860/docs)
- Health check: [http://localhost:7860/health](http://localhost:7860/health)

### Step 4: Running with Docker

```bash
# Build and run
docker compose up --build

# Or build and run in detached mode
docker compose up --build -d
```

The Docker build pre-downloads both ML models into the image layer, so container startup is faster. The app is exposed on port 8000 (mapped from container port 7860).

Once running: [http://localhost:8000/ui](http://localhost:8000/ui)

---

## Deployment

### Deploying to HuggingFace Spaces

HuggingFace Spaces provides free hosting for Gradio and Docker-based applications. RagCore is pre-configured for deployment there.

**Step-by-step:**

1. **Create a HuggingFace account** at [https://huggingface.co](https://huggingface.co) if you do not have one.

2. **Create a new Space:**
   - Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
   - Choose a name (e.g., `ragcore`).
   - Select **Docker** as the SDK.
   - Choose the **Free** CPU basic tier.
   - Click "Create Space".

3. **Configure secrets:**
   - Go to your Space's Settings > Repository secrets.
   - Add the following secrets:
     - `GEMINI_API_KEY` -- your Google Gemini API key
     - `QDRANT_URL` -- your Qdrant Cloud cluster URL
     - `QDRANT_API_KEY` -- your Qdrant Cloud API key

4. **Push the code:**

   ```bash
   cd ragcore
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ragcore
   git push space main
   ```

   Alternatively, upload files via the HuggingFace web interface.

5. **Wait for the build** -- the Docker image will be built on HuggingFace's infrastructure. The first build takes 5-10 minutes due to model downloads. The Space will show "Running" when ready.

6. **Access your app** at `https://YOUR_USERNAME-ragcore.hf.space`.

**Important notes:**
- HuggingFace Spaces exposes port 7860 by default, which matches the Dockerfile's `EXPOSE 7860`.
- The free tier has 2 vCPU and 16 GB RAM, which is sufficient for RagCore.
- Spaces may sleep after inactivity. The first request after sleep triggers a cold start (30-60 seconds).

---

## Configuration Reference

All settings are managed via environment variables, loaded from a `.env` file by `pydantic-settings`.

| Variable | Type | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | string | `""` | **Required.** Google Gemini API key for LLM generation. |
| `QDRANT_URL` | string | `""` | **Required.** Full URL of the Qdrant Cloud cluster (including port). |
| `QDRANT_API_KEY` | string | `""` | **Required.** Qdrant Cloud API key for authentication. |
| `EMBEDDING_MODEL` | string | `all-MiniLM-L6-v2` | HuggingFace model name for sentence-transformers. |
| `EMBEDDING_DIM` | integer | `384` | Dimensionality of the embedding vectors. Must match the model. |
| `QDRANT_COLLECTION` | string | `ragcore_docs` | Name of the Qdrant collection to use. Created automatically if missing. |
| `CHUNK_SIZE` | integer | `512` | Maximum number of words per text chunk. |
| `CHUNK_OVERLAP` | integer | `50` | Number of words overlapping between consecutive chunks. |
| `TOP_K` | integer | `10` | Number of chunks retrieved by the hybrid retriever. |
| `RERANK_TOP_K` | integer | `5` | Number of chunks kept after cross-encoder reranking. |
| `DENSE_WEIGHT` | float | `0.6` | Weight for dense (vector) search in RRF fusion. Range: 0.0-1.0. |
| `SPARSE_WEIGHT` | float | `0.4` | Weight for sparse (BM25) search in RRF fusion. Range: 0.0-1.0. |
| `GEMINI_MODEL` | string | `gemini-2.0-flash` | Gemini model identifier. |
| `GEMINI_RPM_LIMIT` | integer | `15` | Maximum requests per minute to the Gemini API. |
| `GEMINI_TEMPERATURE` | float | `0.3` | LLM generation temperature. Lower values produce more deterministic output. |
| `GEMINI_MAX_TOKENS` | integer | `2048` | Maximum number of output tokens per LLM generation. |
| `LOG_LEVEL` | string | `INFO` | Logging level. Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL. |
| `MAX_FILE_SIZE_MB` | integer | `10` | Maximum allowed file size for upload in megabytes. |

---

## How It Works End-to-End

This section traces a complete user interaction: uploading a PDF and then asking a question about it.

### Phase 1: Document Ingestion

**User action:** Uploads `annual-report-2024.pdf` (2.1 MB, 45 pages) via the Gradio Documents tab.

1. The Gradio UI reads the file and sends it as a multipart POST to `http://localhost:7860/api/ingest`.

2. **Validation** (`ingest.py`):
   - Filename is checked: extension `.pdf` is in `SUPPORTED_EXTENSIONS`.
   - File size 2.1 MB is under the 10 MB limit.

3. **Parsing** (`parsers.py`):
   - `parse_pdf()` creates a `PdfReader` from the bytes.
   - Iterates over all 45 pages, extracting text from each.
   - Joins page texts with double newlines.
   - `clean_text()` normalizes whitespace: collapses 3+ consecutive newlines to 2, collapses horizontal whitespace to single spaces, trims each line.
   - Result: ~85,000 characters of cleaned text.

4. **Metadata extraction** (`metadata.py`):
   - `extract_title()` returns `"Annual Report 2024 - Acme Corporation"` (first meaningful line).
   - `extract_dates()` finds `"2024-03-15"` in the first 2000 chars, parses it to `datetime(2024, 3, 15)`.
   - `extract_tags()` finds frequent capitalized phrases: `["acme corporation", "revenue growth", "machine learning", ...]`.
   - `get_page_count()` returns `45`.
   - Final `DocumentMetadata`: source="annual-report-2024.pdf", doc_type="pdf", title="Annual Report 2024 - Acme Corporation", created_date=2024-03-15, tags=[...], page_count=45.

5. **Chunking** (`chunker.py`):
   - Splits the ~85,000 chars into sentences via `(?<=[.!?])\s+`.
   - Accumulates sentences until the word count exceeds 512.
   - Produces ~32 chunks, each with 50-word overlap with the next.
   - Each chunk records start_char, end_char, and chunk_index.

6. **Embedding** (`embedder.py`):
   - `embed_texts()` encodes all 32 chunk texts in a single batch (batch_size=64).
   - Returns 32 vectors, each of dimension 384, L2-normalized.

7. **Vector storage** (`vectorstore.py`):
   - `upsert_chunks()` creates 32 `PointStruct` objects with the vectors and payload.
   - Since 32 < 100, they are uploaded in a single batch.
   - Each point's payload includes text, document_id, chunk_index, source, doc_type, title, created_date, tags, page_count.

8. **BM25 indexing** (`bm25.py`):
   - `add_documents()` tokenizes each chunk (lowercase, remove stop words, remove single chars).
   - Appends to the document list and rebuilds the full BM25Okapi index.

9. **Response:** Returns `IngestResponse` with document_id, filename, num_chunks=32, and success message.

### Phase 2: Querying

**User action:** Types `"What was the revenue growth last year from PDFs?"` in the Ask tab with streaming enabled.

1. The Gradio UI sends a POST to `http://localhost:7860/api/ask` with:
   ```json
   {"query": "What was the revenue growth last year from PDFs?", "top_k": 10, "rerank_top_k": 5, "stream": true, "filters": {"doc_type": "pdf"}}
   ```
   (Note: the UI sets `doc_type` filter from the dropdown if not "All".)

2. **Query analysis** (`query_analyzer.py`):
   - Doc type extraction: matches "PDFs" -> `filters.doc_type = "pdf"`.
   - Date extraction: matches "last year" -> `filters.date_from = 2025-03-17`, `filters.date_to = 2026-03-17`.
   - Clean query: removes "last year" and "PDFs" -> `"What was the revenue growth"`.
   - Intent: matches `^(?:what|...)` -> `"factual"`.
   - Confidence: 0.5 + 0.1 (doc_type) + 0.1 (date) = 0.7.

3. **Hybrid retrieval** (`retriever.py`):
   - Embeds the clean query `"What was the revenue growth"` to a 384-dim vector.
   - **Dense search:** Queries Qdrant with the vector, limit=20 (top_k * 2), with filters for doc_type="pdf" and date range. Returns 20 results ranked by cosine similarity.
   - **Sparse search:** Tokenizes query to `["what", "revenue", "growth"]` (stop words removed), scores all BM25 documents, returns top 20 by BM25 score. Post-filters by doc_type="pdf".
   - **RRF fusion:** For each chunk, computes `score = 0.6 * 1/(60+dense_rank) + 0.4 * 1/(60+sparse_rank)`. Chunks appearing in both lists get boosted scores.
   - Deduplicates by chunk_id, takes top 10.

4. **Reranking** (`reranker.py`):
   - Creates passage pairs: (query, chunk_text) for all 10 retrieved chunks.
   - The FlashRank cross-encoder scores each pair jointly.
   - Returns the top 5 by cross-encoder score, with updated scores and ranks.

5. **Answer generation** (`generator.py`):
   - Builds context with numbered passages:
     ```
     [1] (Source: annual-report-2024.pdf)
     Revenue increased by 23% year-over-year...

     [2] (Source: annual-report-2024.pdf)
     The growth was primarily driven by...
     ```
   - Constructs the SYSTEM_PROMPT with context and query.
   - Calls `llm.generate_stream()` which respects the rate limit, then yields text chunks.

6. **Streaming response** (`query.py`):
   - Each text chunk from Gemini is wrapped as `data: {"text": "..."}\n\n` (SSE format).
   - The Gradio UI accumulates text and renders it progressively in the answer area.
   - Final SSE event includes `{"done": true, "sources": [...], "model": "gemini-2.0-flash", "time_ms": 3420}`.
   - Gradio formats the sources as styled cards showing filename, score, and snippet.

---

## Testing

### Running Tests

```bash
# Run all unit tests (excluding integration tests)
pytest tests/ -v --ignore=tests/test_integration.py -x

# Run a specific test file
pytest tests/test_chunker.py -v

# Run with coverage (install pytest-cov first)
pytest tests/ -v --ignore=tests/test_integration.py --cov=app
```

### Test Coverage

| Test File | Module Under Test | What Is Tested |
|---|---|---|
| `test_chunker.py` | `app.core.chunker` | Empty input, single sentence, multiple chunks, overlap behavior, chunk size limits |
| `test_parsers.py` | `app.utils.parsers` | UTF-8 text, Latin-1 fallback, HTML tag stripping, unsupported extensions, empty files, extension-based dispatch |
| `test_query_analyzer.py` | `app.core.query_analyzer` | Intent classification (factual, comparative, summarize, explanatory), doc type extraction, date extraction, clean query preservation |
| `test_retrieval.py` | `app.core.retriever` | RRF fusion (basic, empty lists, single list, weighted), metadata filter application |
| `test_api.py` | `app.main` (FastAPI) | Health endpoint returns 200 with components, root redirects to `/ui`, `/docs` page loads |

### Test Fixtures

Defined in `tests/conftest.py`:
- `client` -- A `FastAPI TestClient` instance for API testing.
- `sample_text` -- A paragraph about RAG for use in unit tests.

**Note:** Unit tests mock or avoid external dependencies (Qdrant, Gemini). The CI pipeline sets dummy API keys via environment variables. Integration tests (if present in `tests/test_integration.py`) are excluded from the default test run.

---

## CI/CD

### GitHub Actions Pipeline (`.github/workflows/ci.yml`)

The CI pipeline runs on every push to `main` and on every pull request targeting `main`.

**Pipeline steps:**

| Step | Description |
|---|---|
| Checkout | Clones the repository using `actions/checkout@v4` |
| Set up Python | Installs Python 3.12 via `actions/setup-python@v5` |
| Install dependencies | Runs `pip install -r requirements.txt` |
| Lint | Runs `ruff check .` for code style and quality |
| Unit tests | Runs `pytest tests/ -v --ignore=tests/test_integration.py -x` |

**Environment variables set during testing:**

```yaml
env:
  GEMINI_API_KEY: "test"
  QDRANT_URL: "http://localhost:6333"
  QDRANT_API_KEY: "test"
```

These are dummy values that allow the application to initialize its settings without connecting to real services. Tests that would require live connections are either mocked or skipped.

The `-x` flag causes pytest to stop on the first failure for faster feedback.

---

## Performance and Limits

### Free Tier Limits

| Service | Limit | Impact |
|---|---|---|
| **Qdrant Cloud** (free tier) | 1 GB storage | Approximately 500,000-700,000 chunks at 384 dimensions. More than sufficient for thousands of documents. |
| **Google Gemini** (free tier) | 15 requests per minute | RagCore enforces this with built-in rate limiting (4-second minimum interval between calls). Each question costs 1 API call. |
| **HuggingFace Spaces** (free tier) | 2 vCPU, 16 GB RAM | Sufficient for running the embedding model, reranker, and BM25 index concurrently. |

### Expected Latency

| Operation | Typical Latency | Notes |
|---|---|---|
| Document ingestion (10-page PDF) | 3-8 seconds | Dominated by embedding time on CPU |
| Document ingestion (50-page PDF) | 10-20 seconds | Linear with number of chunks |
| Query (hybrid retrieval only) | 100-300 ms | Embedding + Qdrant + BM25 + RRF |
| Query (full RAG with answer) | 3-8 seconds | Dominated by Gemini API call |
| Query (streaming, time to first token) | 1-3 seconds | Reranking + Gemini startup |
| BM25 rebuild on startup | 50-500 ms | Depends on collection size (scrolls all points from Qdrant) |
| Embedding model cold load | 2-5 seconds | First request only; cached thereafter |
| Reranker model cold load | 1-3 seconds | First request only; cached thereafter |

### Capacity Guidelines

- **Small deployment** (< 100 documents, < 5,000 chunks): Everything runs comfortably within free tiers.
- **Medium deployment** (100-1,000 documents, 5,000-50,000 chunks): BM25 index may use 50-500 MB RAM. Qdrant free tier still has ample space.
- **Large deployment** (> 1,000 documents): Consider upgrading Qdrant to a paid tier and running the embedder on GPU for faster ingestion.

---

## Troubleshooting

### Common Errors and Fixes

**Error: `"Unsupported file type '.docx'"` or similar**

Only PDF, TXT, and HTML files are supported. Convert other formats to one of these before uploading. For DOCX files, export to PDF from your word processor.

---

**Error: `"File too large. Maximum size is 10MB"`**

Increase the limit by setting `MAX_FILE_SIZE_MB` in your `.env` file, or split the file into smaller parts.

---

**Error: `"Could not extract text from file"`**

The PDF may be image-based (scanned document) without an embedded text layer. pypdf cannot extract text from images. Use an OCR tool (e.g., Tesseract) to add a text layer first.

---

**Error: Qdrant connection timeout or `"Connection refused"`**

- Verify your `QDRANT_URL` includes the port (typically `:6333`).
- Verify your `QDRANT_API_KEY` is correct.
- Check that your Qdrant Cloud cluster is active (free clusters may be paused after inactivity).

---

**Error: `"Gemini generation failed"` or `"429 Too Many Requests"`**

You have exceeded the Gemini API rate limit. RagCore has built-in rate limiting, but if multiple users are sharing the same API key, collisions can occur. Solutions:
- Wait a few seconds and retry.
- Reduce `GEMINI_RPM_LIMIT` to add more buffer between calls.
- Upgrade to a paid Gemini plan for higher limits.

---

**Error: `"Embedder initialization deferred"`**

This warning during startup means the embedding model could not be loaded immediately. This usually resolves on the first request. If it persists:
- Check internet connectivity (the model needs to be downloaded on first use).
- Ensure sufficient disk space (~200 MB for cached models).
- Check if the `EMBEDDING_MODEL` name is correct.

---

**BM25 index shows 0 documents after restart**

This is expected on first startup with a fresh Qdrant collection. The BM25 index rebuilds from Qdrant on startup. If Qdrant has data but BM25 shows 0, check the Qdrant connection settings.

---

**Gradio UI not loading or showing "Connecting..."**

- Ensure the server is running on port 7860 (or whichever port you configured).
- The Gradio UI communicates with the API via `http://localhost:7860`. If running in Docker, this internal URL is correct. If running behind a reverse proxy, the UI may need adjustment.

---

**Slow first request after startup**

The first request triggers lazy loading of the reranker model. This is a one-time cost of 1-3 seconds. Subsequent requests are fast.

---

**Docker build fails at model download step**

The Dockerfile pre-downloads ML models during build. This requires internet access during `docker build`. If building behind a corporate proxy, configure Docker's proxy settings. If the download fails, the build will fail. Retry usually resolves transient network issues.
