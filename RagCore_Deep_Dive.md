# RagCore — Deep Technical Reference & Interview Prep

This document explains every part of the RagCore system in depth. It's written so that even if you forget everything about this project, you can read this and fully understand it — and confidently answer any interview question about it.

---

## Table of Contents

1. [What is RAG and Why Does It Exist?](#1-what-is-rag-and-why-does-it-exist)
2. [System Architecture](#2-system-architecture)
3. [The Ingestion Pipeline (How Documents Get In)](#3-the-ingestion-pipeline)
4. [The Query Pipeline (How Questions Get Answered)](#4-the-query-pipeline)
5. [Module-by-Module Breakdown](#5-module-by-module-breakdown)
6. [How Streaming Works](#6-how-streaming-works)
7. [Deployment Architecture](#7-deployment-architecture)
8. [Design Patterns Used](#8-design-patterns-used)
9. [What I'd Improve for Production](#9-what-id-improve-for-production)
10. [Interview Questions (50+) with Answers](#10-interview-questions-with-answers)

---

## 1. What is RAG and Why Does It Exist?

### The Problem
Large Language Models (like GPT, Gemini) are trained on public data up to a cutoff date. They:
- **Don't know your private documents** (company reports, personal PDFs)
- **Hallucinate** — confidently make up facts when they don't know
- **Can't cite sources** — you don't know where the answer came from

### The Solution: RAG
**Retrieval-Augmented Generation** adds a retrieval step before generation:

```
Traditional LLM:  Question → LLM → Answer (might hallucinate)

RAG:  Question → SEARCH your documents → Found relevant passages →
      Give passages + question to LLM → Answer (grounded in YOUR data)
```

By feeding the LLM actual passages from your documents, it:
1. Answers based on **your** data, not training data
2. Can **cite sources** (we know which passages it used)
3. **Reduces hallucination** because the answer is constrained to the context

### Why This Project Matters
RagCore isn't a toy demo. It implements the same architecture used by production RAG systems:
- **Hybrid retrieval** (most production systems use this, not just vectors)
- **Reranking** (used by Google Search, Bing, Cohere)
- **Streaming** (real-time token delivery like ChatGPT)
- **Metadata filtering** (filter by date, type, source)

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
│                       (Port 7860)                            │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    Gradio UI (/)                       │  │
│  │   Ask Tab                    Documents Tab             │  │
│  │   • Query input              • File upload             │  │
│  │   • Doc type filter          • Document table          │  │
│  │   • Stream toggle            • Delete dropdown         │  │
│  │   • Example queries          • Upload & Index btn      │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │ httpx (internal HTTP calls)       │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │                  REST API Layer                        │  │
│  │                                                       │  │
│  │  POST /api/ingest    → Ingestion Pipeline             │  │
│  │  POST /api/ask       → Query Pipeline (+ streaming)   │  │
│  │  POST /api/search    → Search only (no LLM)           │  │
│  │  GET  /api/documents → List indexed docs              │  │
│  │  DELETE /api/documents/{id} → Remove doc + chunks     │  │
│  │  GET  /health        → Component status               │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Core Engine                         │  │
│  │                                                       │  │
│  │  QueryAnalyzer → HybridRetriever → Reranker →        │  │
│  │  AnswerGenerator → GeminiService                      │  │
│  │                                                       │  │
│  │  Embedder ─── BM25Index ─── Chunker ─── Parsers      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
             ┌──────────┐  ┌──────────┐  ┌──────────────┐
             │  Qdrant  │  │  Google   │  │  HuggingFace │
             │  Cloud   │  │  Gemini   │  │  Spaces      │
             │(vectors) │  │  (LLM)   │  │  (hosting)   │
             └──────────┘  └──────────┘  └──────────────┘
```

### Why FastAPI + Gradio Together?

Most tutorials show either FastAPI OR Gradio. We use both:
- **FastAPI** provides a proper REST API (testable with curl, Postman, any frontend)
- **Gradio** provides an instant UI without writing HTML/CSS/JS
- **They share the same process and port** — deployed as one container

```python
# In main.py — this is how they combine:
app = FastAPI(...)                                    # FastAPI handles /api/* routes
gradio_app = create_gradio_app()                      # Gradio creates the UI
app = gr.mount_gradio_app(app, gradio_app, path="/")  # Mount Gradio at root
```

When a request comes in:
- `/api/*` → FastAPI handles it
- Everything else → Gradio handles it (serves the web UI)

---

## 3. The Ingestion Pipeline

When a user uploads a document, this is what happens step by step:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Upload  │───▶│  Parse   │───▶│  Chunk   │───▶│  Embed   │───▶│  Store   │
│  File    │    │  PDF/HTML │    │  Text    │    │  Vectors │    │  Qdrant  │
│          │    │  /TXT    │    │  512w    │    │  384-dim │    │  + BM25  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Step 1: File Upload (ingest.py)
```
User drops file → FastAPI receives UploadFile → Validates:
  ✓ Has filename
  ✓ Extension is .pdf/.txt/.html/.htm
  ✓ File size ≤ 10MB (configurable)
  ✓ Not already indexed (409 Conflict if duplicate filename)
```

### Step 2: Parse (parsers.py)
Extracts raw text from the file based on its type:

| Format | Method | Notes |
|--------|--------|-------|
| **PDF** | `pypdf.PdfReader` | Extracts text page by page, joins with `\n\n` |
| **HTML** | `BeautifulSoup` | Removes `<script>`, `<style>`, `<nav>`, `<footer>` tags first |
| **TXT** | Direct decode | UTF-8, falls back to Latin-1 |

### Step 3: Extract Metadata (metadata.py)
From the raw text, we extract:
- **Title**: First non-empty line (max 200 chars)
- **Dates**: Scans first 2000 chars for date patterns (YYYY-MM-DD, MM/DD/YYYY, etc.)
- **Tags**: Capitalized phrases appearing 2+ times (e.g., "Machine Learning", "New York")
- **Doc type**: From file extension
- **Page count**: PDF only

This metadata is stored with every chunk — enabling filters like "search only PDFs from last month".

### Step 4: Chunk (chunker.py)
Splits the text into overlapping pieces:

```
Original text (5000 words)
    ↓
Split on sentence boundaries (regex: (?<=[.!?])\s+)
    ↓
Group into ~512-word chunks
    ↓
Overlap: last 50 words of chunk N become first 50 words of chunk N+1
    ↓
Result: ~11 chunks (for a 16-page PDF)
```

**Why chunk at all?** Embedding models have limited input (256-512 tokens). Also, smaller chunks mean more precise retrieval — instead of returning a whole chapter, we return the exact relevant paragraph.

**Why overlap?** Without overlap, important context at chunk boundaries is lost. A sentence like "As mentioned above, the results show..." would lose its reference if the "above" part is in the previous chunk.

**Why 512 words?** Sweet spot between:
- Too small (100w) → fragments lose context
- Too large (2000w) → retrieval becomes imprecise

### Step 5: Embed (embedder.py)
Each chunk is converted to a 384-dimensional vector using `all-MiniLM-L6-v2`:

```
"Machine learning is a subset of AI" → [0.023, -0.156, 0.891, ..., 0.044]  (384 numbers)
```

These vectors capture **semantic meaning**. Similar texts produce similar vectors:
- "The car is fast" ↔ "The automobile is speedy" → high similarity (≈0.85)
- "The car is fast" ↔ "The banana is yellow" → low similarity (≈0.15)

### Step 6: Store (vectorstore.py + bm25.py)
Chunks are stored in TWO places:

1. **Qdrant Cloud** — vector database for semantic (dense) search
   - Stores: vector + payload (text, document_id, source, doc_type, tags, dates)
   - Enables: "find chunks similar to this query vector"

2. **BM25 Index** — in-memory keyword index
   - Stores: tokenized text with term frequencies
   - Enables: "find chunks containing these exact words"

---

## 4. The Query Pipeline

When a user asks a question, this is what happens:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  User    │───▶│  Analyze │───▶│ Retrieve │───▶│  Fuse    │───▶│ Rerank   │───▶│ Generate │
│  Query   │    │  Intent  │    │  Dense + │    │  RRF     │    │ FlashRank│    │  Gemini  │
│          │    │  Filters │    │  Sparse  │    │  Merge   │    │  Top 5   │    │  Stream  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Step 1: Query Analysis (query_analyzer.py)

The analyzer extracts structured information from natural language:

```
Input:  "Summarize the PDFs from last week"
Output:
  intent: "summarize"
  doc_type: "pdf"
  date_from: 2026-03-10  (calculated from "last week")
  clean_query: "Summarize"  (filter phrases removed)
```

**What it detects:**

| Pattern | Example | Extracted |
|---------|---------|-----------|
| **Intent** | "Summarize...", "Compare X and Y" | summarize, comparative, list, explanatory, factual |
| **Doc type** | "in the PDFs", "from text files" | pdf, html, txt |
| **Dates** | "from last week", "after 2024-01-01" | date_from, date_to |
| **Source** | "from report.pdf" | source filename |

**Why analyze queries?** Without this, a query like "compare the PDFs" would search for the word "PDFs" instead of filtering by document type and using a comparison prompt.

### Step 2: Hybrid Retrieval (retriever.py)

Two search methods run in parallel:

**Dense Search (Qdrant)**
```
Query → Embed to vector → Find nearest vectors in Qdrant → Top 20 results
```
- Finds semantically similar chunks (meaning-based)
- Great for: "What are the key findings?" (no exact keyword match needed)
- Weak for: Acronyms, names, specific terms ("What does LSTM stand for?")

**Sparse Search (BM25)**
```
Query → Tokenize → Score documents by term frequency → Top 20 results
```
- Finds keyword-matching chunks (exact terms)
- Great for: "What is LSTM?" (exact term matching)
- Weak for: "What are the main takeaways?" (no exact keyword overlap)

**Why both?** Each method has blind spots. Dense misses exact terms, sparse misses semantic similarity. Together they cover both cases.

### Step 3: RRF Fusion (retriever.py → rrf_fuse)

Reciprocal Rank Fusion merges the two ranked lists into one:

```
Dense results:  [Doc_A (rank 1), Doc_C (rank 2), Doc_B (rank 3)]
Sparse results: [Doc_B (rank 1), Doc_A (rank 2), Doc_D (rank 3)]

RRF formula: score = weight × (1 / (k + rank))    where k = 60

Doc_A: dense_score = 0.6 × 1/(60+1) = 0.00984
        sparse_score = 0.4 × 1/(60+2) = 0.00645
        total = 0.01629  ← HIGHEST

Doc_B: dense_score = 0.6 × 1/(60+3) = 0.00952
        sparse_score = 0.4 × 1/(60+1) = 0.00656
        total = 0.01608

Doc_C: dense_score = 0.6 × 1/(60+2) = 0.00968
        sparse_score = 0.0 (not in sparse list)
        total = 0.00968

Doc_D: dense_score = 0.0 (not in dense list)
        sparse_score = 0.4 × 1/(60+3) = 0.00635
        total = 0.00635

Final ranking: [Doc_A, Doc_B, Doc_C, Doc_D]
```

**Why k=60?** It's a dampening factor. Higher k means rank differences matter less (Doc at rank 1 vs rank 5 scores similarly). 60 is the standard value from the original RRF paper.

**Why not just average the scores?** Dense scores (cosine similarity 0-1) and sparse scores (BM25, unbounded) are on different scales. RRF only uses ranks, not scores — no normalization needed.

### Step 4: Reranking (reranker.py)

The reranker takes the top 10 fused results and re-scores them:

```
Input:  10 chunks from RRF fusion
Process: For each chunk, score (query, chunk_text) pair with cross-encoder
Output: Top 5 chunks, reordered by cross-encoder score
```

**Why rerank?**
- **Bi-encoders** (embedding model) encode query and document SEPARATELY → fast but less precise
- **Cross-encoders** (reranker) encode query and document TOGETHER → slow but much more precise

```
Bi-encoder:   embed("What is ML?") → vec1,  embed("ML is a subset of AI") → vec2,  similarity(vec1, vec2)
Cross-encoder: score("What is ML?" + "ML is a subset of AI") → 0.92  (sees both at once!)
```

Cross-encoders are 100x slower, so we only apply them to the top-10 results (not all 289 chunks).

### Step 5: Answer Generation (generator.py + llm.py)

The top 5 reranked chunks become the context for the LLM:

```
SYSTEM PROMPT:
"You are a helpful assistant. Answer ONLY from the provided context.
Cite sources using [1], [2] etc. If the answer isn't in the context, say so."

CONTEXT:
[1] Source: report.pdf
Machine learning is a method of data analysis that automates...

[2] Source: notes.txt
The key difference between ML and traditional programming is...

QUESTION: What is machine learning?
```

The LLM generates an answer grounded in these passages, with citations like [1], [2].

**Streaming:** Instead of waiting for the full answer, we stream tokens as they're generated using Server-Sent Events (SSE). More on this in Section 6.

---

## 5. Module-by-Module Breakdown

### app/main.py — Application Entry Point

**What it does:** Creates the FastAPI app, wires everything together, handles startup/shutdown.

**Startup sequence (lifespan context manager):**
```
1. setup_logging()          → Configure log format and level
2. get_embedder()           → Load MiniLM-L6-v2 model (~400ms)
3. get_vectorstore()        → Connect to Qdrant Cloud
4. get_bm25()               → Create empty BM25 index
5. bm25.rebuild_from_vectorstore(vs) → Scroll all Qdrant docs → build BM25
6. Mount Gradio at "/"      → Web UI ready
7. yield                    → App is running
```

**Key design:** BM25 is in-memory, so on every restart it needs to rebuild from Qdrant. This takes ~500ms for ~300 documents.

---

### app/config.py — Configuration

**What it does:** Reads environment variables using pydantic-settings.

**How it works:**
```python
class Settings(BaseSettings):
    gemini_api_key: str = ""              # Read from GEMINI_API_KEY env var
    chunk_size: int = 512                 # Read from CHUNK_SIZE, default 512
    dense_weight: float = 0.6            # Read from DENSE_WEIGHT, default 0.6

# Priority: Environment variable > .env file > Default value
```

**Why pydantic-settings?** Type-safe configuration. If someone sets `CHUNK_SIZE=abc`, it fails immediately at startup instead of crashing later with a confusing error.

---

### app/core/embedder.py — Embedding Service

**What it does:** Converts text strings into 384-dimensional vectors.

**Key details:**
- Model: `all-MiniLM-L6-v2` (22M parameters, fast, good quality)
- Runs on CPU (no GPU required)
- Batch encoding with size 64 for memory efficiency
- Vectors are L2-normalized (unit length) for cosine similarity

**Singleton pattern:**
```python
_embedder = None  # Module-level global

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = EmbedderService(model_name="all-MiniLM-L6-v2")
    return _embedder
```

Every module uses this pattern — the model loads once and is reused for all requests.

---

### app/core/vectorstore.py — Qdrant Client

**What it does:** Stores and searches vector embeddings in Qdrant Cloud.

**Key operations:**
- `upsert_chunks()` — Store chunks with vectors and metadata payloads
- `search()` — Find nearest vectors with optional metadata filters
- `delete_document()` — Find all chunks for a doc, then delete by point IDs
- `scroll_all()` — Read all stored chunks (used to sync BM25 on startup)
- `get_document_ids()` — Aggregate chunks into document-level summary

**Payload indexes:**
```python
# These indexes make filtered queries fast (like SQL indexes):
"document_id": KEYWORD   # For deletion
"source": KEYWORD         # Filter by filename
"doc_type": KEYWORD       # Filter by pdf/txt/html
"tags": KEYWORD           # Filter by extracted tags
"created_date": KEYWORD   # Filter by date range
```

**Delete approach:** We first scroll to find all point IDs belonging to a document, then delete by ID list. This requires only "write" permission (not "manage"), which is important for Qdrant Cloud API key permissions.

---

### app/core/bm25.py — Keyword Index

**What it does:** Classic keyword search using BM25 (Best Match 25) algorithm.

**How BM25 scoring works:**
```
score(query, document) = Σ IDF(term) × (TF(term, doc) × (k₁ + 1)) / (TF(term, doc) + k₁ × (1 - b + b × |doc|/avgdl))
```

In plain English:
- **TF** (Term Frequency): How often does the query term appear in this document?
- **IDF** (Inverse Document Frequency): How rare is this term across all documents? (rare terms matter more)
- **Document length normalization**: Longer documents don't get unfair advantage

**Tokenization:**
```python
def tokenize(text):
    tokens = re.findall(r'\w+', text.lower())  # Split on word boundaries
    return [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]

# STOP_WORDS includes 49 common words: "a", "the", "is", "and", ...
```

**Why in-memory?** BM25 is fast in memory (~1ms for 300 docs). A persistent BM25 index would add complexity. The trade-off is that it rebuilds on every restart (~500ms from Qdrant).

---

### app/core/retriever.py — Hybrid Search

**What it does:** Runs dense + sparse search, then combines results with RRF.

**Key method: `retrieve()`**
```python
def retrieve(self, query, top_k=10, filters=None):
    # 1. Over-fetch 2x (we'll lose some to deduplication)
    fetch_k = top_k * 2  # 20 from each source

    # 2. Dense search via Qdrant
    query_vector = self.embedder.embed_query(query)
    dense_results = self.vectorstore.search(query_vector, limit=fetch_k, filters=filters)

    # 3. Sparse search via BM25
    sparse_results = self.bm25.search(query, top_k=fetch_k)
    if filters:
        sparse_results = self._apply_filters(sparse_results, filters)

    # 4. Fuse with RRF
    fused = self.rrf_fuse(
        [dense_results, sparse_results],
        weights=[0.6, 0.4]  # Dense weighted higher
    )

    return fused[:top_k]  # Return top 10
```

**Why 0.6/0.4 weighting?** Dense (semantic) search is generally more useful for natural language questions. BM25 is the safety net for exact terms. These weights are tunable via config.

---

### app/core/reranker.py — Cross-Encoder

**What it does:** Re-scores search results using FlashRank (ms-marco-MiniLM-L-12-v2).

**How cross-encoders work:**
```
Bi-encoder (embedder):     encode(query) → vec1    encode(doc) → vec2    dot(vec1, vec2)
Cross-encoder (reranker):  encode(query + " [SEP] " + doc) → scalar score
```

The cross-encoder sees both texts simultaneously, so it captures:
- Word interactions ("not good" → negative, which a bi-encoder might miss)
- Position-dependent relevance
- Query-specific term importance

**FlashRank** is a lightweight cross-encoder (~50MB model) that runs fast on CPU.

---

### app/core/query_analyzer.py — Query Understanding

**What it does:** Extracts intent and filters from natural language queries.

**Pattern matching (regex-based, not LLM):**
```python
# Document type detection
"in the PDFs"       → doc_type: "pdf"
"from text files"   → doc_type: "txt"

# Date extraction
"from last week"    → date_from: (today - 7 days)
"after 2024-01-01"  → date_from: 2024-01-01

# Source extraction
"from report.pdf"   → source: "report.pdf"

# Intent classification
"Summarize..."      → intent: "summarize"  (uses SUMMARY_PROMPT)
"Compare X and Y"   → intent: "comparative"
"What is...?"       → intent: "factual"    (uses SYSTEM_PROMPT)
```

**Why rules instead of an LLM?** Speed. An LLM call takes 1-3 seconds. Regex analysis takes <1ms. For the filters we need (date, type, source), rules are accurate enough.

---

### app/core/generator.py — Answer Generation

**What it does:** Builds the prompt, calls Gemini, and formats the response.

**Prompt construction:**
```python
def _build_context(self, chunks):
    context = ""
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        context += f"\n[{i}] Source: {source}\n{chunk.text}\n"
    return context

# Result:
# [1] Source: report.pdf
# Machine learning is a method of data analysis...
#
# [2] Source: notes.txt
# The key difference between ML and traditional...
```

**Two prompt modes:**
- `SYSTEM_PROMPT` (factual): "Answer ONLY from the provided context. Cite using [1], [2]."
- `SUMMARY_PROMPT` (summarize): "Provide a comprehensive summary of the context."

---

### app/core/llm.py — Gemini Service

**What it does:** Wraps the Google Gemini API with rate limiting.

**Rate limiting:**
```python
def _wait_for_rate_limit(self):
    elapsed = time.time() - self._last_call_time
    if elapsed < self._min_interval:      # min_interval = 60/15 = 4 seconds
        time.sleep(self._min_interval - elapsed)
    self._last_call_time = time.time()
```

With `rpm_limit=15`, there must be at least 4 seconds between API calls. This prevents 429 (Too Many Requests) errors.

---

### app/core/chunker.py — Text Chunking

**What it does:** Splits text into overlapping, sentence-aware chunks.

**Algorithm:**
```
1. Split text on sentence boundaries: (?<=[.!?])\s+
   "Hello world. This is a test. How are you?"
   → ["Hello world.", "This is a test.", "How are you?"]

2. Accumulate sentences until reaching ~512 words
   chunk_1 = sentences[0:15]   (512 words)
   chunk_2 = sentences[13:28]  (512 words, with 50-word overlap from chunk_1)
   chunk_3 = sentences[26:40]  (remaining)

3. Track character positions for each chunk
   chunk_1: start_char=0, end_char=2847
   chunk_2: start_char=2650, end_char=5492  (notice overlap)
```

---

### app/models/document.py — Data Models

**Core models:**

```python
class DocumentMetadata:
    source: str          # "report.pdf"
    doc_type: str        # "pdf"
    title: str | None    # "Q4 Financial Report"
    created_date: datetime | None
    tags: list[str]      # ["Machine Learning", "Neural Networks"]
    page_count: int | None

class Chunk:
    chunk_id: str        # UUID - unique per chunk
    document_id: str     # UUID - shared by all chunks from same doc
    text: str            # The actual chunk text
    metadata: DocumentMetadata
    chunk_index: int     # 0, 1, 2, ... (position in document)
    start_char: int      # Where this chunk starts in original text
    end_char: int        # Where it ends

class Document:
    document_id: str
    filename: str
    metadata: DocumentMetadata
    chunks: list[Chunk]
    raw_text: str        # Full original text
```

**Key insight:** Metadata is duplicated on every chunk. This seems wasteful, but it means we can filter chunks by metadata in Qdrant without a separate documents table.

---

### app/models/schemas.py — API Contracts

**Request/Response schemas for FastAPI:**

```python
# For POST /api/ask
class QueryRequest:
    query: str               # "What is machine learning?"
    top_k: int = 10         # Retrieve this many chunks
    rerank_top_k: int = 5   # Keep this many after reranking
    filters: SearchFilters   # Optional doc_type, source, date filters
    stream: bool = False     # Enable SSE streaming

# Response (non-streaming)
class GeneratedAnswer:
    query: str
    answer: str              # The LLM's response
    sources: list[RetrievedChunk]  # Which chunks were used
    generation_time_ms: float
    model: str               # "gemini-2.5-flash"
```

---

### app/utils/parsers.py — Document Parsing

**PDF parsing:**
```python
def parse_pdf(file_bytes, filename):
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    return text
```

**HTML parsing:**
```python
def parse_html(file_bytes, filename):
    soup = BeautifulSoup(file_bytes, "html.parser")
    # Remove noise: navigation, footers, scripts
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")
```

---

### app/api/deps.py — Dependency Injection

**What it does:** Connects FastAPI routes to service instances.

```python
# In routes, services are injected via Depends():
@router.post("/api/ask")
async def ask(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(dep_retriever),   # Auto-injected
    generator: AnswerGenerator = Depends(dep_generator),    # Auto-injected
    analyzer: QueryAnalyzer = Depends(dep_query_analyzer),  # Auto-injected
):
    ...
```

**Why dependency injection?**
- Routes don't create their own services → single source of truth
- Easy to test (swap real service with mock)
- Services are lazy-initialized (only created when first used)

---

### app/api/routes/query.py — Query Endpoints

**The most complex module.** Handles both normal and streaming responses.

**Filter resolution logic:**
```python
# Priority: User-specified filters > Analyzer-extracted filters > None
if request.filters and request.filters.has_filters():
    use request.filters
elif analyzed.extracted_filters and analyzed.extracted_filters.has_filters():
    use analyzed.extracted_filters
else:
    no filters
```

---

## 6. How Streaming Works

This is interview gold. Most candidates can't explain streaming end-to-end.

### The Problem
LLMs take 3-10 seconds to generate a full answer. Without streaming, the user stares at a spinner for all that time.

### The Solution: Server-Sent Events (SSE)

```
Timeline:
0ms    User clicks "Ask"
50ms   Query analyzed, retrieval starts
1500ms Retrieval + reranking done
1500ms First SSE event sent: data: {"text": "Based on"}
1600ms                       data: {"text": " the documents,"}
1700ms                       data: {"text": " machine learning"}
...
5000ms Final event:          data: {"done": true, "sources": [...]}
```

### Implementation Layers

**Layer 1: Gemini streams tokens (llm.py)**
```python
async def generate_stream(self, prompt):
    response = self.model.generate_content(prompt, stream=True)
    for chunk in response:
        yield chunk.text  # yields: "Based", " on", " the", ...
```

**Layer 2: Generator wraps with metadata (generator.py)**
```python
async def generate_answer_stream(self, query, chunks, ...):
    # First: rerank chunks
    reranked = self.reranker.rerank(query, chunks, top_k=5)

    # Build prompt with context
    prompt = self._build_prompt(query, reranked, intent)

    # Stream LLM tokens
    async for text_chunk in self.llm.generate_stream(prompt):
        yield text_chunk  # str: "Based on"

    # After streaming completes, yield metadata
    yield GeneratedAnswer(query=query, answer=full_text, sources=reranked, ...)
```

**Layer 3: Route formats as SSE (query.py)**
```python
async def _stream_response(generator, query, chunks, ...):
    async for item in generator.generate_answer_stream(query, chunks, ...):
        if isinstance(item, str):
            yield f"data: {json.dumps({'text': item})}\n\n"
        elif isinstance(item, GeneratedAnswer):
            yield f"data: {json.dumps({'done': True, 'sources': ..., 'time_ms': ...})}\n\n"

# FastAPI returns StreamingResponse
return StreamingResponse(
    _stream_response(...),
    media_type="text/event-stream"
)
```

**Layer 4: Gradio client reads SSE (gradio_app.py)**
```python
with httpx.stream("POST", "/api/ask", json=payload) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if "text" in data:
                answer += data["text"]
                yield f"<div class='answer-box'>{answer}</div>"  # Update UI
            if data.get("done"):
                # Show sources and timing
```

### Why SSE Instead of WebSockets?

| Feature | SSE | WebSocket |
|---------|-----|-----------|
| Direction | Server → Client only | Bidirectional |
| Setup | Standard HTTP | Upgrade handshake |
| Reconnect | Built-in | Manual |
| Complexity | Low | High |
| For our use case | Sufficient (server sends tokens) | Overkill |

---

## 7. Deployment Architecture

### Local Development
```
Your machine
├── Python process (uvicorn + FastAPI + Gradio)
├── .env file (secrets)
└── Calls external:
    ├── Qdrant Cloud (vectors)
    └── Google Gemini API (LLM)
```

### Docker
```
Docker container
├── Python 3.12-slim base image
├── Pre-downloaded models (embedding + reranker)
├── App code
├── Environment variables (from --env-file or HF Secrets)
└── Exposed port: 7860
```

### HuggingFace Spaces
```
HF Infrastructure
├── Reads README.md YAML frontmatter:
│   sdk: docker
│   app_port: 7860
├── Builds Docker image from your Dockerfile
├── Injects Repository Secrets as env vars
├── Runs container on shared GPU/CPU
└── Proxies HTTPS traffic → container port 7860
```

---

## 8. Design Patterns Used

### 1. Lazy Singleton
```python
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = EmbedderService(...)
    return _embedder
```
**Why:** Model loading is expensive. Load once, reuse forever.

### 2. Dependency Injection (FastAPI Depends)
```python
@router.post("/ask")
async def ask(retriever = Depends(dep_retriever)):
    ...
```
**Why:** Decouples route logic from service construction. Enables testing with mocks.

### 3. Strategy Pattern (Parser selection)
```python
PARSERS = {".pdf": parse_pdf, ".html": parse_html, ".txt": parse_text}
def parse_document(file_bytes, filename):
    ext = get_extension(filename)
    return PARSERS[ext](file_bytes, filename)
```
**Why:** Adding a new format (e.g., .docx) means adding one function and one dict entry.

### 4. Pipeline Pattern (Query processing)
```
Analyze → Retrieve → Fuse → Rerank → Generate
```
Each step has one job, takes input from the previous step, produces output for the next.

### 5. Context Manager (Timer)
```python
with timer("retrieval") as elapsed:
    results = retriever.retrieve(query)
    logger.info(f"Retrieved in {elapsed()}ms")
```

### 6. SSE Streaming (Generator functions)
```python
async def _stream_response(...):
    async for item in generator.generate_answer_stream(...):
        yield f"data: {json.dumps(...)}\n\n"
```
**Why:** Memory-efficient — doesn't buffer the entire response.

---

## 9. What I'd Improve for Production

| Area | Current | Production |
|------|---------|------------|
| **Auth** | None | API key or OAuth2 |
| **CORS** | `allow_origins=["*"]` | Specific domains only |
| **Rate limiting** | LLM-level only | Per-user request throttling |
| **BM25** | In-memory, rebuilds on restart | Persistent index (Elasticsearch) |
| **Monitoring** | Logs only | Prometheus + Grafana |
| **Error tracking** | Log files | Sentry |
| **Caching** | None | Redis for frequent queries |
| **File storage** | Qdrant payloads only | S3 for original files |
| **Testing** | Unit tests | + integration + load tests |
| **CI/CD** | Manual git push | GitHub Actions → HF Spaces |
| **Embedding model** | MiniLM (384d) | E5-large or Cohere embed (1024d) |
| **LLM** | Single provider (Gemini) | Fallback chain (Gemini → GPT → local) |

---

## 10. Interview Questions with Answers

### RAG Fundamentals

**Q1: What is RAG? Explain it to a non-technical person.**
> Imagine you have a huge filing cabinet of documents. RAG is like having a smart assistant who first FINDS the most relevant pages from your cabinet, then reads those specific pages to answer your question — instead of guessing from memory.

**Q2: Why not just put the entire document into the LLM prompt?**
> Two reasons: (1) LLMs have context limits (even 200K tokens isn't enough for a library of documents), and (2) retrieval focuses the LLM on the RELEVANT parts, reducing noise and hallucination. A 500-page manual stuffed into the prompt would dilute the answer.

**Q3: What's the difference between naive RAG and advanced RAG?**
> **Naive RAG:** embed chunks → vector search → LLM. Simple but has problems: misses keyword matches, no reranking, no query understanding.
> **Advanced RAG (what RagCore does):** query analysis → hybrid retrieval (dense + sparse) → fusion → reranking → prompt engineering with citations. Each step improves answer quality.

**Q4: What are the failure modes of RAG?**
> 1. **Retrieval failure** — relevant chunks not found (fix: hybrid search, better chunking)
> 2. **Context window pollution** — irrelevant chunks dilute the context (fix: reranking)
> 3. **LLM ignoring context** — model generates from training data instead of context (fix: prompt engineering)
> 4. **Chunking artifacts** — important info split across chunks (fix: overlap, larger chunks)
> 5. **Stale data** — documents updated but not re-indexed (fix: document versioning)

---

### Embeddings & Vectors

**Q5: What is a vector embedding?**
> A fixed-size array of numbers (384 floats for MiniLM) that represents the MEANING of text. "Happy" and "joyful" produce similar vectors. "Happy" and "banana" produce very different vectors. The model learns these representations from training on billions of text pairs.

**Q6: Why all-MiniLM-L6-v2? Why not a bigger model?**
> Trade-off: MiniLM is 22M parameters, produces 384-dim vectors, encodes in ~5ms per sentence on CPU. A bigger model (e.g., E5-large, 335M params, 1024-dim) would be 10x slower and require GPU. For a demo/small-scale system, MiniLM's quality is sufficient. In production with millions of docs, I'd upgrade.

**Q7: What is cosine similarity?**
> Measures the angle between two vectors, ignoring magnitude. Range: -1 (opposite) to 1 (identical).
> Formula: `cos(θ) = (A·B) / (|A| × |B|)`
> With normalized vectors (what MiniLM outputs), cosine similarity equals dot product.

**Q8: Why normalize embeddings?**
> So that cosine similarity = dot product. Dot product is faster to compute (no division). Also, normalized vectors live on a unit sphere — distance comparisons are more meaningful.

---

### Search & Retrieval

**Q9: Why hybrid search instead of just vector search?**
> Vector search is semantic — "automobile" matches "car". But it fails on:
> - Acronyms: "LSTM" won't match "Long Short-Term Memory" well
> - Names: "Dr. Smith" might match any doctor
> - Rare terms: Uncommon technical terms have poor embeddings
> BM25 catches these because it does exact keyword matching. Together, they cover each other's weaknesses.

**Q10: Explain BM25 in simple terms.**
> BM25 scores documents by: "How many of the query's words appear in this document, weighted by how RARE those words are?" The word "the" appearing 50 times doesn't help — but "mitochondria" appearing once is very informative. It also adjusts for document length so long documents don't unfairly dominate.

**Q11: What is Reciprocal Rank Fusion? Why not just concatenate results?**
> RRF assigns scores based on RANK position, not raw scores. This is important because dense scores (0-1 cosine) and sparse scores (unbounded BM25) can't be compared directly.
> RRF formula: `score = Σ weight × 1/(k + rank)`. A document ranked #1 in both lists gets a higher combined score than one ranked #1 in only one list.
> Simple concatenation would just merge lists without considering cross-list agreement.

**Q12: What does the reranker add that retrieval doesn't?**
> Retrieval uses bi-encoders (encode query and doc separately → compare). This is fast but misses:
> - Negation: "not effective" vs "effective"
> - Position importance: which part of the query matches which part of the doc
> - Cross-attention: subtle relationships between query terms and doc terms
> The cross-encoder sees both together, so it catches these nuances. It's 100x slower, which is why we only apply it to top-10 results.

---

### FastAPI & Backend

**Q13: Why FastAPI over Flask or Django?**
> 1. **Async native** — our app calls external APIs (Qdrant, Gemini). Async means we don't block the server while waiting for responses.
> 2. **Automatic OpenAPI docs** — free Swagger UI at `/docs`
> 3. **Pydantic validation** — request/response schemas validated automatically
> 4. **Performance** — one of the fastest Python frameworks

**Q14: Explain the lifespan context manager.**
> It replaces `@app.on_event("startup/shutdown")` (deprecated). Code before `yield` runs on startup, after `yield` on shutdown. We use it to pre-load the embedding model, connect to Qdrant, and build the BM25 index — so the first request doesn't have cold-start latency.

**Q15: How does dependency injection work in your app?**
> FastAPI's `Depends()` mechanism. Route functions declare what they need as parameters:
> ```python
> async def ask(retriever = Depends(dep_retriever)):
> ```
> FastAPI calls `dep_retriever()` before each request and passes the result. This means routes don't know HOW services are created — they just use them.

**Q16: How do you handle the streaming response?**
> 1. Route returns `StreamingResponse(generator, media_type="text/event-stream")`
> 2. FastAPI sends each `yield` immediately (doesn't buffer)
> 3. Generator yields `data: {"text": "chunk"}\n\n` for each LLM token
> 4. Final yield includes `{"done": true, "sources": [...]}` with metadata
> 5. Client reads line-by-line and updates the UI progressively

---

### Docker & Deployment

**Q17: Walk me through your Dockerfile.**
> ```
> FROM python:3.12-slim         # Minimal Python base image
> WORKDIR /app                  # Set working directory
> RUN apt-get install build-essential  # C compiler for some pip packages
> COPY requirements.txt .       # Copy deps FIRST (for layer caching)
> RUN pip install ...           # Install Python packages (cached if unchanged)
> RUN python -c "... SentenceTransformer('all-MiniLM-L6-v2')"  # Pre-download models
> COPY . .                      # Copy app code (rebuilds on every change)
> CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
> ```
> The order matters for layer caching: requirements and models rarely change, so those layers are cached. Only the `COPY . .` layer rebuilds when code changes.

**Q18: How does HF Spaces deployment work?**
> 1. Push code to HF Spaces git repo (every Space IS a git repo)
> 2. HF reads README.md YAML: `sdk: docker`, `app_port: 7860`
> 3. HF builds the Docker image from the Dockerfile
> 4. HF injects Repository Secrets as environment variables
> 5. HF runs the container and proxies HTTPS → port 7860
> 6. Any `git push` triggers a new build automatically

**Q19: How do you handle secrets?**
> - Local: `.env` file (in `.gitignore`, never committed)
> - Docker: `docker run --env-file .env`
> - HF Spaces: Repository Secrets (encrypted, injected as env vars)
> - Code: `pydantic-settings` reads env vars automatically with type validation
> - Safety: `.dockerignore` also excludes `.env` from Docker images

---

### System Design & Architecture

**Q20: If the BM25 index is in-memory, what happens on restart?**
> It rebuilds from Qdrant. On startup, we `scroll_all()` from Qdrant (reads every stored chunk), then rebuild the BM25 index. For 300 docs this takes ~500ms. For 100K docs, I'd switch to Elasticsearch or a persistent sparse index.

**Q21: What happens if Qdrant is down?**
> The health endpoint reports "degraded". Ingestion and search fail with 500 errors. The Gradio UI still loads (it's served from the same process), but queries won't work. In production, I'd add a circuit breaker and fallback to BM25-only search.

**Q22: How would you scale this to 1 million documents?**
> 1. **BM25**: Replace in-memory with Elasticsearch (persistent, distributed)
> 2. **Embeddings**: Use GPU and batch processing (not real-time)
> 3. **Qdrant**: Use multiple shards and replicas
> 4. **Chunking**: Run as async background job (not in request path)
> 5. **Caching**: Redis for frequent queries
> 6. **Workers**: Multiple uvicorn workers behind a load balancer

**Q23: Why does the Gradio UI call localhost via httpx instead of calling Python functions directly?**
> Separation of concerns. The UI communicates via HTTP (same as curl, Postman, or a mobile app). This means:
> 1. The API is testable independently
> 2. The UI could be replaced without changing the backend
> 3. The same API serves multiple clients
> 4. Gradio's streaming uses httpx.stream, which naturally works with SSE

---

### Chunking & Text Processing

**Q24: Why 512-word chunks with 50-word overlap?**
> **512 words**: Fits within the embedding model's context window (~256 tokens ≈ ~400 words). Also long enough to capture a complete thought or paragraph, but short enough for precise retrieval.
> **50-word overlap**: ~2-3 sentences. Ensures that important context at boundaries isn't lost. Without overlap, a sentence like "As mentioned above..." loses its reference.

**Q25: Why sentence-aware splitting?**
> Splitting at character count 512 could break mid-sentence: "Machine learning is a meth..." / "od of data analysis." Neither fragment is useful alone. Sentence-aware splitting finds the nearest sentence boundary, keeping each chunk semantically coherent.

**Q26: What would you do differently for code documents?**
> Code has different structure than prose:
> - Split by function/class boundaries (AST parsing)
> - Use code-specific embedding model (CodeBERT, StarCoder embeddings)
> - Keep indentation and context (imports, class hierarchy)
> - Larger chunks (functions can be 50+ lines)

---

### LLM & Generation

**Q27: Why Gemini 2.5 Flash specifically?**
> Fast, cheap, good quality, and supports streaming. "Flash" models are optimized for speed (vs. "Pro" for quality). For RAG, the LLM's job is mostly formatting and synthesizing already-retrieved context — it doesn't need to be the smartest model, just fast and reliable.

**Q28: How do you prevent hallucination?**
> 1. **Prompt engineering**: "Answer ONLY from the provided context. If the answer isn't there, say so."
> 2. **Citation requirement**: Force the model to cite [1], [2] — makes it harder to make things up
> 3. **Limited context**: Only top-5 reranked chunks (not everything) — less noise
> 4. **Temperature 0.3**: Low temperature = more deterministic, less creative/hallucinatory

**Q29: What's the difference between your SYSTEM_PROMPT and SUMMARY_PROMPT?**
> `SYSTEM_PROMPT` (factual): Strict citation mode. "Answer from context only, cite sources."
> `SUMMARY_PROMPT` (summarize): Broader synthesis. "Provide a comprehensive summary of the key themes."
> The query analyzer detects intent and selects the appropriate prompt.

**Q30: How does rate limiting work in your LLM service?**
> Simple token bucket: track the time of the last API call. If less than `60/rpm_limit` seconds have passed, sleep for the difference. Default is 15 RPM → 4 seconds between calls. This prevents 429 errors from Google's API.

---

### Metadata & Filtering

**Q31: How does metadata filtering work in Qdrant?**
> Qdrant supports payload filters on indexed fields:
> ```python
> filter = Filter(must=[
>     FieldCondition(key="doc_type", match=MatchValue(value="pdf")),
>     FieldCondition(key="created_date", range=Range(gte="2024-01-01")),
> ])
> results = client.search(vector=query_vec, query_filter=filter)
> ```
> This happens DURING vector search (not after), so it's efficient. The payload indexes we create at startup make these filters fast.

**Q32: How do you extract tags from documents?**
> Find capitalized phrases (e.g., "Machine Learning", "New York") that appear 2+ times. Regex: consecutive capitalized words. Frequency threshold filters out one-off mentions. Return top 10. It's heuristic, not ML-based — good enough for basic categorization.

---

### Security & Error Handling

**Q33: What security concerns does this app have?**
> 1. **CORS `*`** — anyone can call the API (fine for demo, restrict in prod)
> 2. **No auth** — anyone can upload/delete documents
> 3. **File upload** — potential for malicious files (mitigated by size limit and parsing only)
> 4. **API keys in env** — must never be committed to git
> 5. **Prompt injection** — user could craft queries to manipulate the LLM
> 6. **SSRF** — httpx calls to localhost could be exploited if user controls URLs

**Q34: How do you handle errors gracefully?**
> - Startup: try/except around each service initialization. If embedder fails, log warning and continue (other features may still work)
> - Ingestion: Validate file type, size, and content before processing. Return specific HTTP codes (400, 409, 413, 422, 500)
> - Query: Streaming errors are caught and sent as `data: {"error": "msg", "done": true}` so the client handles it gracefully
> - Health: `/health` endpoint reports component-level status ("ok"/"degraded")

---

### Scenario-Based Questions

**Q35: A user says "search is returning irrelevant results." How do you debug?**
> 1. Check which path is failing: add logging to see dense vs sparse scores
> 2. Look at the query analyzer output — is it extracting wrong filters?
> 3. Check chunk quality — are chunks too large/small? Open the doc and see
> 4. Look at the embedding similarity scores — if all are low (<0.3), the query might be too different from the document content
> 5. Check if the reranker is helping — compare pre/post rerank ordering
> 6. Try the query on `/api/search` (no LLM) to isolate retrieval from generation

**Q36: The app is slow. Where's the bottleneck?**
> Measure each stage:
> ```
> Query analysis:  <1ms    (regex, instant)
> Embedding:       ~5ms    (single query, fast)
> Qdrant search:   ~200ms  (network round-trip to cloud)
> BM25 search:     ~1ms    (in-memory)
> RRF fusion:      <1ms    (simple math)
> Reranking:       ~300ms  (cross-encoder on CPU, 10 pairs)
> LLM generation:  2-5s    (Gemini API, depends on response length)
> ```
> The LLM is always the bottleneck. Options: use a faster model, reduce context length, or cache frequent queries.

**Q37: How would you add DOCX support?**
> 1. `pip install python-docx`, add to requirements.txt
> 2. Add `parse_docx()` in parsers.py using `docx.Document` API
> 3. Add `".docx"` to `SUPPORTED_EXTENSIONS`
> 4. Add `".docx"` to Gradio file_types list
> 5. Rebuild Docker image (may need system libs for python-docx)
> Total: ~30 minutes of work

**Q38: A document was uploaded but questions about it return no results. Why?**
> Possible causes:
> 1. **BM25 not synced** — check if BM25 doc count matches Qdrant count
> 2. **Embedding mismatch** — query and doc embedded with different models
> 3. **Chunk too small** — text might have been poorly extracted (check parser output)
> 4. **Filters active** — doc_type filter might exclude it
> 5. **Low similarity** — the question is phrased very differently from the document content
> Debug: Call `/api/search` directly with no filters, check if chunks appear

**Q39: What if Gemini changes their API or model gets deprecated?**
> Already happened! `gemini-2.0-flash` was deprecated mid-project. Fix:
> 1. Update model name in `.env` and `config.py`
> 2. The error was caught and logged clearly
> 3. For production: abstract the LLM behind an interface, support multiple providers, add fallback chain

**Q40: How would you add user authentication?**
> 1. Add `python-jose` and `passlib` to requirements
> 2. Create a `/auth/login` endpoint that returns JWT tokens
> 3. Add a `get_current_user` dependency that validates JWT
> 4. Add `Depends(get_current_user)` to protected routes
> 5. Scope documents per user (add `user_id` to metadata)
> 6. Filter queries by user's documents only

---

### Advanced / Senior-Level Questions

**Q41: Compare your chunking strategy to alternatives.**
> | Strategy | Pros | Cons |
> |----------|------|------|
> | **Fixed-size (ours)** | Simple, predictable | May split semantic units |
> | **Recursive character** | Better boundary detection | More complex |
> | **Semantic chunking** | Groups by meaning | Requires embeddings (slow) |
> | **Document-structure** | Respects headers/sections | Format-specific |
> | **Sliding window** | Dense overlap | High redundancy |
>
> We chose fixed-size with sentence awareness as the best simplicity/quality trade-off.

**Q42: Why RRF over other fusion methods?**
> | Method | Pros | Cons |
> |--------|------|------|
> | **RRF** | No score normalization needed, simple | Ignores score magnitude |
> | **CombSUM** | Uses actual scores | Needs score normalization |
> | **CombMNZ** | Rewards multi-list hits | Complex normalization |
> | **Learned fusion** | Optimal weights | Needs training data |
>
> RRF is the industry standard because it's simple, effective, and doesn't require tuning score distributions.

**Q43: What's the theoretical maximum recall of your system?**
> We fetch top-20 from each retriever (dense and sparse), then fuse. If a relevant chunk isn't in the top-20 of EITHER method, it's lost. For typical queries, this gives ~95%+ recall on relevant content. Edge case: a chunk that's neither semantically similar NOR keyword-matched is unreachable.

**Q44: How would you evaluate RAG quality systematically?**
> 1. **Retrieval quality**: Precision@K, Recall@K, MRR (Mean Reciprocal Rank) — using labeled query-document pairs
> 2. **Reranking quality**: NDCG (Normalized Discounted Cumulative Gain) — are the best results at the top?
> 3. **Generation quality**: BLEU/ROUGE (automated), human evaluation (faithfulness, relevance, completeness)
> 4. **End-to-end**: Build a test set of questions with known answers, measure exact match and semantic similarity

**Q45: Explain the trade-off between chunk size and retrieval quality.**
> - **Small chunks (100 words)**: High precision (exact paragraph), low context (answer might span multiple chunks)
> - **Large chunks (2000 words)**: High context (full section), low precision (lots of irrelevant text in the chunk)
> - **Sweet spot (512 words)**: Enough context for a complete thought, small enough for precise retrieval
> - The overlap (50 words) helps by ensuring boundary content appears in at least one chunk fully

**Q46: What happens to your BM25 index if two users upload simultaneously?**
> BM25 index uses a Python list and dict (not thread-safe). Concurrent writes could corrupt the index. Fixes:
> 1. Add a threading.Lock around index modifications
> 2. Use asyncio.Lock for async context
> 3. Queue uploads and process sequentially
> 4. In production: use Elasticsearch instead (handles concurrency natively)

**Q47: How does Qdrant's HNSW index work internally?**
> HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor algorithm:
> 1. Builds a multi-layer graph of vectors
> 2. Top layers have few nodes (coarse navigation)
> 3. Bottom layers have all nodes (fine search)
> 4. Search: start at top layer, greedily navigate to nearest neighbor, drop to next layer, repeat
> 5. Time complexity: O(log N) vs brute force O(N)
> Trade-off: Uses more memory (graph edges) for much faster search.

**Q48: If you had to build this without Qdrant, what would you use?**
> Options by complexity:
> 1. **FAISS** (Facebook) — in-memory, fast, but no cloud persistence
> 2. **ChromaDB** — simpler API, local-first, good for prototyping
> 3. **Pinecone** — managed cloud, similar to Qdrant Cloud
> 4. **pgvector** — PostgreSQL extension, good if already using Postgres
> 5. **Weaviate** — full-featured, built-in hybrid search
> I'd choose pgvector for simplicity if I already have a Postgres DB, or Pinecone for managed cloud.

**Q49: What's the cost of running this system?**
> - **Qdrant Cloud**: Free tier (1GB, 1M vectors) — sufficient for demo
> - **Gemini API**: Free tier (15 RPM, 1M tokens/day) — sufficient for demo
> - **HF Spaces**: Free tier (2 vCPU, 16GB RAM) — sufficient but slow
> - **Production estimate**: ~$50-200/month (paid Qdrant + Gemini + GPU instance)

**Q50: How would you add multi-language support?**
> 1. Use a multilingual embedding model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`)
> 2. BM25 tokenization needs language-specific stop words
> 3. Query analyzer date patterns need locale awareness
> 4. LLM prompt should specify: "Answer in the same language as the question"
> 5. Consider: translate queries to English for search, then translate answers back

---

## Quick Revision Checklist

Before an interview, review these key concepts:

- [ ] RAG = Retrieval + Augmented + Generation
- [ ] Hybrid search = Dense (vectors, semantic) + Sparse (BM25, keywords)
- [ ] RRF = Reciprocal Rank Fusion (combines ranked lists by rank position)
- [ ] Cross-encoder reranker sees query+doc together (more accurate, slower)
- [ ] Bi-encoder embedder encodes separately (fast, less accurate)
- [ ] Chunking: 512 words, 50 overlap, sentence-aware boundaries
- [ ] Streaming: SSE (Server-Sent Events) over HTTP
- [ ] FastAPI + Gradio on same port, Gradio at "/", API at "/api/*"
- [ ] Docker layer caching: COPY requirements BEFORE COPY code
- [ ] Secrets: .env locally, HF Secrets in production, never in git
- [ ] Singleton pattern for expensive resources (models, DB connections)
- [ ] Dependency injection via FastAPI Depends()
