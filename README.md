# RagCore

A production-ready Retrieval-Augmented Generation (RAG) system that lets you upload documents (PDF, TXT, HTML) and ask questions about them using natural language. It combines hybrid search (dense vectors + BM25 keywords), cross-encoder reranking, and Google Gemini to generate grounded, citation-backed answers.

**Live Demo:** [huggingface.co/spaces/NinjainPJs/RagCore](https://huggingface.co/spaces/NinjainPJs/RagCore)

---

## Screenshots

### Ask Tab — Query your documents with natural language
![Ask Tab](https://github.com/ninjacode911/Project-RagCore/blob/main/Pictures/1.png)

### Documents Tab — Upload, index, and manage documents
![Documents Tab](https://github.com/ninjacode911/Project-RagCore/blob/main/Pictures/2.png)

### Streaming Response — Real-time answer generation with source citations
![Streaming Response](https://github.com/ninjacode911/Project-RagCore/blob/main/Pictures/3.png)

---

## How It Works

Unlike naive RAG that relies only on vector similarity, RagCore uses a multi-stage pipeline:

```
                         ┌──────────────────┐
                         │   User Question   │
                         └────────┬─────────┘
                                  ▼
                     ┌────────────────────────┐
                     │   Query Analyzer        │
                     │   • Detects intent      │
                     │   • Extracts filters    │
                     │     (date, type, source)│
                     └────────┬───────────────┘
                              ▼
              ┌───────────────┴───────────────┐
              ▼                               ▼
   ┌─────────────────────┐       ┌─────────────────────┐
   │  Dense Retrieval     │       │  Sparse Retrieval    │
   │  (Qdrant Vectors)    │       │  (BM25 Keywords)     │
   │  Semantic meaning    │       │  Exact term matching  │
   └──────────┬──────────┘       └──────────┬──────────┘
              │                              │
              └──────────┬───────────────────┘
                         ▼
              ┌──────────────────────┐
              │  Reciprocal Rank     │
              │  Fusion (RRF)        │
              │  Merges both lists   │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  FlashRank Reranker  │
              │  Cross-encoder       │
              │  re-scores top 10    │
              │  → keeps top 5       │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  Google Gemini LLM   │
              │  Generates answer    │
              │  with [1][2] source  │
              │  citations           │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  Streaming Response  │
              │  (Server-Sent Events)│
              └──────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI + Uvicorn | Async REST API with streaming support |
| **Frontend** | Gradio | Interactive web UI mounted inside FastAPI |
| **Vector DB** | Qdrant Cloud | Dense vector storage and similarity search |
| **Embeddings** | all-MiniLM-L6-v2 | 384-dim sentence embeddings via sentence-transformers |
| **Keyword Search** | BM25 (rank_bm25) | Sparse retrieval for exact term matching |
| **Reranker** | FlashRank | Cross-encoder reranking (ms-marco-MiniLM-L-12-v2) |
| **LLM** | Google Gemini 2.5 Flash | Answer generation with streaming |
| **Parsing** | pypdf, BeautifulSoup | PDF, HTML, TXT document parsing |
| **Deployment** | Docker, HuggingFace Spaces | Containerized cloud deployment |

---

## Project Structure

```
ragcore/
├── app/
│   ├── main.py                    # FastAPI app + Gradio mount + startup
│   ├── config.py                  # Pydantic settings (env vars)
│   ├── api/
│   │   ├── deps.py                # Dependency injection
│   │   └── routes/
│   │       ├── health.py          # GET /health
│   │       ├── ingest.py          # POST /api/ingest, GET/DELETE /api/documents
│   │       └── query.py           # POST /api/search, POST /api/ask (streaming)
│   ├── core/
│   │   ├── embedder.py            # Sentence-transformers wrapper
│   │   ├── vectorstore.py         # Qdrant client wrapper
│   │   ├── bm25.py                # BM25 keyword index
│   │   ├── retriever.py           # Hybrid search + RRF fusion
│   │   ├── reranker.py            # FlashRank cross-encoder
│   │   ├── query_analyzer.py      # Intent detection + filter extraction
│   │   ├── generator.py           # Prompt building + answer generation
│   │   ├── llm.py                 # Gemini API with rate limiting
│   │   ├── chunker.py             # Sentence-aware text chunking
│   │   └── metadata.py            # Title/date/tag extraction
│   ├── models/
│   │   ├── document.py            # Chunk, Document, DocumentMetadata
│   │   └── schemas.py             # API request/response schemas
│   ├── ui/
│   │   └── gradio_app.py          # Gradio web interface
│   └── utils/
│       ├── parsers.py             # PDF/HTML/TXT parsing
│       └── helpers.py             # ID generation, text cleaning, retry
├── tests/                         # Test suite
├── Dockerfile                     # Production container
├── docker-compose.yml             # Local dev setup
├── requirements.txt               # Python dependencies
└── .env                           # Environment variables (not committed)
```

---

## Setup

### Prerequisites
- Python 3.10+
- [Qdrant Cloud](https://cloud.qdrant.io/) account (free tier works)
- [Google Gemini API key](https://aistudio.google.com/apikey)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/Project-RagCore.git
cd Project-RagCore/ragcore

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the `ragcore/` directory:

```env
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# Optional (these have sensible defaults)
EMBEDDING_MODEL=all-MiniLM-L6-v2
QDRANT_COLLECTION=ragcore_docs
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=10
RERANK_TOP_K=5
GEMINI_MODEL=gemini-2.5-flash
LOG_LEVEL=INFO
```

### 3. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Open http://localhost:7860 in your browser.

### Docker

```bash
docker build -t ragcore .
docker run --env-file .env -p 7860:7860 ragcore
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with component status |
| `POST` | `/api/ingest` | Upload and index a document (multipart file) |
| `GET` | `/api/documents` | List all indexed documents |
| `DELETE` | `/api/documents/{id}` | Delete a document and its chunks |
| `POST` | `/api/search` | Hybrid search (returns chunks) |
| `POST` | `/api/ask` | Ask a question (supports streaming) |

### Example: Ask a question

```bash
curl -X POST http://localhost:7860/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "top_k": 10,
    "rerank_top_k": 5,
    "stream": true
  }'
```

Streaming response (SSE):
```
data: {"text": "Based on the documents, "}
data: {"text": "the key findings include..."}
data: {"done": true, "sources": [...], "model": "gemini-2.5-flash", "time_ms": 3420}
```

---

## Deployment on HuggingFace Spaces

1. Create a Space with **Docker** SDK at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Push the `ragcore/` folder:
   ```bash
   cd ragcore
   git init
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/RagCore
   git add -A && git commit -m "Initial deploy"
   git push origin master:main
   ```
3. Add secrets in Space Settings: `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`

The Dockerfile handles everything — installs dependencies, pre-downloads ML models, and starts the server on port 7860.

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **Hybrid search (dense + BM25)** | Dense catches semantic similarity ("car" ↔ "automobile"), BM25 catches exact terms and acronyms that embeddings miss |
| **RRF fusion** | Combines ranked lists without needing score normalization — simple and effective |
| **Cross-encoder reranker** | More accurate than bi-encoder for final ranking, but slower — so only applied to top-K |
| **Sentence-aware chunking** | Avoids breaking mid-sentence; overlap prevents lost context at boundaries |
| **In-memory BM25** | Fast keyword search; synced from Qdrant on startup. Trade-off: rebuilds on restart |
| **Streaming SSE** | Real-time token delivery without WebSocket complexity; works with any HTTP client |
| **Gradio inside FastAPI** | Single process, single port — simplifies deployment while keeping a proper REST API |

---

## License

MIT
