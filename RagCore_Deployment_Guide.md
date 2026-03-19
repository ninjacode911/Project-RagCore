# RagCore - Complete Deployment Guide & Learning Reference

## Table of Contents
1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Local Development Setup](#2-local-development-setup)
3. [Hugging Face Spaces Deployment](#3-hugging-face-spaces-deployment)
4. [Every Command Explained](#4-every-command-explained)
5. [Docker Deep Dive](#5-docker-deep-dive)
6. [Git & Version Control](#6-git--version-control)
7. [Environment Variables & Secrets](#7-environment-variables--secrets)
8. [Troubleshooting Log](#8-troubleshooting-log)
9. [Interview Questions & Answers](#9-interview-questions--answers)

---

## 1. Project Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    USER'S BROWSER                        │
│              (Gradio Frontend - HTML/JS)                  │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP Requests
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Server (Port 7860)              │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │ /api/ingest  │  │ /api/ask     │  │ /api/documents│   │
│  │ (Upload docs)│  │ (Query RAG)  │  │ (List/Delete) │   │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘   │
│         │                │                   │           │
│  ┌──────▼────────────────▼───────────────────▼───────┐   │
│  │              Core RAG Pipeline                     │   │
│  │                                                    │   │
│  │  PDF/TXT/HTML Parser → Chunker → Embedder          │   │
│  │       ↓                                            │   │
│  │  Vector Store (Qdrant Cloud) + BM25 Index          │   │
│  │       ↓                                            │   │
│  │  Hybrid Retriever → FlashRank Reranker             │   │
│  │       ↓                                            │   │
│  │  Google Gemini LLM → Streaming Response            │   │
│  └────────────────────────────────────────────────────┘   │
│                                                          │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Gradio UI (mounted at "/" path)                    │   │
│  │  - Ask Tab: Query documents                         │   │
│  │  - Documents Tab: Upload/manage files               │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                       │
          External Services (Cloud)
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │  Qdrant  │  │  Google   │  │ HuggingFace│
   │  Cloud   │  │  Gemini   │  │  (Hosting) │
   │(Vectors) │  │  (LLM)   │  │            │
   └──────────┘  └──────────┘  └──────────┘
```

### Key Components
- **FastAPI**: Python web framework that serves both the REST API and the Gradio UI
- **Gradio**: Creates the web interface, mounted inside FastAPI
- **Qdrant Cloud**: Vector database storing document embeddings
- **Google Gemini**: LLM that generates answers from retrieved context
- **FlashRank**: Cross-encoder model that reranks search results
- **BM25**: Keyword-based search (sparse retrieval) complementing vector search (dense retrieval)
- **Sentence Transformers**: Creates vector embeddings from text chunks

---

## 2. Local Development Setup

### Prerequisites
```bash
# Check Python version (need 3.10+)
python --version

# Check pip
pip --version
```

### Install Dependencies
```bash
# Navigate to project
cd C:\Projects\Project-RagCore\ragcore

# Create virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate it (Git Bash / Linux / Mac)
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Configure Environment
```bash
# Create .env file with your API keys
# NEVER commit this file (it's in .gitignore)
```

`.env` contents:
```
GEMINI_API_KEY=your_gemini_key_here
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
EMBEDDING_MODEL=all-MiniLM-L6-v2
QDRANT_COLLECTION=ragcore_docs
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=10
RERANK_TOP_K=5
GEMINI_MODEL=gemini-2.5-flash
LOG_LEVEL=INFO
```

### Run Locally
```bash
# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 7860

# What this does:
#   uvicorn          → ASGI server (runs async Python web apps)
#   app.main:app     → module path "app/main.py", variable "app" (the FastAPI instance)
#   --host 0.0.0.0   → listen on all network interfaces (not just localhost)
#   --port 7860      → port number (7860 is HF Spaces default)
```

### Run with Docker Locally
```bash
# Build the Docker image
docker build -t ragcore .

# What this does:
#   docker build     → build a Docker image from the Dockerfile
#   -t ragcore       → tag/name the image as "ragcore"
#   .                → use current directory as build context

# Run the container
docker run --env-file .env -p 8000:7860 ragcore

# What this does:
#   docker run       → create and start a container from the image
#   --env-file .env  → load environment variables from .env file
#   -p 8000:7860     → map host port 8000 → container port 7860
#   ragcore          → the image to run

# Or use docker-compose (reads docker-compose.yml)
docker compose up

# What this does:
#   Reads docker-compose.yml which defines:
#   - Which image to build
#   - Port mapping
#   - Environment file
#   - Other configuration
```

---

## 3. Hugging Face Spaces Deployment

### Step 1: Install HF CLI
```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"

# What this does:
#   -ExecutionPolicy ByPass  → allow running downloaded scripts
#   irm                      → Invoke-RestMethod (download the script)
#   |                        → pipe the downloaded script to...
#   iex                      → Invoke-Expression (execute it)
#
# The script:
#   1. Checks for Python 3.9+
#   2. Creates a virtual environment at ~/.hf-cli
#   3. Installs the huggingface_hub Python package
#   4. Creates hf.exe launcher at ~/.local/bin/
#   5. Adds to PATH if needed
```

### Step 2: Authenticate
```bash
# Login to Hugging Face
hf auth login --token YOUR_HF_TOKEN --add-to-git-credential

# What this does:
#   hf auth login              → start the login process
#   --token YOUR_HF_TOKEN      → use this access token (from hf.co/settings/tokens)
#   --add-to-git-credential    → save token in git's credential manager
#                                 so git push to HF repos works automatically
#
# Where tokens are saved:
#   ~/.cache/huggingface/token           → main token file
#   ~/.cache/huggingface/stored_tokens   → named token storage
#   git credential manager               → for git operations
```

### Step 3: Create a Space
```bash
# Option A: Via CLI
hf repo create YOUR_USERNAME/RagCore --type space --space-sdk docker

# What this does:
#   hf repo create           → create a new repository on HuggingFace
#   YOUR_USERNAME/RagCore    → namespace/repo-name
#   --type space             → it's a Space (not a model or dataset)
#   --space-sdk docker       → use Docker SDK (alternatives: gradio, streamlit, static)
#
# Space SDK options:
#   gradio     → HF manages the Gradio server, you just provide app.py
#   streamlit  → HF manages Streamlit, you provide app.py
#   docker     → YOU control everything via Dockerfile (our choice - more control)
#   static     → just serves static HTML files

# Option B: Via browser
# Go to huggingface.co → New Space → Choose Docker SDK
```

### Step 4: Initialize Git & Push Code
```bash
# Navigate to your project code
cd C:\Projects\Project-RagCore\ragcore

# Initialize a new git repository
git init
# What this does: creates a .git/ folder, making this directory a git repo

# Add the HF Space as a remote
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/RagCore
# What this does:
#   git remote add    → register a remote repository
#   origin            → conventional name for the primary remote
#   https://...       → the URL of your HF Space's git repo
#                       (every HF Space IS a git repo behind the scenes)

# Stage all files for commit
git add -A
# What this does:
#   git add    → stage files for the next commit
#   -A         → stage ALL changes (new files, modified files, deleted files)
#   NOTE: .gitignore controls what gets excluded (like .env, __pycache__)

# Check what's staged
git status
# Shows: which files are staged, modified, or untracked
# IMPORTANT: verify .env is NOT listed (should be in .gitignore)

# Create the commit
git commit -m "Initial deploy: RagCore RAG system"
# What this does:
#   git commit   → create a snapshot of all staged changes
#   -m "..."     → the commit message

# Push to HF Spaces
git push origin master:main
# What this does:
#   git push           → send commits to the remote repository
#   origin             → which remote (our HF Space)
#   master:main        → push local "master" branch to remote "main" branch
#                        (git init creates "master" by default,
#                         but HF Spaces expects "main")
#
# IMPORTANT: This push TRIGGERS the Docker build on HF Spaces!
# HF sees the Dockerfile and starts building your image in the cloud.
```

### Step 5: Configure Secrets on HF Spaces

**This is done in the browser, NOT via CLI:**

1. Go to `https://huggingface.co/spaces/YOUR_USERNAME/RagCore/settings`
2. Scroll to **"Repository secrets"**
3. Add each secret:

| Secret Name      | What It Is                                    |
|------------------|-----------------------------------------------|
| `GEMINI_API_KEY` | Your Google Gemini API key                     |
| `QDRANT_URL`     | Your Qdrant Cloud cluster URL                  |
| `QDRANT_API_KEY`  | Your Qdrant Cloud API key                      |

**How secrets work on HF Spaces:**
- Secrets are injected as environment variables into the Docker container at runtime
- They are NEVER visible in the code, logs, or git history
- Your app reads them via `pydantic-settings` (BaseSettings class in config.py)
- The `.env` file is NOT pushed (it's in .gitignore), secrets replace it

### Step 6: Monitor the Build
```bash
# Check Space status via API
curl -sL -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
  "https://huggingface.co/api/spaces/YOUR_USERNAME/RagCore"

# Or just watch the build logs in the browser:
# https://huggingface.co/spaces/YOUR_USERNAME/RagCore → "Logs" tab
```

### Step 7: Push Updates (After Making Changes)
```bash
# After editing code locally:
git add app/main.py app/ui/gradio_app.py    # stage specific changed files
git commit -m "Fix: description of what changed"
git push origin master:main

# This triggers a NEW build on HF Spaces automatically
# The old container stops, new image builds, new container starts
```

---

## 4. Every Command Explained

### Git Commands Used in This Project

```bash
# ─── Repository Setup ───

git init
# Creates a new git repository in the current directory
# Creates .git/ folder containing all version history

git remote add origin <URL>
# Adds a remote repository reference
# "origin" is just a name (convention for primary remote)
# URL points to where the code lives remotely (HF Spaces, GitHub, etc.)

git remote -v
# List all remotes and their URLs (verify your setup)


# ─── Staging & Committing ───

git status
# Shows the current state: what's modified, staged, untracked
# First command you run when unsure about repo state

git add <file>
# Stage a specific file for the next commit
# Example: git add app/main.py

git add -A
# Stage ALL changes (new, modified, deleted files)
# Respects .gitignore rules

git diff
# Show unstaged changes (what you modified but haven't staged)

git diff --staged
# Show staged changes (what will go into the next commit)

git commit -m "message"
# Create a commit (snapshot) with a message
# Only includes staged files

git log --oneline -5
# Show last 5 commits in compact format


# ─── Pushing ───

git push origin master:main
# Push local "master" branch to remote "main" branch
# Syntax: git push <remote> <local-branch>:<remote-branch>

git push origin master:main --force
# Force push - overwrites remote history with local
# DANGEROUS: use only when you know what you're doing
# We used this because HF Space had initial content we wanted to replace


# ─── Inspection ───

git remote -v
# Show all configured remotes

git branch
# List local branches (* marks current)

git log --oneline
# Compact commit history
```

### HF CLI Commands

```bash
# ─── Authentication ───

hf auth login --token <TOKEN> --add-to-git-credential
# Login with an access token
# --add-to-git-credential saves it for git push operations

hf auth whoami
# Check who you're logged in as

hf auth token
# Display your current token


# ─── Repository Management ───

hf repo create <namespace>/<name> --type space --space-sdk docker
# Create a new HF repository
# --type: model, dataset, or space
# --space-sdk: docker, gradio, streamlit, static (only for spaces)


# ─── Downloading (useful for other projects) ───

hf download <repo-id>
# Download a model/dataset from HuggingFace
# Example: hf download meta-llama/Llama-2-7b
```

### Docker Commands

```bash
# ─── Build ───

docker build -t ragcore .
# Build image from Dockerfile in current directory
# -t ragcore → name the image "ragcore"

docker build -t ragcore . --no-cache
# Build from scratch (ignore cached layers)


# ─── Run ───

docker run --env-file .env -p 8000:7860 ragcore
# Run a container from the "ragcore" image
# --env-file .env → inject environment variables
# -p 8000:7860    → map host:container ports

docker run -d --env-file .env -p 8000:7860 ragcore
# Same but detached (-d) → runs in background

docker compose up
# Read docker-compose.yml and start all defined services

docker compose up --build
# Rebuild images before starting


# ─── Inspection ───

docker ps
# List running containers

docker logs <container_id>
# View container logs

docker exec -it <container_id> bash
# Open a shell inside a running container


# ─── Cleanup ───

docker stop <container_id>
# Stop a running container

docker system prune
# Remove unused images, containers, networks
```

### Uvicorn Commands

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
# Start ASGI server
#   app.main:app  → import "app" from "app/main.py"
#   --host 0.0.0.0 → accept connections from any IP
#   --port 7860    → listen on port 7860

uvicorn app.main:app --reload
# Auto-restart on code changes (development only!)

uvicorn app.main:app --workers 4
# Run 4 worker processes (production scaling)
```

---

## 5. Docker Deep Dive

### The Dockerfile Explained Line by Line

```dockerfile
FROM python:3.12-slim
# Start from official Python 3.12 image (slim = smaller, no extras)
# This is the "base image" - a minimal Linux + Python environment
# "slim" variant is ~150MB vs full ~900MB

WORKDIR /app
# Set working directory inside the container
# All subsequent commands run from /app
# Like doing: mkdir /app && cd /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*
# Install system-level build tools needed by some Python packages
#   apt-get update              → refresh package index
#   apt-get install -y          → install without prompting
#   --no-install-recommends     → skip optional packages (smaller image)
#   build-essential             → gcc, make, etc. (needed by sentence-transformers)
#   rm -rf /var/lib/apt/lists/* → clean up package cache (smaller image)

COPY requirements.txt .
# Copy ONLY requirements.txt first (not all code)
# WHY? Docker caches each layer. If requirements.txt hasn't changed,
# Docker skips the pip install step on rebuild → MUCH faster builds

RUN pip install --no-cache-dir -r requirements.txt
# Install Python dependencies
#   --no-cache-dir → don't cache pip downloads (smaller image)
#   -r requirements.txt → install from requirements file

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# Pre-download the embedding model into the image
# Without this, the model downloads on EVERY container start (~80MB)
# By doing it at build time, it's cached in the image layer

RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2')"
# Pre-download the reranker model too
# Same reason: avoid runtime downloads

COPY . .
# Copy ALL remaining project files into the container
# .dockerignore controls what's excluded (.env, .git, tests, etc.)

EXPOSE 7860
# Document that this container listens on port 7860
# This is informational only - doesn't actually open the port
# The actual port mapping happens at runtime with -p flag

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
# Default command when the container starts
# Starts the FastAPI app via uvicorn on port 7860
```

### The .dockerignore Explained

```
.git            # Git history (not needed in container, saves ~100MB+)
__pycache__     # Python bytecode cache (regenerated automatically)
*.pyc           # Compiled Python files
.env            # SECRETS! Never include in images
.venv           # Local virtual environment (container has its own)
venv            # Alternative venv name
tests/          # Tests aren't needed in production
.github/        # CI/CD workflows
.pytest_cache   # Test cache
.coverage       # Coverage reports
htmlcov         # Coverage HTML reports
*.egg-info      # Package metadata
flashrank_cache/  # Local model cache (re-downloaded in Dockerfile)
.cache/         # General cache
```

### Docker Build Layer Caching

```
Layer 1: FROM python:3.12-slim          ← Cached (rarely changes)
Layer 2: RUN apt-get install...         ← Cached (rarely changes)
Layer 3: COPY requirements.txt          ← Cached if file unchanged
Layer 4: RUN pip install                ← Cached if requirements unchanged
Layer 5: RUN pre-download models        ← Cached if pip install unchanged
Layer 6: COPY . .                       ← ALWAYS rebuilds (code changes)
Layer 7: CMD                            ← ALWAYS rebuilds

WHY THIS ORDER MATTERS:
- Most changes are code changes (Layer 6)
- Docker rebuilds from the first changed layer onward
- By putting rarely-changing steps first, most builds only rebuild Layer 6+7
- This saves 5-10 minutes per build (pip install + model downloads are slow)
```

---

## 6. Git & Version Control

### .gitignore Explained

```gitignore
# Python bytecode - generated automatically, never commit
__pycache__/
*.py[cod]        # .pyc, .pyo, .pyd files
*$py.class
*.so             # Compiled C extensions
*.egg-info/
dist/
build/

# Virtual environments - each developer creates their own
.venv/
venv/
env/

# SECRETS - NEVER COMMIT API KEYS
.env

# IDE settings - personal preference, not project code
.vscode/
.idea/

# OS files - system-generated junk
.DS_Store        # macOS folder metadata
Thumbs.db        # Windows thumbnail cache

# Model caches - large binary files, re-downloaded as needed
flashrank_cache/
.cache/

# Upload folder - runtime data, not code
uploads/

# Test artifacts - regenerated on each test run
.pytest_cache/
htmlcov/
.coverage
```

### Why master:main?

```bash
git push origin master:main
```

- `git init` creates a branch called `master` (Git's default)
- HuggingFace Spaces expects the branch to be called `main`
- `master:main` syntax means: "push my local master TO the remote main"
- Alternative: rename your local branch first with `git branch -M main`

---

## 7. Environment Variables & Secrets

### How Environment Variables Flow

```
LOCAL DEVELOPMENT:
  .env file → pydantic BaseSettings → Python code
  (file on disk)

HF SPACES (PRODUCTION):
  HF Secrets UI → Docker container env vars → pydantic BaseSettings → Python code
  (encrypted storage)

DOCKER LOCAL:
  docker run --env-file .env → container env vars → pydantic BaseSettings → Python code
```

### How pydantic-settings Works (config.py)

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",          # Read from .env file if it exists
        env_file_encoding="utf-8",
        extra="ignore",           # Ignore unknown env vars
    )

    gemini_api_key: str = ""      # Maps to GEMINI_API_KEY env var
    qdrant_url: str = ""          # Maps to QDRANT_URL env var
    gemini_model: str = "gemini-2.5-flash"  # Default if env var not set

# Priority order (highest to lowest):
# 1. Actual environment variables (from HF Secrets or docker --env-file)
# 2. .env file values
# 3. Default values in the class definition
```

---

## 8. Troubleshooting Log

### Issue 1: Gemini Model Deprecated (404 Error)
```
ERROR: 404 This model models/gemini-2.0-flash is no longer available
```
**Cause:** Google deprecated `gemini-2.0-flash`
**Fix:** Changed to `gemini-2.5-flash` in both `.env` and `config.py`
**Lesson:** Always check model availability; cloud APIs deprecate models regularly

### Issue 2: Blank Screen on HF Spaces
```
GET /ui HTTP/1.1" 307 Temporary Redirect
```
**Cause:** Gradio was mounted at `/ui`, root `/` redirected to `/ui`.
HF Spaces serves apps through an iframe/proxy, and the redirect chain broke.
**Fix:** Mount Gradio at `/` instead of `/ui`
**Lesson:** When deploying behind a reverse proxy, minimize redirects.
The hosting platform expects content at the root path.

### Issue 3: 409 Conflict on /api/ingest
```
POST /api/ingest HTTP/1.1" 409 Conflict
```
**Cause:** Tried to upload a document that was already ingested (duplicate detection)
**Fix:** Not a bug — working as intended. Delete the document first or upload a different one.

---

## 9. Interview Questions & Answers

### A. RAG (Retrieval-Augmented Generation)

**Q1: What is RAG and why do we need it?**
> RAG combines a retrieval system with a language model. Instead of relying solely on the LLM's training data (which can be outdated or hallucinate), we first RETRIEVE relevant documents from our own database, then pass them as CONTEXT to the LLM to GENERATE an answer. This grounds the response in actual data.

**Q2: Explain the difference between dense and sparse retrieval.**
> **Dense retrieval** uses vector embeddings (like all-MiniLM-L6-v2) to find semantically similar text. "automobile" matches "car" because they have similar vector representations.
> **Sparse retrieval** (BM25) uses keyword matching with TF-IDF scoring. It's better for exact terms, names, and acronyms that embedding models might miss.
> RagCore uses BOTH (hybrid search) and combines results via Reciprocal Rank Fusion.

**Q3: What is Reciprocal Rank Fusion (RRF)?**
> RRF combines ranked lists from multiple retrieval methods. For each document, it computes: `score = Σ 1/(k + rank_i)` where k is a constant (usually 60) and rank_i is the document's position in each list. Documents ranked highly by BOTH methods get the best combined scores.

**Q4: Why use a reranker after retrieval?**
> Initial retrieval (dense + sparse) is fast but approximate. A reranker (FlashRank cross-encoder) takes each query-document PAIR and scores them together, capturing fine-grained relevance that bi-encoders miss. It's more accurate but slower, so we only rerank the top-K results (e.g., 10 → 5).

**Q5: What is chunking and why does overlap matter?**
> Chunking splits documents into smaller pieces for embedding and retrieval. We use 512-token chunks with 50-token overlap. Overlap ensures that important context at chunk boundaries isn't lost — a sentence split across two chunks appears fully in at least one of them.

**Q6: How does streaming work in your /api/ask endpoint?**
> We use Server-Sent Events (SSE). The server sends `data: {"text": "chunk"}` lines as the LLM generates tokens. The Gradio frontend reads these incrementally and updates the UI in real-time. The final event includes `{"done": true, "sources": [...]}` with metadata.

---

### B. FastAPI & Backend

**Q7: Why FastAPI instead of Flask?**
> FastAPI is async-native (supports `async/await`), has automatic OpenAPI docs, uses Pydantic for request/response validation, and is significantly faster than Flask for I/O-bound tasks (like calling external APIs). Our app makes many async calls to Qdrant and Gemini.

**Q8: Explain the lifespan context manager in your app.**
> FastAPI's `lifespan` replaces the old `@app.on_event("startup")` pattern. It's an async context manager — code before `yield` runs on startup (load models, connect to DB), code after `yield` runs on shutdown (cleanup). It's better because resources are properly scoped.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: load models, connect to Qdrant, build BM25 index
    yield
    # SHUTDOWN: cleanup
```

**Q9: How does Gradio mount inside FastAPI?**
> `gr.mount_gradio_app(app, gradio_app, path="/")` adds the Gradio ASGI app as a sub-application of FastAPI. FastAPI handles `/api/*` routes, and anything else falls through to Gradio. They share the same port and process.

**Q10: What is CORS and why allow all origins?**
> CORS (Cross-Origin Resource Sharing) controls which domains can call your API from a browser. `allow_origins=["*"]` means any website can call our API. For a demo/portfolio project this is fine. In production, you'd restrict to specific domains.

---

### C. Docker & Deployment

**Q11: What's the difference between CMD and RUN in a Dockerfile?**
> `RUN` executes during IMAGE BUILD (install packages, download models). Each RUN creates a new image layer. `CMD` specifies the DEFAULT COMMAND when a CONTAINER STARTS from the image. There's only one CMD per Dockerfile.

**Q12: Why do you COPY requirements.txt before COPY . .?**
> Docker layer caching. Each instruction creates a cached layer. If `requirements.txt` hasn't changed, Docker reuses the cached `pip install` layer (which takes 2-5 minutes). Only the final `COPY . .` layer rebuilds when code changes. This makes rebuilds fast.

**Q13: What does `EXPOSE 7860` actually do?**
> Almost nothing! It's documentation — tells humans and tools that the container listens on 7860. The actual port mapping happens at runtime with `docker run -p 8000:7860` or in HF Spaces configuration. Without `-p`, the port isn't accessible from outside the container.

**Q14: How does HF Spaces know how to build your app?**
> HF Spaces reads the `README.md` YAML frontmatter:
```yaml
---
sdk: docker        # Use Dockerfile to build
app_port: 7860     # Where the app listens
---
```
> When you push code, HF Spaces finds the Dockerfile, builds the image, and runs it. Secrets are injected as environment variables.

**Q15: What's the difference between `sdk: docker` and `sdk: gradio` on HF Spaces?**
> With `sdk: gradio`, HF manages everything — you just provide a `app.py` with a Gradio interface. With `sdk: docker`, YOU control the entire environment via Dockerfile. We chose Docker because we need FastAPI + Gradio + custom dependencies (sentence-transformers, flashrank, etc.) that require system-level packages.

---

### D. Vector Databases & Embeddings

**Q16: What is a vector embedding?**
> A vector embedding is a fixed-size array of numbers (e.g., 384 dimensions for MiniLM) that represents the MEANING of text. Similar texts have similar vectors (measured by cosine similarity). The embedding model (neural network) learns to map text → vectors during training on large text corpora.

**Q17: Why use Qdrant Cloud instead of a local vector DB?**
> Qdrant Cloud provides persistence, scalability, and availability without managing infrastructure. On HF Spaces (or any serverless platform), the filesystem is ephemeral — local data is lost on restart. Qdrant Cloud preserves our indexed documents across deployments.

**Q18: What is cosine similarity vs dot product?**
> Both measure vector similarity. **Cosine similarity** measures the angle between vectors (range -1 to 1), ignoring magnitude. **Dot product** considers both angle AND magnitude. For normalized embeddings (unit length), they give identical rankings. MiniLM produces normalized embeddings.

---

### E. Security & Production

**Q19: How do you handle secrets in your project?**
> - `.env` file for local development (in `.gitignore`, never committed)
> - HF Spaces Repository Secrets for production (encrypted, injected as env vars)
> - `.dockerignore` excludes `.env` from Docker images
> - `pydantic-settings` reads env vars with a priority chain: env vars > .env file > defaults

**Q20: What would you change for a production deployment?**
> 1. Restrict CORS to specific domains (not `*`)
> 2. Add authentication (API keys or OAuth)
> 3. Rate limiting (beyond just Gemini RPM)
> 4. Move from free HF Spaces CPU to GPU or dedicated server
> 5. Add monitoring/alerting (Prometheus, Grafana)
> 6. Use a proper CI/CD pipeline (not manual git push)
> 7. Add request logging and audit trail
> 8. Error reporting service (Sentry)

---

### F. General Software Engineering

**Q21: Explain the request flow when a user asks a question.**
> 1. User types question in Gradio UI (browser)
> 2. Gradio frontend sends JS event to Gradio backend (same server)
> 3. Gradio backend calls `ask_question()` Python function
> 4. Function makes HTTP POST to `http://127.0.0.1:7860/api/ask` (self-call)
> 5. FastAPI routes to query handler
> 6. Query analyzer extracts intent and filters
> 7. Hybrid retriever runs dense search (Qdrant) + sparse search (BM25)
> 8. Results are merged via Reciprocal Rank Fusion
> 9. FlashRank reranker scores and filters top results
> 10. Context + query sent to Gemini LLM
> 11. Gemini streams response tokens back
> 12. SSE events stream through FastAPI → Gradio → browser

**Q22: Why does the Gradio app call localhost instead of using Python imports directly?**
> Separation of concerns. The Gradio UI communicates with the backend via HTTP (same as any external client would). This means: (a) the API is testable independently, (b) the UI could be replaced without changing the backend, (c) the same API serves both the Gradio UI and any other client (curl, Postman, mobile app).

**Q23: What is `pydantic-settings` and how does it differ from regular Pydantic?**
> Regular Pydantic validates data structures (request/response models). `pydantic-settings` extends this to read configuration from environment variables, .env files, and other sources with automatic type conversion. It provides a typed, validated configuration layer — if `CHUNK_SIZE=512` is set as a string env var, it auto-converts to `int`.

---

### G. Scenario-Based Questions

**Q24: Your HF Space shows a blank screen. How do you debug?**
> 1. Check the Logs tab on HF Spaces for build/runtime errors
> 2. Verify the Docker build completed successfully
> 3. Check if the app started (`Uvicorn running on...` in logs)
> 4. Look for redirect issues (307/302 responses in logs)
> 5. Test if API endpoints work: hit `/health` endpoint
> 6. Check if secrets are configured (missing API keys cause silent failures)
> 7. Try accessing the Space URL directly (not in iframe)

**Q25: A user reports the app is slow. Where do you look?**
> 1. **Embedding model loading** — cold start takes ~2-5s (pre-loaded in lifespan)
> 2. **Qdrant latency** — network round-trip to cloud DB (~100-500ms)
> 3. **Reranker inference** — FlashRank runs on CPU (~200-500ms for 10 docs)
> 4. **Gemini API** — LLM generation time (~2-5s depending on response length)
> 5. **BM25 index size** — linear scan, slow with 100K+ documents
> 6. Check: is it the free CPU tier on HF Spaces? Upgrade to GPU.

**Q26: How would you add a new document type (e.g., DOCX)?**
> 1. Add `python-docx` to `requirements.txt`
> 2. Add a `parse_docx()` function in `app/utils/parsers.py`
> 3. Update the file type routing in the ingest endpoint
> 4. Add `.docx` to the Gradio file upload `file_types` list
> 5. Update Dockerfile if new system dependencies are needed
> 6. Test locally, commit, push to HF Spaces

**Q27: The Gemini API starts returning 429 (rate limited). How do you handle it?**
> RagCore already has `gemini_rpm_limit=15` in config. To improve:
> 1. Implement exponential backoff with retry (wait 1s, 2s, 4s...)
> 2. Add a request queue with rate limiting
> 3. Cache frequent queries (same question → same answer)
> 4. Consider a fallback LLM (e.g., switch to a different model if quota exhausted)
> 5. Upgrade API tier for higher rate limits

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│                 DEPLOYMENT CHEAT SHEET                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  LOCAL DEV:                                              │
│    pip install -r requirements.txt                       │
│    uvicorn app.main:app --port 7860                      │
│                                                          │
│  DOCKER LOCAL:                                           │
│    docker build -t ragcore .                             │
│    docker run --env-file .env -p 8000:7860 ragcore       │
│                                                          │
│  HF SPACES:                                              │
│    hf auth login --token TOKEN --add-to-git-credential   │
│    git init && git remote add origin HF_SPACE_URL        │
│    git add -A && git commit -m "msg"                     │
│    git push origin master:main                           │
│    → Add secrets in HF Space Settings                    │
│                                                          │
│  UPDATE DEPLOYED APP:                                    │
│    git add <files> && git commit -m "msg"                │
│    git push origin master:main                           │
│                                                          │
│  CHECK STATUS:                                           │
│    Browser: https://huggingface.co/spaces/USER/REPO      │
│    Logs: same URL → "Logs" tab                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```
