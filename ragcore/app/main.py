import logging
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.routes import health, ingest, query
from app.config import get_settings, setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("RagCore starting up...")

    settings = get_settings()

    # Initialize services that need warm-up
    try:
        from app.core.embedder import get_embedder

        get_embedder()
        logger.info("Embedder loaded")
    except Exception as e:
        logger.warning(f"Embedder initialization deferred: {e}")

    try:
        from app.core.vectorstore import get_vectorstore
        from app.core.bm25 import get_bm25

        vs = get_vectorstore()
        bm25 = get_bm25()
        bm25.rebuild_from_vectorstore(vs)
        logger.info(f"BM25 index ready: {bm25.doc_count} documents")
    except Exception as e:
        logger.warning(f"Vectorstore/BM25 initialization deferred: {e}")

    logger.info("RagCore ready!")
    yield
    logger.info("RagCore shutting down...")


app = FastAPI(
    title="RagCore",
    description="RAG system with hybrid search and metadata filtering",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)

# Mount Gradio UI at root
from app.ui.gradio_app import create_gradio_app

gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/")
