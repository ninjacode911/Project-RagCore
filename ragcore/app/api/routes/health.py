from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    status = {"status": "ok", "components": {}}

    try:
        from app.core.embedder import _embedder
        status["components"]["embedder"] = "loaded" if _embedder else "not loaded"
    except Exception:
        status["components"]["embedder"] = "error"

    try:
        from app.core.bm25 import _bm25
        if _bm25:
            status["components"]["bm25"] = f"{_bm25.doc_count} documents"
        else:
            status["components"]["bm25"] = "not initialized"
    except Exception:
        status["components"]["bm25"] = "error"

    try:
        from app.core.vectorstore import _vectorstore
        if _vectorstore:
            count = _vectorstore.count()
            status["components"]["vectorstore"] = f"connected ({count} points)"
        else:
            status["components"]["vectorstore"] = "not connected"
    except Exception as e:
        status["components"]["vectorstore"] = f"error: {e}"
        status["status"] = "degraded"

    try:
        from app.core.llm import _llm
        status["components"]["llm"] = f"ready ({_llm.model_name})" if _llm else "not initialized"
    except Exception:
        status["components"]["llm"] = "error"

    return status
