from app.core.retriever import HybridRetriever


def test_rrf_fusion_basic():
    dense = [
        {"chunk_id": "a", "text": "doc a", "score": 0.9, "metadata": {}},
        {"chunk_id": "b", "text": "doc b", "score": 0.8, "metadata": {}},
    ]
    sparse = [
        {"chunk_id": "b", "text": "doc b", "score": 5.0, "metadata": {}},
        {"chunk_id": "c", "text": "doc c", "score": 4.0, "metadata": {}},
    ]
    fused = HybridRetriever.rrf_fuse([dense, sparse])
    ids = [item["chunk_id"] for item in fused]
    # "b" appears in both lists so should rank highest
    assert ids[0] == "b"
    assert len(fused) == 3


def test_rrf_fusion_empty():
    fused = HybridRetriever.rrf_fuse([[], []])
    assert fused == []


def test_rrf_fusion_single_list():
    results = [
        {"chunk_id": "x", "text": "x", "score": 1.0, "metadata": {}},
    ]
    fused = HybridRetriever.rrf_fuse([results])
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "x"


def test_rrf_fusion_with_weights():
    dense = [
        {"chunk_id": "a", "text": "a", "score": 0.9, "metadata": {}},
    ]
    sparse = [
        {"chunk_id": "b", "text": "b", "score": 5.0, "metadata": {}},
    ]
    fused = HybridRetriever.rrf_fuse([dense, sparse], weights=[1.0, 0.0])
    # With weight 0 on sparse, only dense matters
    assert fused[0]["chunk_id"] == "a"


def test_apply_filters():
    results = [
        {"chunk_id": "1", "text": "t", "score": 1, "metadata": {"doc_type": "pdf", "source": "a.pdf", "tags": []}},
        {"chunk_id": "2", "text": "t", "score": 1, "metadata": {"doc_type": "html", "source": "b.html", "tags": []}},
    ]
    from app.models.schemas import SearchFilters

    filters = SearchFilters(doc_type="pdf")
    filtered = HybridRetriever._apply_filters(results, filters)
    assert len(filtered) == 1
    assert filtered[0]["chunk_id"] == "1"
