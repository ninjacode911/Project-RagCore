from app.core.query_analyzer import QueryAnalyzer


def test_intent_factual():
    qa = QueryAnalyzer()
    result = qa.analyze("what is RAG?")
    assert result.intent == "factual"


def test_intent_comparative():
    qa = QueryAnalyzer()
    result = qa.analyze("compare BM25 and dense search")
    assert result.intent == "comparative"


def test_intent_summarize():
    qa = QueryAnalyzer()
    result = qa.analyze("summarize the report")
    assert result.intent == "summarize"


def test_intent_explanatory():
    qa = QueryAnalyzer()
    result = qa.analyze("why is RAG useful?")
    assert result.intent == "explanatory"


def test_doctype_extraction():
    qa = QueryAnalyzer()
    result = qa.analyze("search PDFs about machine learning")
    assert result.extracted_filters.doc_type == "pdf"


def test_no_filters():
    qa = QueryAnalyzer()
    result = qa.analyze("what is machine learning?")
    assert result.extracted_filters.doc_type is None
    assert result.extracted_filters.source is None
    assert result.clean_query == result.original_query


def test_date_extraction_last_month():
    qa = QueryAnalyzer()
    result = qa.analyze("documents from last month")
    assert result.extracted_filters.date_from is not None
    assert result.extracted_filters.date_to is not None


def test_clean_query_preserves_meaning():
    qa = QueryAnalyzer()
    result = qa.analyze("what is machine learning?")
    assert "machine learning" in result.clean_query
