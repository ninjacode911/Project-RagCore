import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_text():
    return (
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "information retrieval with text generation. It was introduced by "
        "Facebook AI Research in 2020. RAG systems first retrieve relevant "
        "documents from a knowledge base, then use a language model to generate "
        "answers based on those documents. This approach reduces hallucinations "
        "and provides more factual responses compared to pure generation."
    )
