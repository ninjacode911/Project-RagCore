from app.core.chunker import chunk_text


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_single_sentence():
    chunks = chunk_text("This is a single sentence.", chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "This is a single sentence."
    assert chunks[0]["chunk_index"] == 0


def test_multiple_chunks():
    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."
    chunks = chunk_text(text, chunk_size=5, chunk_overlap=2)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i
        assert chunk["text"]
        assert chunk["start_char"] >= 0
        assert chunk["end_char"] > chunk["start_char"]


def test_overlap_present():
    text = "Alpha bravo charlie delta. Echo foxtrot golf hotel. India juliet kilo lima."
    chunks = chunk_text(text, chunk_size=4, chunk_overlap=2)
    if len(chunks) > 1:
        first_words = chunks[0]["text"].split()
        second_words = chunks[1]["text"].split()
        overlap = set(first_words[-2:]) & set(second_words[:2])
        assert len(overlap) > 0


def test_chunk_size_respected():
    text = " ".join(["word"] * 100) + "."
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    for chunk in chunks[:-1]:  # Last chunk can be smaller
        word_count = len(chunk["text"].split())
        assert word_count <= 25  # Allow some slack for sentence boundaries
