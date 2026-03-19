import logging
import re

logger = logging.getLogger(__name__)

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    if not text or not text.strip():
        return []

    sentences = SENTENCE_PATTERN.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    current_words: list[str] = []
    current_start = 0
    char_pos = 0

    for sentence in sentences:
        words = sentence.split()

        if current_words and len(current_words) + len(words) > chunk_size:
            chunk_text_str = " ".join(current_words)
            chunk_end = current_start + len(chunk_text_str)
            chunks.append({
                "text": chunk_text_str,
                "start_char": current_start,
                "end_char": chunk_end,
                "chunk_index": len(chunks),
            })

            # Overlap: keep last chunk_overlap words
            overlap_words = current_words[-chunk_overlap:] if chunk_overlap > 0 else []
            overlap_text = " ".join(overlap_words)
            current_start = chunk_end - len(overlap_text)
            current_words = overlap_words

        current_words.extend(words)

    # Last chunk
    if current_words:
        chunk_text_str = " ".join(current_words)
        chunks.append({
            "text": chunk_text_str,
            "start_char": current_start,
            "end_char": current_start + len(chunk_text_str),
            "chunk_index": len(chunks),
        })

    logger.info(f"Chunked text into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
