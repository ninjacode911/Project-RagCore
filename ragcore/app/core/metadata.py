import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.models.document import DocumentMetadata

logger = logging.getLogger(__name__)

DATE_PATTERNS = [
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{2}/\d{2}/\d{4})\b"),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4})\b"
    ),
]

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%B %d %Y"]


def extract_title(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) > 3:
            return line[:200]
    return None


def extract_dates(text: str) -> datetime | None:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text[:2000])  # Only scan beginning
        if match:
            date_str = match.group(1)
            for fmt in DATE_FORMATS:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
    return None


def extract_tags(text: str, max_tags: int = 10) -> list[str]:
    words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    counts = Counter(words)
    tags = [word.lower() for word, count in counts.most_common(max_tags * 2) if count >= 2]
    return tags[:max_tags]


def extract_metadata(raw_text: str, filename: str, page_count: int | None = None) -> DocumentMetadata:
    ext = Path(filename).suffix.lower().lstrip(".")
    doc_type = ext if ext else "unknown"

    return DocumentMetadata(
        source=filename,
        doc_type=doc_type,
        title=extract_title(raw_text),
        created_date=extract_dates(raw_text),
        tags=extract_tags(raw_text),
        page_count=page_count,
    )
