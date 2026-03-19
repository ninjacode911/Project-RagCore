import logging
import re
from datetime import datetime, timedelta

from dateutil import parser as date_parser

from app.models.schemas import AnalyzedQuery, SearchFilters

logger = logging.getLogger(__name__)

# Doc type patterns
DOCTYPE_PATTERNS = {
    "pdf": re.compile(r"\bpdfs?\b", re.IGNORECASE),
    "html": re.compile(r"\bhtml\b", re.IGNORECASE),
    "txt": re.compile(r"\btext\s+files?\b|\btxt\b", re.IGNORECASE),
}

# Relative date patterns
RELATIVE_DATE_PATTERNS = [
    (re.compile(r"\blast\s+week\b", re.IGNORECASE), lambda: (datetime.now() - timedelta(weeks=1), datetime.now())),
    (re.compile(r"\blast\s+month\b", re.IGNORECASE), lambda: (datetime.now() - timedelta(days=30), datetime.now())),
    (re.compile(r"\blast\s+year\b", re.IGNORECASE), lambda: (datetime.now() - timedelta(days=365), datetime.now())),
    (re.compile(r"\bthis\s+week\b", re.IGNORECASE), lambda: (datetime.now() - timedelta(days=datetime.now().weekday()), datetime.now())),
    (re.compile(r"\bthis\s+month\b", re.IGNORECASE), lambda: (datetime.now().replace(day=1), datetime.now())),
    (re.compile(r"\bthis\s+year\b", re.IGNORECASE), lambda: (datetime.now().replace(month=1, day=1), datetime.now())),
    (re.compile(r"\btoday\b", re.IGNORECASE), lambda: (datetime.now().replace(hour=0, minute=0, second=0), datetime.now())),
    (re.compile(r"\byesterday\b", re.IGNORECASE), lambda: (datetime.now() - timedelta(days=1), datetime.now())),
]

# Absolute date patterns
AFTER_DATE = re.compile(r"\bafter\s+(\S+)\b", re.IGNORECASE)
BEFORE_DATE = re.compile(r"\bbefore\s+(\S+)\b", re.IGNORECASE)
FROM_SOURCE = re.compile(r"\bfrom\s+(\S+\.\w{2,4})\b", re.IGNORECASE)

# Intent patterns
INTENT_PATTERNS = [
    ("summarize", re.compile(r"\bsummar(?:ize|y)\b|\boverview\b", re.IGNORECASE)),
    ("comparative", re.compile(r"\bcompar[ei]\b|\bdifference\b|\bvs\.?\b|\bversus\b", re.IGNORECASE)),
    ("list", re.compile(r"\blist\b|\benumerate\b|\bwhat are all\b", re.IGNORECASE)),
    ("explanatory", re.compile(r"^(?:why|how|explain)\b", re.IGNORECASE)),
    ("factual", re.compile(r"^(?:what|who|when|where|how many|how much)\b", re.IGNORECASE)),
]


class QueryAnalyzer:
    def analyze(self, query: str) -> AnalyzedQuery:
        filters = SearchFilters()
        clean = query
        confidence = 0.5
        phrases_to_remove = []

        # Extract doc type
        for doc_type, pattern in DOCTYPE_PATTERNS.items():
            match = pattern.search(clean)
            if match:
                filters.doc_type = doc_type
                phrases_to_remove.append(match.group())
                confidence += 0.1

        # Extract relative dates
        for pattern, date_fn in RELATIVE_DATE_PATTERNS:
            match = pattern.search(clean)
            if match:
                date_from, date_to = date_fn()
                filters.date_from = date_from
                filters.date_to = date_to
                phrases_to_remove.append(match.group())
                confidence += 0.1
                break

        # Extract absolute dates
        if not filters.date_from:
            match = AFTER_DATE.search(clean)
            if match:
                try:
                    filters.date_from = date_parser.parse(match.group(1))
                    phrases_to_remove.append(match.group())
                    confidence += 0.1
                except (ValueError, OverflowError):
                    pass

        if not filters.date_to:
            match = BEFORE_DATE.search(clean)
            if match:
                try:
                    filters.date_to = date_parser.parse(match.group(1))
                    phrases_to_remove.append(match.group())
                    confidence += 0.1
                except (ValueError, OverflowError):
                    pass

        # Extract source
        match = FROM_SOURCE.search(clean)
        if match:
            filters.source = match.group(1)
            phrases_to_remove.append(match.group())
            confidence += 0.1

        # Clean query by removing extracted filter phrases
        for phrase in phrases_to_remove:
            clean = clean.replace(phrase, "")
        clean = re.sub(r"\s+", " ", clean).strip()
        # Remove dangling prepositions and leading ones
        clean = re.sub(r"\b(?:about|from|in|on)\s*$", "", clean).strip()
        clean = re.sub(r"^\b(?:about|from|in|on)\s+", "", clean).strip()

        if not clean:
            clean = query

        # Classify intent
        intent = "factual"
        for intent_name, pattern in INTENT_PATTERNS:
            if pattern.search(query):
                intent = intent_name
                break

        confidence = min(confidence, 1.0)

        analyzed = AnalyzedQuery(
            original_query=query,
            clean_query=clean,
            intent=intent,
            extracted_filters=filters,
            confidence=confidence,
        )
        logger.info(f"Query analyzed: intent={intent}, filters={filters.model_dump(exclude_none=True)}")
        return analyzed
