import re
import time
import uuid
import logging
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)


def generate_id() -> str:
    return str(uuid.uuid4())


def count_words(text: str) -> int:
    return len(text.split())


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


@contextmanager
def timer(label: str = "operation"):
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"{label} completed in {elapsed:.1f}ms")


def retry_with_backoff(retries: int = 3, base_delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
        return wrapper
    return decorator
