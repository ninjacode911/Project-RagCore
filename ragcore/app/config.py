import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    gemini_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Qdrant
    qdrant_collection: str = "ragcore_docs"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 10
    rerank_top_k: int = 5
    dense_weight: float = 0.6
    sparse_weight: float = 0.4

    # LLM
    gemini_model: str = "gemini-2.5-flash"
    gemini_rpm_limit: int = 15
    gemini_temperature: float = 0.3
    gemini_max_tokens: int = 2048

    # App
    log_level: str = "INFO"
    max_file_size_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()


def setup_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
