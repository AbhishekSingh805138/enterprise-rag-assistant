"""Central configuration. Loads from .env once and exposes typed settings."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root

PROJECT_ROOT = Path(__file__).parent.resolve()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chroma_dir: str = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "enterprise_docs")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Retrieval defaults
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4

    def validate(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")


settings = Settings()


def setup_logging() -> None:
    """Configure logging for the entire application. Call once at entry point."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid adding duplicate handlers on repeated calls.
    if not root.handlers:
        root.addHandler(handler)
    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "chromadb", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
