"""Central configuration. Loads from .env once and exposes typed settings."""
from __future__ import annotations

import os
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

    # Retrieval defaults
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4

    def validate(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )


settings = Settings()
