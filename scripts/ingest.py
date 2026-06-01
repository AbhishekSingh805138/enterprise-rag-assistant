"""Ingest documents into ChromaDB.

Usage:
    python -m scripts.ingest ./data/sample_docs
    python -m scripts.ingest ./some_file.pdf
"""
from __future__ import annotations

import sys

from config import settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_path
from src.vectorstore.chroma_store import add_chunks


def main() -> None:
    settings.validate()
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/sample_docs"

    print(f"Loading from {target} ...")
    docs = load_path(target)
    print(f"  loaded {len(docs)} document(s)")

    chunks = chunk_documents(docs)
    print(f"  split into {len(chunks)} chunk(s)")

    n = add_chunks(chunks)
    print(f"Embedded and persisted {n} chunk(s) to {settings.chroma_dir}")


if __name__ == "__main__":
    main()
