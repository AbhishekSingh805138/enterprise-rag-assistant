"""Split documents into chunks for embedding.

Phase 1 uses RecursiveCharacterTextSplitter -- a solid, boring baseline.
In Phase 3 you'll swap/compare strategies (semantic chunking, parent-document,
markdown-header-aware) and measure the impact on context precision.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)


def chunk_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split *docs* into chunks, preserving and inheriting metadata."""
    if not docs:
        logger.warning("chunk_documents called with empty document list")
        return []

    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)
    logger.info(
        "Chunked %d document(s) into %d chunk(s) (size=%d, overlap=%d)",
        len(docs), len(chunks), size, overlap,
    )
    return chunks
