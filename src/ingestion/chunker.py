"""Split documents into chunks for embedding.

Phase 1 uses RecursiveCharacterTextSplitter -- a solid, boring baseline.
In Phase 3 you'll swap/compare strategies (semantic chunking, parent-document,
markdown-header-aware) and measure the impact on context precision.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def chunk_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        # Split on natural boundaries first, fall back to chars.
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,  # records char offset in metadata -- useful for citations
    )
    return splitter.split_documents(docs)
