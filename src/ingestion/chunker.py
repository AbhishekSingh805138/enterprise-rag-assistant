"""Split documents into chunks for embedding.

Supports two strategies:
  - Default: RecursiveCharacterTextSplitter (works for all doc types)
  - Markdown-aware: MarkdownHeaderTextSplitter for .md files (Phase 8),
    which splits on headers first, then applies recursive splitting on
    oversized sections. Header hierarchy is preserved in metadata.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import settings

logger = logging.getLogger(__name__)

# Markdown headers to split on (H1 through H3)
_MD_HEADERS = [
    ("#", "header_h1"),
    ("##", "header_h2"),
    ("###", "header_h3"),
]


def _chunk_markdown(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split a Markdown document on headers, then recursively split oversized sections."""
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS,
        strip_headers=False,
    )
    sections = md_splitter.split_text(doc.page_content)

    # Merge parent metadata into each section
    for section in sections:
        for key, value in doc.metadata.items():
            section.metadata.setdefault(key, value)

    # Apply recursive splitting to any oversized sections
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = recursive.split_documents(sections)

    logger.debug(
        "Markdown chunking: %d sections → %d chunks for %s",
        len(sections), len(chunks), doc.metadata.get("filename", "?"),
    )
    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split *docs* into chunks, preserving and inheriting metadata.

    When MARKDOWN_AWARE_CHUNKING is enabled, .md files are split on headers
    first, preserving header hierarchy in metadata (header_h1, header_h2, header_h3).
    """
    if not docs:
        logger.warning("chunk_documents called with empty document list")
        return []

    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    use_md_chunking = settings.markdown_aware_chunking
    all_chunks: list[Document] = []

    # Separate markdown docs from others if markdown-aware chunking is enabled
    md_docs: list[Document] = []
    other_docs: list[Document] = []

    if use_md_chunking:
        for doc in docs:
            doc_type = doc.metadata.get("doc_type", "")
            filename = doc.metadata.get("filename", "")
            if doc_type == "md" or filename.endswith(".md"):
                md_docs.append(doc)
            else:
                other_docs.append(doc)
    else:
        other_docs = docs

    # Markdown-aware chunking for .md files
    for md_doc in md_docs:
        all_chunks.extend(_chunk_markdown(md_doc, size, overlap))

    # Standard recursive chunking for everything else
    if other_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )
        all_chunks.extend(splitter.split_documents(other_docs))

    logger.info(
        "Chunked %d document(s) into %d chunk(s) (size=%d, overlap=%d, md_aware=%s, md_docs=%d)",
        len(docs), len(all_chunks), size, overlap, use_md_chunking, len(md_docs),
    )
    return all_chunks
