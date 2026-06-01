"""Tests for the document chunker."""
from __future__ import annotations

from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents


class TestChunkDocuments:
    def test_empty_input_returns_empty(self):
        assert chunk_documents([]) == []

    def test_short_doc_stays_single_chunk(self):
        doc = Document(
            page_content="Short text.",
            metadata={"source": "test.md", "filename": "test.md"},
        )
        chunks = chunk_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."

    def test_long_doc_is_split(self):
        # 3000 chars should produce multiple chunks at default 1000/150
        doc = Document(
            page_content="word " * 600,
            metadata={"source": "long.md", "filename": "long.md"},
        )
        chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=150)
        assert len(chunks) > 1

    def test_metadata_preserved(self):
        doc = Document(
            page_content="Some content here. " * 100,
            metadata={
                "source": "test.md",
                "filename": "test.md",
                "department": "hr",
                "access_level": "internal",
            },
        )
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.metadata["department"] == "hr"
            assert chunk.metadata["access_level"] == "internal"
            assert chunk.metadata["filename"] == "test.md"

    def test_start_index_present(self):
        doc = Document(
            page_content="Line one.\n\nLine two.\n\nLine three.",
            metadata={"source": "test.md"},
        )
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert "start_index" in chunk.metadata

    def test_custom_chunk_size(self):
        doc = Document(
            page_content="word " * 200,
            metadata={"source": "test.md"},
        )
        small_chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
        large_chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
        assert len(small_chunks) > len(large_chunks)
