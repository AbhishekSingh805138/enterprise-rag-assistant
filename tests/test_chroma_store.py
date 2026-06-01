"""Tests for the ChromaDB vector store wrapper."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.vectorstore.chroma_store import (
    _content_hash,
    add_chunks,
    collection_stats,
    get_retriever,
    get_vectorstore,
    reset_store,
)


class TestContentHash:
    def test_deterministic(self):
        text = "Some content"
        meta = {"source": "test.md", "start_index": 0}
        h1 = _content_hash(text, meta)
        h2 = _content_hash(text, meta)
        assert h1 == h2

    def test_different_content_different_hash(self):
        meta = {"source": "test.md", "start_index": 0}
        h1 = _content_hash("Content A", meta)
        h2 = _content_hash("Content B", meta)
        assert h1 != h2

    def test_different_source_different_hash(self):
        text = "Same content"
        h1 = _content_hash(text, {"source": "a.md", "start_index": 0})
        h2 = _content_hash(text, {"source": "b.md", "start_index": 0})
        assert h1 != h2

    def test_different_offset_different_hash(self):
        text = "Same content"
        h1 = _content_hash(text, {"source": "a.md", "start_index": 0})
        h2 = _content_hash(text, {"source": "a.md", "start_index": 100})
        assert h1 != h2

    def test_returns_hex_string(self):
        h = _content_hash("text", {"source": "f.md"})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex


class TestResetStore:
    def test_reset_clears_singletons(self):
        """reset_store should clear module-level caches."""
        reset_store()
        # After reset, the next get_vectorstore call creates a fresh instance.
        import src.vectorstore.chroma_store as mod
        assert mod._embeddings is None
        assert mod._vectorstore is None
