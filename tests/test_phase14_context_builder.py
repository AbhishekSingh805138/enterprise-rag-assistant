"""Phase 14: Context Builder tests.

Tests for:
- Content-hash deduplication
- Token budget enforcement
- Proportional source allocation
- Empty/edge case handling
- Integration with generate node
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from config import settings


def _set_setting(name: str, value):
    object.__setattr__(settings, name, value)


def _make_doc(content: str, filename: str = "test.md") -> Document:
    return Document(page_content=content, metadata={"filename": filename, "source": filename})


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Test content-hash deduplication."""

    def test_removes_exact_duplicates(self):
        from src.context.context_builder import build_context

        docs = [
            _make_doc("Same content here", "a.md"),
            _make_doc("Same content here", "b.md"),
            _make_doc("Different content", "c.md"),
        ]
        result = build_context(docs)
        assert result.docs_included == 2  # one duplicate removed

    def test_keeps_unique_docs(self):
        from src.context.context_builder import build_context

        docs = [
            _make_doc("Content one", "a.md"),
            _make_doc("Content two", "b.md"),
            _make_doc("Content three", "c.md"),
        ]
        result = build_context(docs)
        assert result.docs_included == 3

    def test_no_duplicates_in_output(self):
        from src.context.context_builder import build_context

        content = "This is duplicated text."
        docs = [_make_doc(content, "a.md")] * 5
        result = build_context(docs)
        assert result.docs_included == 1
        assert result.text.count(content) == 1


# ---------------------------------------------------------------------------
# Token budget tests
# ---------------------------------------------------------------------------


class TestTokenBudget:
    """Test token budget enforcement."""

    def test_respects_max_tokens(self):
        from src.context.context_builder import build_context

        # Create docs with ~500 chars each => ~125 tokens each
        docs = [_make_doc("x" * 500, f"doc{i}.md") for i in range(20)]
        result = build_context(docs, max_tokens=500)
        # Should be well under 500 tokens (~2000 chars)
        assert result.token_count <= 500

    def test_truncates_when_over_budget(self):
        from src.context.context_builder import build_context

        docs = [_make_doc("y" * 2000, f"doc{i}.md") for i in range(10)]
        result = build_context(docs, max_tokens=200)
        assert result.docs_truncated > 0

    def test_accounts_for_memory_context(self):
        from src.context.context_builder import build_context

        # Memory context takes ~250 tokens (1000 chars)
        memory = "Previous conversation context " * 33  # ~1000 chars
        docs = [_make_doc("x" * 2000, "a.md")]
        result_with_memory = build_context(docs, max_tokens=500, memory_context=memory)
        result_without = build_context(docs, max_tokens=500, memory_context="")
        # With memory, less room for docs
        assert result_with_memory.token_count <= result_without.token_count

    def test_uses_config_default(self):
        from src.context.context_builder import build_context

        orig = settings.context_max_tokens
        _set_setting("context_max_tokens", 100)
        try:
            docs = [_make_doc("z" * 2000, "a.md")]
            result = build_context(docs)
            assert result.token_count <= 100
        finally:
            _set_setting("context_max_tokens", orig)


# ---------------------------------------------------------------------------
# Source grouping tests
# ---------------------------------------------------------------------------


class TestSourceGrouping:
    """Test source grouping and proportional allocation."""

    def test_groups_chunks_by_source(self):
        from src.context.context_builder import build_context

        docs = [
            _make_doc("Part 1 of handbook", "handbook.md"),
            _make_doc("Part 2 of handbook", "handbook.md"),
            _make_doc("Vendor terms info", "vendor.md"),
        ]
        result = build_context(docs)
        assert "handbook.md" in result.sources_used
        assert "vendor.md" in result.sources_used

    def test_sources_used_tracks_included_sources(self):
        from src.context.context_builder import build_context

        docs = [
            _make_doc("Content A", "a.md"),
            _make_doc("Content B", "b.md"),
        ]
        result = build_context(docs)
        assert set(result.sources_used) == {"a.md", "b.md"}

    def test_proportional_allocation(self):
        from src.context.context_builder import build_context

        # One source has much more content than the other
        big_docs = [_make_doc("x" * 800, "big.md") for _ in range(3)]
        small_docs = [_make_doc("y" * 100, "small.md")]
        docs = big_docs + small_docs
        result = build_context(docs, max_tokens=1000)
        # Both sources should be represented
        assert "big.md" in result.sources_used
        assert "small.md" in result.sources_used


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_docs_list(self):
        from src.context.context_builder import build_context

        result = build_context([])
        assert result.text == ""
        assert result.token_count == 0
        assert result.sources_used == []
        assert result.docs_included == 0
        assert result.docs_truncated == 0

    def test_single_document(self):
        from src.context.context_builder import build_context

        docs = [_make_doc("Simple content.", "test.md")]
        result = build_context(docs)
        assert result.docs_included == 1
        assert "Simple content." in result.text
        assert result.sources_used == ["test.md"]

    def test_empty_content_document(self):
        from src.context.context_builder import build_context

        docs = [_make_doc("", "empty.md")]
        result = build_context(docs)
        # Empty content has 0 chars contribution, no error
        assert isinstance(result.text, str)

    def test_no_filename_metadata(self):
        from src.context.context_builder import build_context

        doc = Document(page_content="Some text", metadata={})
        result = build_context([doc])
        assert result.docs_included == 1
        assert "unknown" in result.sources_used

    def test_very_small_budget(self):
        from src.context.context_builder import build_context

        docs = [_make_doc("x" * 1000, "big.md")]
        result = build_context(docs, max_tokens=10)
        # Should handle gracefully — might truncate or skip
        assert isinstance(result.text, str)


# ---------------------------------------------------------------------------
# BuiltContext model tests
# ---------------------------------------------------------------------------


class TestBuiltContext:
    """Test BuiltContext dataclass."""

    def test_fields_present(self):
        from src.context.context_builder import BuiltContext

        ctx = BuiltContext(
            text="hello",
            token_count=1,
            sources_used=["a.md"],
            docs_included=1,
            docs_truncated=0,
        )
        assert ctx.text == "hello"
        assert ctx.token_count == 1
        assert ctx.sources_used == ["a.md"]
        assert ctx.docs_included == 1
        assert ctx.docs_truncated == 0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_config_field_exists(self):
        assert hasattr(settings, "context_max_tokens")

    def test_default_value(self):
        assert settings.context_max_tokens == 4000

    def test_positive_value(self):
        assert settings.context_max_tokens > 0
