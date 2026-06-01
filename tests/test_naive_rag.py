"""Tests for the naive RAG chain."""
from __future__ import annotations

from langchain_core.documents import Document

from src.rag.naive_rag import _format_docs, answer


class TestFormatDocs:
    def test_empty_list(self):
        result = _format_docs([])
        assert "no documents" in result.lower()

    def test_single_doc(self):
        docs = [
            Document(
                page_content="Hello world",
                metadata={"filename": "test.md"},
            )
        ]
        result = _format_docs(docs)
        assert "[source: test.md]" in result
        assert "Hello world" in result

    def test_multiple_docs_separated(self):
        docs = [
            Document(page_content="Doc A", metadata={"filename": "a.md"}),
            Document(page_content="Doc B", metadata={"filename": "b.md"}),
        ]
        result = _format_docs(docs)
        assert "Doc A" in result
        assert "Doc B" in result
        assert "---" in result

    def test_missing_filename_falls_back_to_source(self):
        docs = [
            Document(page_content="text", metadata={"source": "/path/to/file.md"}),
        ]
        result = _format_docs(docs)
        assert "/path/to/file.md" in result


class TestAnswer:
    def test_empty_question_returns_prompt(self):
        assert answer("") == "Please provide a question."
        assert answer("   ") == "Please provide a question."
