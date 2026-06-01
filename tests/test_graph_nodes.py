"""Tests for LangGraph CRAG node functions.

These tests verify routing logic and edge cases without making LLM calls.
LLM-dependent nodes (grade_documents, transform_query, generate) are tested
with mocked LLM responses.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.graph.nodes import (
    MAX_RETRIES,
    decide_after_grade,
    generate,
    retrieve,
    web_search,
)


class TestDecideAfterGrade:
    def test_relevant_goes_to_generate(self):
        state = {"relevant": True, "retries": 0}
        assert decide_after_grade(state) == "generate"

    def test_not_relevant_with_retries_left(self):
        state = {"relevant": False, "retries": 0}
        assert decide_after_grade(state) == "transform_query"

    def test_not_relevant_retries_exhausted(self):
        state = {"relevant": False, "retries": MAX_RETRIES}
        assert decide_after_grade(state) == "web_search"

    def test_not_relevant_over_max_retries(self):
        state = {"relevant": False, "retries": MAX_RETRIES + 5}
        assert decide_after_grade(state) == "web_search"

    def test_missing_relevant_key_treated_as_false(self):
        state = {"retries": 0}
        assert decide_after_grade(state) == "transform_query"

    def test_missing_retries_key_defaults_to_zero(self):
        state = {"relevant": False}
        assert decide_after_grade(state) == "transform_query"


class TestWebSearch:
    def test_appends_fallback_document(self):
        existing = [Document(page_content="existing", metadata={"filename": "a.md"})]
        state = {"question": "test", "documents": existing}
        result = web_search(state)
        assert len(result["documents"]) == 2
        assert result["web_fallback_used"] is True
        assert result["documents"][0].page_content == "existing"

    def test_handles_empty_documents(self):
        state = {"question": "test", "documents": []}
        result = web_search(state)
        assert len(result["documents"]) == 1
        assert result["web_fallback_used"] is True

    def test_handles_missing_documents_key(self):
        state = {"question": "test"}
        result = web_search(state)
        assert len(result["documents"]) == 1


class TestGenerate:
    def test_no_documents_returns_idk(self):
        state = {"question": "What is X?", "documents": []}
        result = generate(state)
        assert "don't have enough information" in result["generation"].lower()

    def test_missing_documents_key_returns_idk(self):
        state = {"question": "What is X?"}
        result = generate(state)
        assert "don't have enough information" in result["generation"].lower()


class TestRetrieve:
    @patch("src.graph.nodes.get_retriever")
    def test_retrieve_returns_docs_and_retries(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="doc1", metadata={"filename": "a.md"})
        ]
        mock_get_retriever.return_value = mock_retriever

        state = {"question": "test question", "retries": 1}
        result = retrieve(state)
        assert len(result["documents"]) == 1
        assert result["retries"] == 1

    @patch("src.graph.nodes.get_retriever")
    def test_retrieve_handles_failure(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = RuntimeError("connection error")
        mock_get_retriever.return_value = mock_retriever

        state = {"question": "test", "retries": 0}
        result = retrieve(state)
        assert result["documents"] == []
