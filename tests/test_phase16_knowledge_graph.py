"""Phase 16: Knowledge Graph tests.

Tests for:
- Triple extraction (mock LLM)
- Knowledge graph store CRUD + persistence
- Graph traversal and neighbor queries
- Knowledge graph retriever
- Factory registration
- Config defaults
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import settings


def _set_setting(name: str, value):
    object.__setattr__(settings, name, value)


# ---------------------------------------------------------------------------
# Models tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_triple_fields(self):
        from src.knowledge_graph.models import Triple

        t = Triple(subject="PTO policy", predicate="allows", object="20 days")
        assert t.subject == "PTO policy"
        assert t.predicate == "allows"
        assert t.object == "20 days"

    def test_entity_fields(self):
        from src.knowledge_graph.models import Entity

        e = Entity(name="HR", type="department", source_doc="handbook.md")
        assert e.name == "HR"
        assert e.type == "department"

    def test_relationship_fields(self):
        from src.knowledge_graph.models import Relationship

        r = Relationship(
            subject="PTO policy", predicate="managed_by", object="HR",
            source_doc="handbook.md", confidence=0.9,
        )
        assert r.confidence == 0.9


# ---------------------------------------------------------------------------
# Extractor tests
# ---------------------------------------------------------------------------


class TestExtractor:
    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig = settings.knowledge_graph_enabled
        yield
        _set_setting("knowledge_graph_enabled", orig)

    def test_disabled_returns_empty(self):
        _set_setting("knowledge_graph_enabled", False)

        from src.knowledge_graph.extractor import extract_triples

        result = extract_triples("Some text about PTO policy")
        assert result == []

    def test_empty_text_returns_empty(self):
        _set_setting("knowledge_graph_enabled", True)

        from src.knowledge_graph.extractor import extract_triples

        result = extract_triples("")
        assert result == []

    @patch("src.knowledge_graph.extractor.get_breaker")
    @patch("src.knowledge_graph.extractor.get_llm")
    def test_extracts_triples_from_llm(self, mock_llm, mock_breaker):
        _set_setting("knowledge_graph_enabled", True)

        from src.knowledge_graph.extractor import ExtractionResult, extract_triples

        mock_cb = MagicMock()
        mock_breaker.return_value = mock_cb
        mock_cb.call.return_value = ExtractionResult(triples=[
            {"subject": "PTO policy", "predicate": "allows", "object": "20 days per year"},
            {"subject": "PTO policy", "predicate": "managed_by", "object": "HR department"},
        ])

        result = extract_triples("PTO policy allows 20 days per year, managed by HR.")
        assert len(result) == 2
        assert result[0].subject == "PTO policy"
        assert result[0].predicate == "allows"

    @patch("src.knowledge_graph.extractor.get_breaker")
    def test_handles_llm_failure(self, mock_breaker):
        _set_setting("knowledge_graph_enabled", True)

        mock_cb = MagicMock()
        mock_breaker.return_value = mock_cb
        mock_cb.call.side_effect = RuntimeError("LLM down")

        from src.knowledge_graph.extractor import extract_triples

        result = extract_triples("Some text")
        assert result == []


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------


class TestKnowledgeGraphStore:
    @pytest.fixture
    def store(self, tmp_path):
        from src.knowledge_graph.store import KnowledgeGraphStore

        path = str(tmp_path / "test_kg.json")
        return KnowledgeGraphStore(persist_path=path)

    def test_add_triples(self, store):
        from src.knowledge_graph.models import Triple

        triples = [
            Triple(subject="PTO policy", predicate="allows", object="20 days"),
            Triple(subject="PTO policy", predicate="managed_by", object="HR"),
        ]
        added = store.add_triples(triples, source_doc="handbook.md")
        assert added == 2

    def test_add_empty_triples(self, store):
        added = store.add_triples([])
        assert added == 0

    def test_add_triples_with_empty_subject(self, store):
        from src.knowledge_graph.models import Triple

        triples = [Triple(subject="", predicate="is", object="something")]
        added = store.add_triples(triples)
        assert added == 0

    def test_query_neighbors(self, store):
        from src.knowledge_graph.models import Triple

        triples = [
            Triple(subject="PTO policy", predicate="allows", object="20 days"),
            Triple(subject="PTO policy", predicate="managed_by", object="HR"),
            Triple(subject="HR", predicate="manages", object="onboarding"),
        ]
        store.add_triples(triples)

        # Depth 1: direct neighbors of "PTO policy"
        neighbors = store.query_neighbors("PTO policy", depth=1)
        assert len(neighbors) >= 2  # at least the two direct edges

    def test_query_neighbors_depth_2(self, store):
        from src.knowledge_graph.models import Triple

        triples = [
            Triple(subject="A", predicate="links", object="B"),
            Triple(subject="B", predicate="links", object="C"),
            Triple(subject="C", predicate="links", object="D"),
        ]
        store.add_triples(triples)

        # Depth 2: should reach C from A
        neighbors = store.query_neighbors("A", depth=2)
        objects = [r["object"] for r in neighbors]
        assert "B" in objects
        assert "C" in objects

    def test_query_unknown_entity(self, store):
        neighbors = store.query_neighbors("nonexistent")
        assert neighbors == []

    def test_case_insensitive_lookup(self, store):
        from src.knowledge_graph.models import Triple

        store.add_triples([Triple(subject="HR", predicate="manages", object="PTO")])
        neighbors = store.query_neighbors("hr")  # lowercase
        assert len(neighbors) > 0

    def test_search_entities(self, store):
        from src.knowledge_graph.models import Triple

        store.add_triples([
            Triple(subject="PTO policy", predicate="allows", object="20 days"),
            Triple(subject="NDA agreement", predicate="requires", object="signature"),
        ])
        results = store.search_entities("pto")
        assert "PTO policy" in results

    def test_stats(self, store):
        from src.knowledge_graph.models import Triple

        store.add_triples([
            Triple(subject="A", predicate="links", object="B"),
        ])
        stats = store.stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1

    def test_clear(self, store):
        from src.knowledge_graph.models import Triple

        store.add_triples([Triple(subject="A", predicate="links", object="B")])
        store.clear()
        assert store.stats() == {"nodes": 0, "edges": 0}

    def test_persistence(self, tmp_path):
        from src.knowledge_graph.models import Triple
        from src.knowledge_graph.store import KnowledgeGraphStore

        path = str(tmp_path / "persist_test.json")

        # Create and populate
        store1 = KnowledgeGraphStore(persist_path=path)
        store1.add_triples([Triple(subject="X", predicate="rel", object="Y")])

        # Reload from disk
        store2 = KnowledgeGraphStore(persist_path=path)
        stats = store2.stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------


class TestKnowledgeGraphRetriever:

    @pytest.fixture
    def populated_store(self, tmp_path):
        from src.knowledge_graph.models import Triple
        from src.knowledge_graph.store import KnowledgeGraphStore, reset_kg_store

        reset_kg_store()
        path = str(tmp_path / "retriever_test.json")

        # Patch the singleton to use our temp path
        store = KnowledgeGraphStore(persist_path=path)
        store.add_triples([
            Triple(subject="PTO policy", predicate="allows", object="20 days per year"),
            Triple(subject="PTO policy", predicate="managed_by", object="HR department"),
            Triple(subject="HR department", predicate="manages", object="onboarding process"),
            Triple(subject="engineering", predicate="follows", object="onboarding process"),
        ], source_doc="handbook.md")
        return store

    @patch("src.knowledge_graph.store.get_kg_store")
    def test_retriever_finds_docs(self, mock_get_store, populated_store):
        mock_get_store.return_value = populated_store

        from src.knowledge_graph.retriever import KnowledgeGraphRetriever

        retriever = KnowledgeGraphRetriever(k=4, depth=2)
        docs = retriever.invoke("What is the engineering onboarding process?")
        assert len(docs) > 0
        assert all(hasattr(d, "page_content") for d in docs)

    @patch("src.knowledge_graph.store.get_kg_store")
    def test_retriever_empty_graph(self, mock_get_store):
        from src.knowledge_graph.store import KnowledgeGraphStore

        empty_store = KnowledgeGraphStore(persist_path=":memory:")
        mock_get_store.return_value = empty_store

        from src.knowledge_graph.retriever import KnowledgeGraphRetriever

        retriever = KnowledgeGraphRetriever(k=4)
        docs = retriever.invoke("What is the PTO policy?")
        assert docs == []

    @patch("src.knowledge_graph.store.get_kg_store")
    def test_retriever_respects_k(self, mock_get_store, populated_store):
        mock_get_store.return_value = populated_store

        from src.knowledge_graph.retriever import KnowledgeGraphRetriever

        retriever = KnowledgeGraphRetriever(k=1, depth=2)
        docs = retriever.invoke("What does HR manage?")
        assert len(docs) <= 1


# ---------------------------------------------------------------------------
# Factory registration tests
# ---------------------------------------------------------------------------


class TestFactoryRegistration:
    def test_knowledge_graph_in_registry(self):
        from src.retrieval.factory import STRATEGIES

        assert "knowledge_graph" in STRATEGIES

    def test_knowledge_graph_builds(self):
        from src.retrieval.factory import get_retriever

        retriever = get_retriever("knowledge_graph", k=3)
        assert hasattr(retriever, "invoke")
        assert retriever.k == 3


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_config_fields_exist(self):
        assert hasattr(settings, "knowledge_graph_enabled")
        assert hasattr(settings, "kg_max_depth")
        assert hasattr(settings, "kg_persist_path")

    def test_defaults(self):
        assert settings.knowledge_graph_enabled is False
        assert settings.kg_max_depth == 2
        assert isinstance(settings.kg_persist_path, str)
