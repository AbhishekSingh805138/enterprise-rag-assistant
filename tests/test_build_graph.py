"""Tests for graph building and the ask() interface."""
from __future__ import annotations

from src.graph.build_graph import ask, build_graph


class TestBuildGraph:
    def test_graph_compiles(self):
        """The graph should compile without errors."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"retrieve", "grade_documents", "transform_query", "web_search", "generate"}
        # LangGraph adds __start__ and __end__ nodes
        assert expected.issubset(node_names)


class TestAsk:
    def test_empty_question(self):
        result = ask("")
        assert result == "Please provide a question."

    def test_whitespace_question(self):
        result = ask("   ")
        assert result == "Please provide a question."
