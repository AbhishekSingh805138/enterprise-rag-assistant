"""Build and compile the Corrective RAG (CRAG) graph.

Flow:

    START -> retrieve -> grade_documents -> (conditional)
                                              |- relevant -> generate -> END
                                              |- retry     -> transform_query -> retrieve
                                              |- exhausted -> web_search -> generate -> END

The conditional edge after grading is the whole reason to use LangGraph
instead of a linear LCEL chain: the model's verdict steers the path, and the
graph can loop back to re-retrieve. The checkpointer persists state per
thread_id so you get resumability and human-in-the-loop for free.
"""
from __future__ import annotations

import logging
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    decide_after_grade,
    generate,
    grade_documents,
    retrieve,
    transform_query,
    web_search,
)
from src.graph.state import RAGState

logger = logging.getLogger(__name__)

# Module-level cached graph — compiled once, reused across calls.
_compiled_graph = None


def build_graph(checkpointer=None):
    """Build and compile the CRAG StateGraph."""
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("transform_query", transform_query)
    builder.add_node("web_search", web_search)
    builder.add_node("generate", generate)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade_documents")

    builder.add_conditional_edges(
        "grade_documents",
        decide_after_grade,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "web_search",
        },
    )

    builder.add_edge("transform_query", "retrieve")  # the corrective loop
    builder.add_edge("web_search", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def get_graph():
    """Return the cached compiled graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
        logger.info("CRAG graph compiled and cached")
    return _compiled_graph


def ask(question: str, thread_id: str | None = None) -> str:
    """Run a question through the CRAG graph. Returns the answer string."""
    if not question or not question.strip():
        return "Please provide a question."

    tid = thread_id or uuid.uuid4().hex[:12]
    logger.info("CRAG query (thread=%s): %s", tid, question[:120])

    graph = get_graph()
    config = {"configurable": {"thread_id": tid}}

    try:
        result = graph.invoke({"question": question, "retries": 0}, config)
        answer = result.get("generation", "No answer was generated.")
        logger.info("CRAG answer (thread=%s): %d chars", tid, len(answer))
        return answer
    except Exception:
        logger.exception("CRAG graph failed for: %s", question[:120])
        raise
