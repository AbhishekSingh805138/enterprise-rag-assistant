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


def build_graph(checkpointer=None):
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


def ask(question: str, thread_id: str = "default") -> str:
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"question": question, "retries": 0}, config)
    return result["generation"]
