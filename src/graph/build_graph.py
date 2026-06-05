"""Build and compile the Corrective RAG (CRAG) graph.

Flow (Phase 5 — with planner):

    START -> planner -> (conditional: route_after_plan)
      |
      |- simple question:
      |    retrieve -> grade_documents -> (conditional)
      |                                    |- relevant -> generate -> critic -> END
      |                                    |- retry     -> transform_query -> retrieve
      |                                    |- exhausted -> web_search -> generate -> critic -> END
      |
      |- multi-part question:
           process_sub_query -> (conditional: has_more_sub_queries?)
             |- yes -> process_sub_query (loop)
             |- no  -> synthesize -> critic -> END

The planner detects multi-part questions and decomposes them into sub-queries.
Simple questions go through the original CRAG flow unchanged. Multi-part
questions are processed sequentially, then synthesized into a single response.
The critic node verifies claims in the final answer regardless of path.
"""
from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from config import settings

from src.graph.nodes import (
    critic,
    decide_after_grade,
    generate,
    grade_documents,
    retrieve,
    transform_query,
    web_search,
)
from src.graph.planner import (
    has_more_sub_queries,
    planner,
    process_sub_query,
    route_after_plan,
    synthesize,
)
from src.graph.scope_detector import scope_check
from src.graph.state import RAGState
from src.graph.tool_node import tool_router

logger = logging.getLogger(__name__)

# Module-level cached graph — compiled once, reused across calls.
_compiled_graph = None
_sqlite_conn: sqlite3.Connection | None = None


def _get_sqlite_checkpointer() -> SqliteSaver:
    """Create a SQLite checkpointer for persistent graph state."""
    global _sqlite_conn
    checkpoint_dir = Path(settings.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    db_path = checkpoint_dir / "graph_checkpoints.db"
    _sqlite_conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(_sqlite_conn)
    logger.info("SQLite checkpointer initialized at %s", db_path)
    return saver


def _route_after_scope(state: dict) -> str:
    """Route based on scope check: in-scope -> planner, out-of-scope -> generate."""
    if state.get("in_scope", True):
        return "planner"
    return "generate"


def build_graph(checkpointer=None):
    """Build and compile the CRAG StateGraph with Phase 5 planner."""
    builder = StateGraph(RAGState)

    # Phase 8: scope detection (before planner)
    builder.add_node("scope_check", scope_check)

    # Phase 8: tool routing (optional, gated by ENABLE_TOOLS)
    if settings.enable_tools:
        builder.add_node("tool_router", tool_router)

    # Phase 5: planner and multi-part processing nodes
    builder.add_node("planner", planner)
    builder.add_node("process_sub_query", process_sub_query)
    builder.add_node("synthesize", synthesize)

    # Existing CRAG nodes
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("transform_query", transform_query)
    builder.add_node("web_search", web_search)
    builder.add_node("generate", generate)
    builder.add_node("critic", critic)

    # Entry: scope check → planner or generate (IDK for out-of-scope)
    builder.add_edge(START, "scope_check")
    if settings.enable_tools:
        builder.add_conditional_edges(
            "scope_check",
            _route_after_scope,
            {
                "planner": "tool_router",
                "generate": "generate",
            },
        )
        builder.add_edge("tool_router", "planner")
    else:
        builder.add_conditional_edges(
            "scope_check",
            _route_after_scope,
            {
                "planner": "planner",
                "generate": "generate",
            },
        )
    builder.add_conditional_edges(
        "planner",
        route_after_plan,
        {
            "retrieve": "retrieve",              # simple question → existing CRAG
            "process_sub_query": "process_sub_query",  # multi-part → sub-query loop
        },
    )

    # Simple path: existing CRAG flow (unchanged)
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
    builder.add_edge("generate", "critic")  # critic verifies claims before output

    # Multi-part path: sub-query loop → synthesize
    builder.add_conditional_edges(
        "process_sub_query",
        has_more_sub_queries,
        {
            "process_sub_query": "process_sub_query",  # loop back
            "synthesize": "synthesize",                 # all done
        },
    )
    builder.add_edge("synthesize", "critic")  # critic checks synthesized answer

    # Both paths end after critic
    builder.add_edge("critic", END)

    return builder.compile(checkpointer=checkpointer or _get_sqlite_checkpointer())


def get_graph(checkpointer=None):
    """Return the cached compiled graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph(checkpointer=checkpointer)
        logger.info("CRAG graph compiled and cached")
    return _compiled_graph


def reset_graph() -> None:
    """Reset the cached graph and close the SQLite connection."""
    global _compiled_graph, _sqlite_conn
    _compiled_graph = None
    if _sqlite_conn is not None:
        try:
            _sqlite_conn.close()
        except Exception:
            pass
        _sqlite_conn = None


def ask(
    question: str,
    thread_id: str | None = None,
    retriever_strategy: str = "dense",
) -> str:
    """Run a question through the CRAG graph. Returns the answer string."""
    if not question or not question.strip():
        return "Please provide a question."

    tid = thread_id or uuid.uuid4().hex[:12]
    logger.info("CRAG query (thread=%s, retriever=%s): %s", tid, retriever_strategy, question[:120])

    from src.observability.cost_callback import CostCallbackHandler

    handler = CostCallbackHandler()
    graph = get_graph()
    config = {"configurable": {"thread_id": tid}, "callbacks": [handler]}

    try:
        start = time.perf_counter()
        result = graph.invoke(
            {"question": question, "retries": 0, "retriever_strategy": retriever_strategy},
            config,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        answer = result.get("generation", "No answer was generated.")
        logger.info("CRAG answer (thread=%s): %d chars", tid, len(answer))

        # Record metrics — never let metrics failures break the query
        try:
            from src.graph.tracing import get_last_run_latencies
            from src.observability.cost_callback import is_idk_response

            node_lats = get_last_run_latencies()
            grader_rejected = 1 if not result.get("relevant", True) else 0
            metrics = handler.flush(
                thread_id=tid,
                question=question,
                latency_ms=latency_ms,
                retriever_strategy=retriever_strategy,
                mode="graph",
                is_idk=is_idk_response(answer),
                grader_rejected=grader_rejected,
                node_latencies=node_lats,
            )
            from src.observability.metrics_store import get_store
            get_store().record(metrics)
            logger.info(
                "Query metrics: $%.5f, %d tokens, %.0fms, idk=%s",
                metrics.estimated_cost_usd, metrics.total_tokens, metrics.latency_ms,
                metrics.is_idk,
            )
        except Exception:
            logger.debug("Failed to record query metrics", exc_info=True)

        return answer
    except Exception:
        logger.exception("CRAG graph failed for: %s", question[:120])
        raise
