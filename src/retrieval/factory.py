"""Retriever factory — single entry point for all retrieval strategies.

Usage:
    from src.retrieval import get_retriever

    retriever = get_retriever("dense")         # Phase 1 baseline
    retriever = get_retriever("hybrid")        # BM25 + dense + RRF
    retriever = get_retriever("multi_query")   # LLM query expansion
    retriever = get_retriever("rerank")        # dense + LLM reranking

All returned objects satisfy the LangChain BaseRetriever interface
(i.e., have an .invoke(query) -> list[Document] method).
"""
from __future__ import annotations

import logging

from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

STRATEGIES = ("dense", "hybrid", "multi_query", "rerank", "hybrid_rerank")


def get_retriever(
    strategy: str = "dense",
    k: int | None = None,
    filter: dict | None = None,
) -> BaseRetriever:
    """Return a retriever for the given strategy.

    Args:
        strategy: One of STRATEGIES.
        k: Number of final documents to return.
        filter: Metadata filter dict (passed to dense retrievers).

    Returns:
        A LangChain BaseRetriever.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown retrieval strategy {strategy!r}. "
            f"Choose from: {', '.join(STRATEGIES)}"
        )

    if strategy == "dense":
        from src.vectorstore.chroma_store import get_retriever as _dense
        return _dense(k=k, filter=filter)

    if strategy == "hybrid":
        from src.retrieval.hybrid import build_hybrid_retriever
        return build_hybrid_retriever(k=k, filter=filter)

    if strategy == "multi_query":
        from src.retrieval.multi_query import build_multi_query_retriever
        return build_multi_query_retriever(k=k, filter=filter)

    if strategy == "rerank":
        from src.retrieval.rerank import build_rerank_retriever
        return build_rerank_retriever(k=k, filter=filter)

    if strategy == "hybrid_rerank":
        from src.retrieval.composed import build_composed_retriever
        return build_composed_retriever(k=k, filter=filter, first_stage="hybrid")

    raise ValueError(f"Strategy {strategy!r} not implemented")  # unreachable
