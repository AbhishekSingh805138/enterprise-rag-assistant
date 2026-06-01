"""Shared state for the Corrective RAG (CRAG) graph.

Every node receives this dict, reads what it needs, and returns a partial
update. LangGraph merges the partial back into the running state.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    question: str               # original user question
    documents: list[Document]   # retrieved (and later, filtered) docs
    relevant: bool              # grader verdict: is the context good enough?
    web_fallback_used: bool     # did we fall back to web search?
    generation: str             # final answer
    retries: int                # loop guard
