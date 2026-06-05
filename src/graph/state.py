"""Shared state for the Corrective RAG (CRAG) graph.

Every node receives this dict, reads what it needs, and returns a partial
update. LangGraph merges the partial back into the running state.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    question: str               # original user question (or current sub-question)
    documents: list[Document]   # retrieved (and later, filtered) docs
    relevant: bool              # grader verdict: is the context good enough?
    web_fallback_used: bool     # did we fall back to web search?
    generation: str             # final answer (post-critic)
    retries: int                # loop guard
    retriever_strategy: str     # Phase 3: which retrieval strategy to use
    critic_passed: bool         # Phase 4: did the critic verify all claims?
    claims_removed: int         # Phase 4: number of unsupported claims stripped
    # Phase 5: multi-agent decomposition
    original_question: str      # preserved original question (before decomposition)
    sub_questions: list[str]    # planner output: decomposed sub-queries
    sub_answers: list[str]      # intermediate answers per sub-query
    is_multi_part: bool         # planner verdict: does question need decomposition?
    current_sub_idx: int        # loop index for sequential sub-query processing
    # Phase 8: graph intelligence
    in_scope: bool              # scope detector verdict
    all_sub_documents: list[Document]  # accumulated docs across sub-queries
    tool_results: list[str]     # Phase 8: results from tool invocations
