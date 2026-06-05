"""Composed retriever: chains a first-stage retriever with a reranker.

This enables strategy combinations like hybrid + rerank, where the first
stage maximizes recall (hybrid BM25+dense) and the second stage maximizes
precision (LLM reranking).
"""
from __future__ import annotations

import logging

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from pydantic import Field, PrivateAttr

from config import settings

logger = logging.getLogger(__name__)


class ComposedRetriever(BaseRetriever):
    """Chains a first-stage retriever with LLM reranking."""

    k: int = Field(default=4, description="Number of final documents to return")
    first_stage_strategy: str = Field(default="hybrid", description="First stage retriever strategy")
    filter: dict | None = Field(default=None, description="Metadata filter for retriever")

    _first_stage: BaseRetriever | None = PrivateAttr(default=None)

    def _get_first_stage(self) -> BaseRetriever:
        """Lazily build the first-stage retriever."""
        if self._first_stage is None:
            from src.retrieval.factory import get_retriever
            # Fetch more candidates than final k for reranking
            fetch_k = min(15, self.k * 3)
            self._first_stage = get_retriever(
                strategy=self.first_stage_strategy,
                k=fetch_k,
                filter=self.filter,
            )
        return self._first_stage

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve via first stage, then rerank with LLM."""
        logger.info("Composed retrieval (%s + rerank): %s", self.first_stage_strategy, query[:120])

        # First stage: broad recall
        candidates = self._get_first_stage().invoke(query)
        if not candidates:
            return []

        # Second stage: LLM reranking
        from src.retrieval.rerank import RelevanceScore, _RERANK_SYSTEM

        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _score(doc: Document) -> tuple[Document, int]:
            try:
                scorer = llm.with_structured_output(RelevanceScore)
                result: RelevanceScore = scorer.invoke(
                    [
                        {"role": "system", "content": _RERANK_SYSTEM},
                        {
                            "role": "human",
                            "content": (
                                f"Question: {query}\n\n"
                                f"Document (source: {doc.metadata.get('filename', '?')}):\n"
                                f"{doc.page_content}"
                            ),
                        },
                    ]
                )
                return (doc, result.score)
            except Exception:
                return (doc, 5)

        scored: list[tuple[Document, int]] = []
        max_workers = min(settings.rerank_max_workers, len(candidates))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_score, doc): doc for doc in candidates}
            for future in as_completed(futures):
                scored.append(future.result())

        scored.sort(key=lambda x: x[1], reverse=True)
        result = [doc for doc, _ in scored[:self.k]]

        logger.info(
            "Composed: %d candidates → top-%d reranked",
            len(candidates), len(result),
        )
        return result


def build_composed_retriever(
    k: int | None = None,
    filter: dict | None = None,
    first_stage: str = "hybrid",
) -> ComposedRetriever:
    """Build and return a composed (first-stage + rerank) retriever."""
    return ComposedRetriever(
        k=k or settings.top_k,
        first_stage_strategy=first_stage,
        filter=filter,
    )
