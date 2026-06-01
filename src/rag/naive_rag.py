"""Phase 1: naive RAG baseline.

load -> chunk -> embed -> retrieve top-k -> stuff into prompt -> generate.
This is intentionally simple. It exists to establish a baseline you will
measure (Phase 2) and beat (Phase 3+). Built with LCEL so the pipe operator
makes the data flow explicit.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import settings
from src.vectorstore.chroma_store import get_retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the question using ONLY "
    "the context below. If the context does not contain enough information to "
    "answer the question, respond with: "
    '"I don\'t have enough information in the available documents to answer this '
    'question." Do NOT guess or invent facts.\n\n'
    "For every claim you make, cite the source filename in parentheses — e.g. "
    "(handbook.md). If the answer draws from multiple sources, cite each one.\n\n"
    "Context:\n{context}"
)

_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(no documents retrieved)"
    return "\n\n---\n\n".join(
        f"[source: {d.metadata.get('filename', d.metadata.get('source', '?'))}]\n{d.page_content}"
        for d in docs
    )


def build_naive_rag_chain(k: int | None = None, filter: dict | None = None) -> Runnable:
    """Compose the LCEL chain: {context, question} -> prompt -> llm -> str."""
    retriever = get_retriever(k=k, filter=filter)
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    context_and_question = RunnableParallel(
        context=retriever | _format_docs,
        question=RunnablePassthrough(),
    )

    return context_and_question | _prompt | llm | StrOutputParser()


def answer(question: str, k: int | None = None, filter: dict | None = None) -> str:
    """Answer a question using naive RAG. Returns the LLM response string."""
    if not question or not question.strip():
        return "Please provide a question."

    logger.info("Naive RAG query: %s", question[:120])
    try:
        result = build_naive_rag_chain(k=k, filter=filter).invoke(question)
        logger.info("Naive RAG response length: %d chars", len(result))
        return result
    except Exception:
        logger.exception("Naive RAG failed for query: %s", question[:120])
        raise
