"""Phase 1: naive RAG baseline.

load -> chunk -> embed -> retrieve top-k -> stuff into prompt -> generate.
This is intentionally simple. It exists to establish a baseline you will
measure (Phase 2) and beat (Phase 3+). Built with LCEL so the pipe operator
makes the data flow explicit.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import settings
from src.vectorstore.chroma_store import get_retriever

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the question using ONLY "
    "the context below. If the context does not contain the answer, say you "
    "don't know -- do not invent facts. Cite the source filename in parentheses "
    "after each claim.\n\n"
    "Context:\n{context}"
)

_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[source: {d.metadata.get('filename', d.metadata.get('source', '?'))}]\n{d.page_content}"
        for d in docs
    )


def build_naive_rag_chain(k: int | None = None) -> Runnable:
    """Compose the LCEL chain: {context, question} -> prompt -> llm -> str."""
    retriever = get_retriever(k=k)
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    # Retrieve and format context while passing the question straight through.
    context_and_question = RunnableParallel(
        context=retriever | _format_docs,
        question=RunnablePassthrough(),
    )

    return context_and_question | _prompt | llm | StrOutputParser()


def answer(question: str, k: int | None = None) -> str:
    return build_naive_rag_chain(k=k).invoke(question)
