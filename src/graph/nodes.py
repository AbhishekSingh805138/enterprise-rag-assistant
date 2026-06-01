"""Node functions for the Corrective RAG graph.

Each node is a plain function: (state) -> partial state update.
Keeping nodes small and pure makes them unit-testable and easy to trace
in LangSmith.

The grader uses structured output (Pydantic) so the model returns a typed
boolean rather than free text you have to parse.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import settings
from src.vectorstore.chroma_store import get_retriever

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def _llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )


# --- Node: retrieve --------------------------------------------------------

def retrieve(state: dict) -> dict:
    """Retrieve documents for the current question."""
    question = state["question"]
    logger.info("Retrieve node — query: %s", question[:120])
    try:
        docs = get_retriever().invoke(question)
        logger.info("Retrieved %d document(s)", len(docs))
    except Exception:
        logger.exception("Retrieval failed for: %s", question[:120])
        docs = []
    return {"documents": docs, "retries": state.get("retries", 0)}


# --- Node: grade documents -------------------------------------------------

class GradeResult(BaseModel):
    """Structured verdict from the relevance grader."""
    relevant: bool = Field(
        description="True if the documents contain enough information to answer the question."
    )


_grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether the retrieved documents are "
            "relevant and sufficient to answer the user's question.\n\n"
            "Return relevant=true ONLY if the documents clearly contain "
            "information that can be used to answer the question. "
            "If the documents are off-topic, too vague, or don't address the "
            "question, return relevant=false.",
        ),
        ("human", "Question:\n{question}\n\nDocuments:\n{context}"),
    ]
)


def grade_documents(state: dict) -> dict:
    """Grade whether retrieved documents are relevant to the question."""
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        logger.warning("No documents to grade")
        return {"relevant": False}

    try:
        grader = _grade_prompt | _llm().with_structured_output(GradeResult)
        context = "\n\n".join(d.page_content for d in documents)
        verdict: GradeResult = grader.invoke(
            {"question": question, "context": context}
        )
        logger.info("Grader verdict: relevant=%s", verdict.relevant)
        return {"relevant": verdict.relevant}
    except Exception:
        logger.exception("Grading failed — defaulting to relevant=True to avoid loop")
        return {"relevant": True}


# --- Node: transform query (corrective step) -------------------------------

_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query rewriter. Rewrite the user's question to be "
            "clearer and more retrievable. Expand acronyms, add synonyms, "
            "and rephrase for better semantic search. "
            "Return ONLY the rewritten question, nothing else.",
        ),
        ("human", "{question}"),
    ]
)


def transform_query(state: dict) -> dict:
    """Rewrite the question for better retrieval on the next attempt."""
    original = state["question"]
    retries = state.get("retries", 0) + 1

    try:
        rewriter = _rewrite_prompt | _llm(temperature=0.3) | StrOutputParser()
        rewritten = rewriter.invoke({"question": original})
        logger.info("Query rewritten (retry %d): %s → %s", retries, original[:80], rewritten[:80])
    except Exception:
        logger.exception("Query rewrite failed — keeping original")
        rewritten = original

    return {"question": rewritten, "retries": retries}


# --- Node: web search fallback (stub) --------------------------------------

def web_search(state: dict) -> dict:
    """Stub. In Phase 5 wire this to Tavily/SerpAPI and append results as
    Documents. For now it just flags that corpus retrieval was insufficient."""
    logger.info("Web search fallback triggered (stub — not yet wired)")
    fallback = Document(
        page_content="(web search not yet wired up — see Phase 5)",
        metadata={"filename": "web_fallback", "source": "web"},
    )
    return {
        "documents": state.get("documents", []) + [fallback],
        "web_fallback_used": True,
    }


# --- Node: generate --------------------------------------------------------

_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an enterprise knowledge assistant. Answer the question "
            "using ONLY the provided context. Follow these rules strictly:\n\n"
            "1. If the context does not contain enough information to answer, "
            'respond with: "I don\'t have enough information in the available '
            'documents to answer this question."\n'
            "2. Do NOT guess, infer beyond what's stated, or invent facts.\n"
            "3. Cite the source filename in parentheses after each claim — "
            "e.g. (handbook.md).\n"
            "4. If the answer draws from multiple sources, cite each one.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def generate(state: dict) -> dict:
    """Generate the final cited answer from retrieved context."""
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        logger.warning("Generate called with no documents")
        return {
            "generation": (
                "I don't have enough information in the available documents "
                "to answer this question."
            )
        }

    try:
        chain = _gen_prompt | _llm() | StrOutputParser()
        context = "\n\n".join(
            f"[{d.metadata.get('filename', '?')}] {d.page_content}"
            for d in documents
        )
        answer = chain.invoke({"question": question, "context": context})
        logger.info("Generated answer: %d chars", len(answer))
        return {"generation": answer}
    except Exception:
        logger.exception("Generation failed for: %s", question[:120])
        return {
            "generation": "An error occurred while generating the answer. Please try again."
        }


# --- Router: decide what to do after grading -------------------------------

def decide_after_grade(state: dict) -> str:
    """Conditional edge. Returns the name of the next node."""
    if state.get("relevant"):
        logger.debug("Route → generate (context is relevant)")
        return "generate"
    if state.get("retries", 0) >= MAX_RETRIES:
        logger.debug("Route → web_search (retries exhausted)")
        return "web_search"
    logger.debug("Route → transform_query (context weak, retrying)")
    return "transform_query"
