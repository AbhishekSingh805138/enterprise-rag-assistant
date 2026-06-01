"""Node functions for the Corrective RAG graph.

Each node is a plain function: (state) -> partial state update.
Keeping nodes small and pure makes them unit-testable and easy to trace
in LangSmith.

The grader uses structured output (Pydantic) so the model returns a typed
boolean rather than free text you have to parse.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import settings
from src.vectorstore.chroma_store import get_retriever

MAX_RETRIES = 1


def _llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )


# --- Node: retrieve --------------------------------------------------------
def retrieve(state: dict) -> dict:
    docs = get_retriever().invoke(state["question"])
    return {"documents": docs, "retries": state.get("retries", 0)}


# --- Node: grade documents -------------------------------------------------
class _Grade(BaseModel):
    """Structured verdict from the relevance grader."""
    relevant: bool = Field(description="True if the documents can answer the question.")


_grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether the retrieved documents are "
            "relevant and sufficient to answer the user's question. "
            "Return relevant=true only if they clearly are.",
        ),
        ("human", "Question:\n{question}\n\nDocuments:\n{context}"),
    ]
)


def grade_documents(state: dict) -> dict:
    grader = _grade_prompt | _llm().with_structured_output(_Grade)
    context = "\n\n".join(d.page_content for d in state["documents"])
    verdict: _Grade = grader.invoke(
        {"question": state["question"], "context": context}
    )
    return {"relevant": verdict.relevant}


# --- Node: transform query (corrective step) -------------------------------
_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's question to be clearer and more retrievable "
            "(expand acronyms, add synonyms). Return only the rewritten question.",
        ),
        ("human", "{question}"),
    ]
)


def transform_query(state: dict) -> dict:
    rewriter = _rewrite_prompt | _llm(temperature=0.3) | StrOutputParser()
    better = rewriter.invoke({"question": state["question"]})
    return {"question": better, "retries": state.get("retries", 0) + 1}


# --- Node: web search fallback (stub) --------------------------------------
def web_search(state: dict) -> dict:
    """Stub. In Phase 5 wire this to Tavily/SerpAPI and append results as
    Documents. For now it just flags that corpus retrieval was insufficient."""
    fallback = Document(
        page_content="(web search not yet wired up -- see Phase 5)",
        metadata={"filename": "web_fallback"},
    )
    return {
        "documents": state["documents"] + [fallback],
        "web_fallback_used": True,
    }


# --- Node: generate --------------------------------------------------------
_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using ONLY the context. Cite source filenames in "
            "parentheses. If unsure, say so.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def generate(state: dict) -> dict:
    chain = _gen_prompt | _llm() | StrOutputParser()
    context = "\n\n".join(
        f"[{d.metadata.get('filename', '?')}] {d.page_content}"
        for d in state["documents"]
    )
    out = chain.invoke({"question": state["question"], "context": context})
    return {"generation": out}


# --- Router: decide what to do after grading -------------------------------
def decide_after_grade(state: dict) -> str:
    """Conditional edge. Returns the name of the next node."""
    if state.get("relevant"):
        return "generate"
    if state.get("retries", 0) >= MAX_RETRIES:
        # Out of retries -> fall back to web search rather than loop forever.
        return "web_search"
    return "transform_query"
