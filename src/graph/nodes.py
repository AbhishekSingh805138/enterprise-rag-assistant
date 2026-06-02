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
from src.graph.tracing import traced
from src.retrieval import get_retriever

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def _llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )


# --- Node: retrieve --------------------------------------------------------

@traced
def retrieve(state: dict) -> dict:
    """Retrieve documents for the current question."""
    question = state["question"]
    strategy = state.get("retriever_strategy", "dense")
    logger.info("Retrieve node (strategy=%s) — query: %s", strategy, question[:120])
    try:
        docs = get_retriever(strategy=strategy).invoke(question)
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


@traced
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


@traced
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

@traced
def web_search(state: dict) -> dict:
    """Search the web via Tavily when corpus retrieval is insufficient.

    Appends web results as Documents alongside any existing retrieved docs.
    Gracefully degrades to a placeholder if TAVILY_API_KEY is not set.
    """
    question = state["question"]
    existing_docs = state.get("documents", [])

    if not settings.tavily_api_key:
        logger.warning("Web search skipped — TAVILY_API_KEY not set")
        fallback = Document(
            page_content="(web search unavailable — no API key configured)",
            metadata={"filename": "web_fallback", "source": "web"},
        )
        return {
            "documents": existing_docs + [fallback],
            "web_fallback_used": True,
        }

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        results = client.search(query=question, max_results=3)

        web_docs = [
            Document(
                page_content=r["content"],
                metadata={
                    "source": r["url"],
                    "filename": "web_search",
                    "title": r.get("title", ""),
                },
            )
            for r in results.get("results", [])
        ]
        logger.info("Web search returned %d result(s) for: %s", len(web_docs), question[:80])
        return {
            "documents": existing_docs + web_docs,
            "web_fallback_used": True,
        }
    except Exception:
        logger.exception("Web search failed for: %s", question[:80])
        fallback = Document(
            page_content="(web search failed — see logs for details)",
            metadata={"filename": "web_fallback", "source": "web"},
        )
        return {
            "documents": existing_docs + [fallback],
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


@traced
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


# --- Node: critic (claim verification) ------------------------------------

class ClaimVerdict(BaseModel):
    """Structured output for claim-level verification."""
    supported_claims: list[str] = Field(
        description="List of claims that ARE supported by the source documents"
    )
    unsupported_claims: list[str] = Field(
        description="List of claims that are NOT supported by the source documents"
    )


_critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a fact-checking critic for an enterprise knowledge assistant. "
            "Your job is to verify every factual claim in the generated answer "
            "against the source documents.\n\n"
            "Rules:\n"
            "1. A claim is SUPPORTED only if the source documents explicitly state "
            "or directly imply the information.\n"
            "2. A claim is UNSUPPORTED if it adds details, numbers, or facts not "
            "found in the sources — even if the claim seems reasonable.\n"
            "3. Paraphrases of source content are SUPPORTED.\n"
            "4. General framing sentences (e.g. 'According to the documents...') "
            "are SUPPORTED — they are not factual claims.\n"
            "5. Citations like '(handbook.md)' are not claims themselves.\n\n"
            "Extract each distinct factual claim from the answer, then classify it.",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Generated Answer:\n{answer}\n\n"
            "Source Documents:\n{context}",
        ),
    ]
)

_rewrite_prompt_critic = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are rewriting an answer to remove unsupported claims. "
            "Keep ONLY the supported claims and their citations. "
            "Maintain natural flow and coherence. If no supported claims remain, "
            'respond with exactly: "I don\'t have enough information in the '
            'available documents to answer this question."\n\n'
            "Do NOT add any new information. Only use what is in the supported claims.",
        ),
        (
            "human",
            "Original question: {question}\n\n"
            "Supported claims:\n{supported}\n\n"
            "Rewrite the answer using ONLY these supported claims:",
        ),
    ]
)


@traced
def critic(state: dict) -> dict:
    """Verify claims in the generated answer against source documents.

    If all claims are supported, passes the answer through unchanged.
    If some claims are unsupported, rewrites the answer without them.
    """
    question = state["question"]
    generation = state.get("generation", "")
    documents = state.get("documents", [])

    # Skip critic for "I don't know" answers
    idk_phrases = ["don't have enough information", "cannot answer", "no information available"]
    if any(phrase in generation.lower() for phrase in idk_phrases):
        logger.info("Critic: skipping — answer is already an IDK response")
        return {"critic_passed": True, "claims_removed": 0}

    if not documents:
        logger.warning("Critic: no source documents to verify against")
        return {"critic_passed": True, "claims_removed": 0}

    try:
        context = "\n\n".join(
            f"[{d.metadata.get('filename', '?')}] {d.page_content}"
            for d in documents
        )

        verifier = _critic_prompt | _llm().with_structured_output(ClaimVerdict)
        verdict: ClaimVerdict = verifier.invoke(
            {"question": question, "answer": generation, "context": context}
        )

        n_supported = len(verdict.supported_claims)
        n_unsupported = len(verdict.unsupported_claims)
        logger.info(
            "Critic: %d supported, %d unsupported claims",
            n_supported, n_unsupported,
        )

        if n_unsupported == 0:
            # All claims verified — pass through
            return {"critic_passed": True, "claims_removed": 0}

        if n_supported == 0:
            # Nothing is supported — return IDK
            logger.warning("Critic: no supported claims — replacing with IDK")
            return {
                "generation": (
                    "I don't have enough information in the available documents "
                    "to answer this question."
                ),
                "critic_passed": False,
                "claims_removed": n_unsupported,
            }

        # Some claims unsupported — rewrite without them
        supported_text = "\n".join(f"- {c}" for c in verdict.supported_claims)
        rewriter = _rewrite_prompt_critic | _llm() | StrOutputParser()
        rewritten = rewriter.invoke(
            {"question": question, "supported": supported_text}
        )
        logger.info(
            "Critic: rewrote answer (removed %d unsupported claims, %d chars → %d chars)",
            n_unsupported, len(generation), len(rewritten),
        )
        return {
            "generation": rewritten,
            "critic_passed": False,
            "claims_removed": n_unsupported,
        }

    except Exception:
        logger.exception("Critic failed — passing answer through unchanged")
        return {"critic_passed": True, "claims_removed": 0}


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
