"""ChromaDB vector store wrapper (persistent, OpenAI embeddings).

Exposes helpers to (a) build/add to the store from chunks and (b) get a
retriever for querying. Persistence lives at settings.chroma_dir so ingestion
and querying are separate processes.

Key improvements over the skeleton:
  - Content-hash IDs prevent duplicate chunks on re-ingestion.
  - Singleton pattern avoids recreating embeddings/connections per call.
  - Error handling + logging throughout.
"""
from __future__ import annotations

import hashlib
import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from config import settings

logger = logging.getLogger(__name__)

# Module-level singletons — avoids recreating clients on every call.
_embeddings: OpenAIEmbeddings | None = None
_vectorstore: Chroma | None = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    return _embeddings


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent collection (singleton)."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=_get_embeddings(),
            persist_directory=settings.chroma_dir,
        )
        logger.info(
            "Opened Chroma collection '%s' at %s",
            settings.chroma_collection, settings.chroma_dir,
        )
    return _vectorstore


def reset_store() -> None:
    """Reset singletons (useful for testing or after a full re-ingest)."""
    global _embeddings, _vectorstore
    _embeddings = None
    _vectorstore = None


def _content_hash(text: str, metadata: dict) -> str:
    """Deterministic ID from content + source so re-ingestion is idempotent."""
    source = metadata.get("source", "")
    start = str(metadata.get("start_index", ""))
    payload = f"{source}::{start}::{text}"
    return hashlib.sha256(payload.encode()).hexdigest()


def add_chunks(chunks: list[Document]) -> int:
    """Embed and persist chunks. Deduplicates by content hash.

    Returns the number of *new* chunks actually added.
    """
    if not chunks:
        logger.warning("add_chunks called with empty list")
        return 0

    store = get_vectorstore()

    # Build deterministic IDs to prevent duplicates.
    ids = [_content_hash(c.page_content, c.metadata) for c in chunks]

    # Check which IDs already exist to skip them.
    existing = set()
    try:
        result = store.get(ids=ids, include=[])
        existing = set(result["ids"]) if result and result.get("ids") else set()
    except Exception:
        logger.debug("Could not check existing IDs — will attempt full add")

    new_chunks = []
    new_ids = []
    for chunk, cid in zip(chunks, ids):
        if cid not in existing:
            new_chunks.append(chunk)
            new_ids.append(cid)

    if not new_chunks:
        logger.info("All %d chunk(s) already exist — nothing to add", len(chunks))
        return 0

    store.add_documents(new_chunks, ids=new_ids)
    logger.info(
        "Added %d new chunk(s) to Chroma (%d skipped as duplicates)",
        len(new_chunks), len(chunks) - len(new_chunks),
    )
    return len(new_chunks)


def get_retriever(
    k: int | None = None,
    filter: dict | None = None,
) -> VectorStoreRetriever:
    """Return a retriever with optional metadata filtering.

    Examples:
        get_retriever(filter={"department": "legal"})
        get_retriever(k=8, filter={"access_level": "internal"})
    """
    search_kwargs: dict = {"k": k or settings.top_k}
    if filter:
        search_kwargs["filter"] = filter
    return get_vectorstore().as_retriever(search_kwargs=search_kwargs)


def collection_stats() -> dict:
    """Return basic stats about the current collection."""
    store = get_vectorstore()
    try:
        count = store._collection.count()
    except Exception:
        count = -1
    return {
        "collection": settings.chroma_collection,
        "persist_directory": settings.chroma_dir,
        "document_count": count,
    }
