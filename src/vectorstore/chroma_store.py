"""ChromaDB vector store wrapper (persistent, OpenAI embeddings).

Exposes helpers to (a) build/add to the store from chunks and (b) get a
retriever for querying. Persistence lives at settings.chroma_dir so ingestion
and querying are separate processes.

Key improvements over the skeleton:
  - Content-hash IDs prevent duplicate chunks on re-ingestion.
  - Singleton pattern avoids recreating embeddings/connections per call.
  - Error handling + logging throughout.
  - Phase 8: staleness detection (auto-refresh after interval).
  - Phase 8: document TTL (stale document detection & cleanup).
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from config import settings

logger = logging.getLogger(__name__)

# Module-level singletons — avoids recreating clients on every call.
_embeddings: OpenAIEmbeddings | None = None
_vectorstore: Chroma | None = None
_last_refresh: float = 0.0  # monotonic timestamp of last refresh
_lock = threading.Lock()


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    return _embeddings


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent collection (singleton).

    Automatically refreshes the connection if CHROMA_REFRESH_INTERVAL has
    elapsed since last open/refresh. Thread-safe via module lock.
    """
    global _vectorstore, _last_refresh
    with _lock:
        now = time.monotonic()
        if _vectorstore is not None:
            elapsed = now - _last_refresh
            if elapsed > settings.chroma_refresh_interval:
                logger.info(
                    "ChromaDB refresh: %.0fs since last refresh (interval=%ds)",
                    elapsed, settings.chroma_refresh_interval,
                )
                _vectorstore = None  # force re-creation

        if _vectorstore is None:
            _vectorstore = Chroma(
                collection_name=settings.chroma_collection,
                embedding_function=_get_embeddings(),
                persist_directory=settings.chroma_dir,
            )
            _last_refresh = now
            logger.info(
                "Opened Chroma collection '%s' at %s",
                settings.chroma_collection, settings.chroma_dir,
            )
        return _vectorstore


def refresh_store() -> None:
    """Force a refresh of the vectorstore connection on next access."""
    global _vectorstore, _last_refresh
    _vectorstore = None
    _last_refresh = 0.0
    logger.info("ChromaDB store marked for refresh")


def reset_store() -> None:
    """Reset singletons (useful for testing or after a full re-ingest)."""
    global _embeddings, _vectorstore, _last_refresh
    with _lock:
        _embeddings = None
        _vectorstore = None
        _last_refresh = 0.0


def _content_hash(text: str, metadata: dict) -> str:
    """Deterministic ID from content + origin so re-ingestion is idempotent.

    Uses filename + department (stable across uploads) when available,
    falling back to the full source path for CLI/ingest-based ingestion.
    """
    filename = metadata.get("filename", "")
    department = metadata.get("department", "")
    if filename:
        origin = f"{department}/{filename}"
    else:
        origin = metadata.get("source", "")
    start = str(metadata.get("start_index", ""))
    payload = f"{origin}::{start}::{text}"
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

    # Invalidate BM25 cache since corpus changed
    try:
        from src.retrieval.hybrid import reset_bm25_cache
        reset_bm25_cache()
    except ImportError:
        pass

    # Verify persistence: confirm the chunks are queryable.
    try:
        verify = store.get(ids=new_ids[:1], include=[])
        if not verify or not verify.get("ids"):
            logger.error(
                "Persistence verification FAILED — chunks may not have been stored"
            )
    except Exception:
        logger.warning("Could not verify chunk persistence", exc_info=True)

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


def get_stale_documents(max_age_days: int | None = None) -> list[str]:
    """Return IDs of documents older than *max_age_days*.

    Requires documents to have 'ingested_at' ISO-format metadata.
    Returns an empty list if TTL is disabled (max_age_days=0).
    """
    from datetime import datetime, timedelta, timezone

    days = max_age_days if max_age_days is not None else settings.document_ttl_days
    if days <= 0:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    store = get_vectorstore()
    stale_ids: list[str] = []

    try:
        result = store.get(include=["metadatas"])
        if not result or not result.get("ids"):
            return []

        for doc_id, meta in zip(result["ids"], result["metadatas"]):
            ingested_at = meta.get("ingested_at", "")
            if not ingested_at:
                continue
            try:
                doc_ts = datetime.fromisoformat(ingested_at)
                if doc_ts < cutoff:
                    stale_ids.append(doc_id)
            except (ValueError, TypeError):
                logger.debug("Invalid ingested_at for doc %s: %s", doc_id, ingested_at)
    except Exception:
        logger.exception("Failed to check for stale documents")

    logger.info("Found %d stale document(s) older than %d days", len(stale_ids), days)
    return stale_ids


def delete_stale_documents(max_age_days: int | None = None) -> int:
    """Delete documents older than *max_age_days*. Returns count deleted."""
    stale_ids = get_stale_documents(max_age_days)
    if not stale_ids:
        return 0

    store = get_vectorstore()
    try:
        store.delete(ids=stale_ids)
        logger.info("Deleted %d stale document(s)", len(stale_ids))

        # Invalidate BM25 cache since corpus changed
        try:
            from src.retrieval.hybrid import reset_bm25_cache
            reset_bm25_cache()
        except ImportError:
            pass

        return len(stale_ids)
    except Exception:
        logger.exception("Failed to delete stale documents")
        return 0


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
