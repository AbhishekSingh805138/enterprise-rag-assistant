"""ChromaDB vector store wrapper (persistent, OpenAI embeddings).

Exposes helpers to (a) build/add to the store from chunks and (b) get a
retriever for querying. Persistence lives at settings.chroma_dir so ingestion
and querying are separate processes.
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from config import settings


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent collection."""
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=_embeddings(),
        persist_directory=settings.chroma_dir,
    )


def add_chunks(chunks: list[Document]) -> int:
    """Embed and persist chunks. Returns the number added."""
    store = get_vectorstore()
    store.add_documents(chunks)
    return len(chunks)


def get_retriever(k: int | None = None, filter: dict | None = None) -> VectorStoreRetriever:
    """Return a retriever. `filter` enables metadata-filtered search,
    e.g. {"doc_type": "pdf"} or {"department": "legal"}."""
    search_kwargs: dict = {"k": k or settings.top_k}
    if filter:
        search_kwargs["filter"] = filter
    return get_vectorstore().as_retriever(search_kwargs=search_kwargs)
