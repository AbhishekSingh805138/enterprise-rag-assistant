"""FastAPI application for the Enterprise RAG Assistant.

Endpoints:
    GET  /health  — Liveness check with collection stats
    POST /ask     — Query the RAG pipeline (supports streaming via SSE)
    POST /ingest  — Trigger document ingestion
    POST /eval    — Run RAGAS evaluation suite
"""
from __future__ import annotations

import json
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from config import settings, setup_logging

from api.models import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    UploadResponse,
)

logger = logging.getLogger(__name__)

VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings.validate()
    logger.info("Enterprise RAG Assistant API starting (v%s)", VERSION)
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Enterprise RAG Assistant",
    description="AI-powered question answering over enterprise documents with source citations.",
    version=VERSION,
    lifespan=lifespan,
)

app.state.limiter = limiter


# Rate limit error handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "detail": str(exc.detail)},
    )


# CORS — configurable via CORS_ORIGINS env var
_cors_origins = [o.strip() for o in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness check with collection stats."""
    try:
        from src.vectorstore.chroma_store import collection_stats
        stats = collection_stats()
        return HealthResponse(
            status="ok",
            collection=stats["collection"],
            document_count=stats["document_count"],
            version=VERSION,
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            collection=settings.chroma_collection,
            document_count=-1,
            version=VERSION,
        )


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------

def _resolve_mode(body: AskRequest) -> str:
    """Resolve 'auto' mode to either 'naive' or 'graph'."""
    if body.mode != "auto":
        return body.mode
    q = body.question.strip().lower()
    # Heuristic: use graph for complex queries
    word_count = len(q.split())
    has_comparison = any(w in q for w in ["compare", "versus", "vs", "difference between", "contrast"])
    has_multi = any(w in q for w in [" and ", "both", "also", "as well as", "additionally"])
    if word_count > 15 or has_comparison or has_multi:
        return "graph"
    return "naive"


def _ask_sync(body: AskRequest) -> AskResponse:
    """Run the query synchronously and return a full AskResponse."""
    from src.observability.cost_callback import CostCallbackHandler, is_idk_response

    resolved_mode = _resolve_mode(body)
    handler = CostCallbackHandler()
    start = time.perf_counter()
    node_latencies = None

    if resolved_mode == "graph":
        from src.graph.build_graph import get_graph

        graph = get_graph()
        import uuid
        tid = uuid.uuid4().hex[:12]
        config = {"configurable": {"thread_id": tid}, "callbacks": [handler]}
        result = graph.invoke(
            {
                "question": body.question,
                "retries": 0,
                "retriever_strategy": body.retriever_strategy,
            },
            config,
        )
        answer = result.get("generation", "No answer was generated.")

        # Capture per-node latencies from tracing
        try:
            from src.graph.tracing import get_last_run_latencies
            node_latencies = get_last_run_latencies()
        except Exception:
            pass
    else:
        from src.rag.naive_rag import build_naive_rag_chain

        chain = build_naive_rag_chain(
            k=body.top_k,
            filter=body.filter,
            retriever_strategy=body.retriever_strategy,
        )
        answer = chain.invoke(body.question, config={"callbacks": [handler]})

    latency_ms = (time.perf_counter() - start) * 1000
    is_idk = is_idk_response(answer)
    metrics = handler.flush(
        thread_id="api",
        question=body.question,
        latency_ms=latency_ms,
        retriever_strategy=body.retriever_strategy,
        mode=resolved_mode,
        is_idk=is_idk,
        node_latencies=node_latencies,
    )

    # Record to metrics store (best-effort)
    try:
        from src.observability.metrics_store import get_store
        get_store().record(metrics)
    except Exception:
        logger.debug("Failed to record API query metrics", exc_info=True)

    return AskResponse(
        answer=answer,
        question=body.question,
        mode=resolved_mode,
        retriever_strategy=body.retriever_strategy,
        cost_usd=metrics.estimated_cost_usd,
        latency_ms=latency_ms,
        tokens_used=metrics.total_tokens,
        node_latencies=node_latencies,
        is_idk=is_idk,
    )


def _stream_graph(body: AskRequest):
    """Generator yielding SSE events for graph mode streaming."""
    try:
        from src.observability.cost_callback import CostCallbackHandler

        handler = CostCallbackHandler()
        start = time.perf_counter()

        import uuid
        from src.graph.build_graph import get_graph

        graph = get_graph()
        tid = uuid.uuid4().hex[:12]
        config = {"configurable": {"thread_id": tid}, "callbacks": [handler]}

        answer = ""
        for step in graph.stream(
            {
                "question": body.question,
                "retries": 0,
                "retriever_strategy": body.retriever_strategy,
            },
            config,
        ):
            for node_name, state_update in step.items():
                event = {"type": "status", "node": node_name}
                if "generation" in state_update:
                    answer = state_update["generation"]
                    event["type"] = "token"
                    event["content"] = answer
                yield f"data: {json.dumps(event)}\n\n"

        latency_ms = (time.perf_counter() - start) * 1000

        # Capture per-node latencies and IDK status
        from src.observability.cost_callback import is_idk_response
        node_latencies = None
        try:
            from src.graph.tracing import get_last_run_latencies
            node_latencies = get_last_run_latencies()
        except Exception:
            pass

        is_idk = is_idk_response(answer)
        metrics = handler.flush(
            thread_id=tid, question=body.question,
            latency_ms=latency_ms, retriever_strategy=body.retriever_strategy,
            mode="graph",
            is_idk=is_idk,
            node_latencies=node_latencies,
        )

        try:
            from src.observability.metrics_store import get_store
            get_store().record(metrics)
        except Exception:
            pass

        done_event = {
            "type": "done",
            "answer": answer,
            "cost_usd": metrics.estimated_cost_usd,
            "latency_ms": latency_ms,
            "tokens_used": metrics.total_tokens,
            "node_latencies": node_latencies,
            "is_idk": is_idk,
        }
        yield f"data: {json.dumps(done_event)}\n\n"
    except Exception as e:
        logger.exception("Streaming graph error")
        err = {"type": "error", "message": _safe_error_detail(e)}
        yield f"data: {json.dumps(err)}\n\n"


def _stream_naive(body: AskRequest):
    """Generator yielding SSE events for naive mode streaming."""
    from src.observability.cost_callback import CostCallbackHandler
    from src.rag.naive_rag import build_naive_rag_chain

    handler = CostCallbackHandler()
    start = time.perf_counter()

    chain = build_naive_rag_chain(
        k=body.top_k,
        filter=body.filter,
        retriever_strategy=body.retriever_strategy,
    )

    full_answer = ""
    for chunk in chain.stream(body.question, config={"callbacks": [handler]}):
        full_answer += chunk
        event = {"type": "token", "content": chunk}
        yield f"data: {json.dumps(event)}\n\n"

    latency_ms = (time.perf_counter() - start) * 1000
    from src.observability.cost_callback import is_idk_response
    is_idk = is_idk_response(full_answer)
    metrics = handler.flush(
        thread_id="api", question=body.question,
        latency_ms=latency_ms, retriever_strategy=body.retriever_strategy,
        mode="naive",
        is_idk=is_idk,
    )

    try:
        from src.observability.metrics_store import get_store
        get_store().record(metrics)
    except Exception:
        pass

    done_event = {
        "type": "done",
        "answer": full_answer,
        "cost_usd": metrics.estimated_cost_usd,
        "latency_ms": latency_ms,
        "tokens_used": metrics.total_tokens,
        "is_idk": is_idk,
    }
    yield f"data: {json.dumps(done_event)}\n\n"


@app.post("/ask", response_model=AskResponse, responses={400: {"model": ErrorResponse}})
@limiter.limit("30/minute")
async def ask_endpoint(request: Request, body: AskRequest):
    """Query the RAG pipeline. Set stream=true for Server-Sent Events."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        if body.stream:
            resolved = _resolve_mode(body)
            gen = _stream_graph(body) if resolved == "graph" else _stream_naive(body)
            return StreamingResponse(gen, media_type="text/event-stream")

        return _ask_sync(body)
    except Exception as e:
        logger.exception("Ask endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse, responses={400: {"model": ErrorResponse}})
async def ingest_endpoint(body: IngestRequest):
    """Ingest documents from a file or directory path."""
    try:
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_path
        from src.vectorstore.chroma_store import add_chunks, collection_stats

        docs = load_path(body.path)
        chunks = chunk_documents(
            docs,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
        )
        added = add_chunks(chunks)
        stats = collection_stats()

        return IngestResponse(
            documents_loaded=len(docs),
            chunks_created=len(chunks),
            chunks_added=added,
            collection_total=stats["document_count"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Ingest endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


@app.post("/upload", response_model=UploadResponse, responses={400: {"model": ErrorResponse}})
async def upload_endpoint(file: UploadFile, department: str = "general"):
    """Upload a document file (PDF, TXT, or MD) and ingest it into the vector store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    try:
        from src.ingestion.chunker import chunk_documents
        from src.ingestion.loader import load_path
        from src.vectorstore.chroma_store import add_chunks, collection_stats

        # Save uploaded file to a temp directory so the loader can read it
        with tempfile.TemporaryDirectory() as tmpdir:
            dept_dir = Path(tmpdir) / department
            dept_dir.mkdir()
            dest = dept_dir / file.filename
            content = await file.read()
            dest.write_bytes(content)
            logger.info(
                "Upload: saved %s (%d bytes) to temp dir, department=%s",
                file.filename, len(content), department,
            )

            docs = load_path(tmpdir)

            # Rewrite source metadata: replace temp path with a stable
            # identifier so citations are meaningful and content-hash
            # deduplication works across re-uploads of the same file.
            stable_source = f"uploads/{department}/{file.filename}"
            for doc in docs:
                doc.metadata["source"] = stable_source

            chunks = chunk_documents(docs)
            added = add_chunks(chunks)
            stats = collection_stats()

        logger.info(
            "Upload complete: %s → %d docs, %d chunks, %d new (total %d)",
            file.filename, len(docs), len(chunks), added,
            stats["document_count"],
        )
        return UploadResponse(
            filename=file.filename,
            documents_loaded=len(docs),
            chunks_created=len(chunks),
            chunks_added=added,
            collection_total=stats["document_count"],
        )
    except Exception as e:
        logger.exception("Upload endpoint failed for file: %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /eval
# ---------------------------------------------------------------------------

@app.post("/eval", response_model=EvalResponse)
async def eval_endpoint(body: EvalRequest):
    """Run the RAGAS evaluation suite. This is a long-running operation."""
    try:
        from src.eval.ragas_eval import evaluate, load_eval_set
        from src.retrieval import get_retriever

        eval_set = load_eval_set(limit=body.limit)
        retriever = get_retriever(strategy=body.retriever_strategy)

        if body.mode == "graph":
            from src.graph.build_graph import ask as graph_ask
            answer_fn = lambda q: graph_ask(q, retriever_strategy=body.retriever_strategy)
        else:
            from src.rag.naive_rag import answer as naive_answer
            answer_fn = lambda q: naive_answer(q, retriever_strategy=body.retriever_strategy)

        start = time.time()
        scores = evaluate(answer_fn, retriever, eval_set=eval_set)
        duration = time.time() - start

        return EvalResponse(
            scores={k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()},
            items_evaluated=len(eval_set),
            mode=body.mode,
            retriever_strategy=body.retriever_strategy,
            duration_s=round(duration, 1),
        )
    except Exception as e:
        logger.exception("Eval endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))
