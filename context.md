# Enterprise RAG Assistant - Project Context

**Last updated:** 2026-06-01 (Session 2)
**Updated by:** Senior AI/ML Developer — Phase 1 hardening complete

---

## Project Overview

An internal AI assistant that answers complex questions over a company's private document corpus (policies, contracts, technical docs) with **grounded, source-cited answers**. Uses a multi-agent retrieval pipeline that retrieves, grades, self-corrects, and verifies claims against sources before answering.

**Tech Stack:** Python 3.11+, LangChain 1.0, LangGraph 1.0, ChromaDB, OpenAI (gpt-4o-mini + text-embedding-3-small), RAGAS, FastAPI

**Key Documents:** PRD.md, TRD.md, IMPLEMENTATION_PLAN.md

---

## Architecture

```
Ingestion (offline):  Loaders (pdf/txt/md) -> Chunker -> OpenAI Embeddings -> ChromaDB
Query - Naive:        Question -> Retriever -> LLM (LCEL chain) -> Cited Answer
Query - Agentic:      Question -> Retrieve -> Grade -> (Rewrite+Retry | Web Fallback) -> Generate -> Answer
```

**Modules:**
- `config.py` — env-driven settings (dataclass + .env), centralized logging setup, extended validation
- `src/ingestion/loader.py` — PDF/TXT/MD loaders with full metadata (source, filename, doc_type, department, access_level)
- `src/ingestion/chunker.py` — RecursiveCharacterTextSplitter (1000/150) with logging
- `src/vectorstore/chroma_store.py` — ChromaDB wrapper with singleton pattern, content-hash deduplication, collection_stats()
- `src/rag/naive_rag.py` — Phase 1 LCEL baseline chain with "I don't know" enforcement
- `src/graph/state.py` — RAGState TypedDict for LangGraph
- `src/graph/nodes.py` — CRAG nodes (retrieve, grade, transform_query, web_search stub, generate) with error handling
- `src/graph/build_graph.py` — Compiles CRAG StateGraph with cached singleton, unique thread IDs
- `src/eval/ragas_eval.py` — RAGAS evaluation harness (scaffolded, 1 Q/A pair — needs 50+)
- `scripts/ingest.py` — CLI ingestion with argparse, error handling, logging
- `scripts/ask.py` — CLI query with argparse, --mode naive|graph, --filter key=value, -k top_k
- `tests/` — 58 unit tests covering all modules (pytest)

---

## Phase Status

| Phase | Status | Details |
|-------|--------|---------|
| 1 — Baseline RAG | **COMPLETE** | All gaps resolved; 58 tests pass; end-to-end verified |
| 2 — Eval harness | SCAFFOLDED | Only 1 Q/A pair; needs 50+; RAGAS wiring exists but untested |
| 3 — Advanced retrieval | NOT STARTED | Hybrid search, parent-doc, query transforms, reranking |
| 4 — LangGraph CRAG | PARTIAL | Graph compiles and runs with corrective loop; missing critic node, SQLite checkpointer |
| 5 — Multi-agent + tools | NOT STARTED | web_search is a stub; no planner/synthesizer/tools |
| 6 — Observability | NOT STARTED | LangSmith config commented out in .env.example |
| 7 — Ship (API/UI/Docker) | NOT STARTED | No FastAPI, no UI, no Docker |

---

## Completed Tasks

### Phase 1 — Baseline RAG (COMPLETE)
- [x] Project skeleton and directory structure
- [x] config.py with env-based settings, logging setup, extended validation
- [x] Document loaders (PDF, TXT, MD) with full metadata (department, access_level from folder structure)
- [x] Chunker with RecursiveCharacterTextSplitter + logging
- [x] ChromaDB persistent vector store with singleton, content-hash dedup, collection_stats()
- [x] Naive RAG LCEL chain with citation prompt and "I don't know" enforcement
- [x] LangGraph CRAG graph (5 nodes, conditional edges, cached compilation, unique thread IDs)
- [x] Structured output grading with Pydantic GradeResult model
- [x] CLI scripts with argparse (ingest.py, ask.py) — --mode, --filter, -k options
- [x] Error handling and logging across all modules
- [x] Input validation for empty/whitespace queries
- [x] Expanded sample corpus: 9 documents across 6 departments (HR, Legal, Engineering, Finance, Security, Operations)
- [x] Duplicate ingestion prevention (content-hash based IDs)
- [x] 58 unit tests passing (pytest) covering config, loader, chunker, chroma_store, naive_rag, graph nodes, graph builder
- [x] Virtual environment (.venv) with all dependencies installed
- [x] requirements.txt with pytest, pytest-cov added
- [x] pytest.ini configuration
- [x] .gitignore comprehensive (secrets, venv, cache, IDE, test artifacts)
- [x] .env.example with LOG_LEVEL setting
- [x] PRD, TRD, and Implementation Plan documents
- [x] context.md for session continuity

## Pending Tasks

### Phase 2 (Eval harness — NEXT PRIORITY):
- [ ] Author 50+ Q/A pairs grounded in the corpus (cover easy/multi-part/out-of-corpus)
- [ ] Test RAGAS evaluation end-to-end
- [ ] Record and commit baseline scores
- [ ] Validate scores meet PRD baseline targets (faithfulness >= 0.65, etc.)

### Phase 3 (Advanced retrieval):
- [ ] Hybrid search (dense + BM25 + RRF)
- [ ] Parent-document / small-to-big retrieval
- [ ] Query transformation (HyDE, multi-query, step-back)
- [ ] Cross-encoder reranking
- [ ] Common retriever interface for A/B testing

### Phase 4 (LangGraph CRAG completion):
- [ ] Add critic node (claim verification against sources)
- [ ] Swap to SQLite checkpointer for production
- [ ] Add per-node tracing hooks
- [ ] Integration test full graph flow with mocked LLM

### Phase 5 (Multi-agent + tools):
- [ ] Wire web_search to Tavily
- [ ] Planner agent for multi-part question decomposition
- [ ] Synthesizer agent
- [ ] Calculator tool
- [ ] SQL-over-structured-data tool

### Phase 6 (Observability):
- [ ] Enable LangSmith tracing
- [ ] Per-query cost/latency tracking
- [ ] Eval datasets in LangSmith

### Phase 7 (Ship):
- [ ] FastAPI endpoints (/ingest, /ask, /health, /eval)
- [ ] Streaming response support
- [ ] Streamlit or Next.js UI
- [ ] Dockerfile + docker-compose
- [ ] Rate limiting and structured logging

---

## Architecture Decisions

1. **LangChain 1.0 + LangGraph 1.0** — Stable GA releases; LCEL for linear, StateGraph for cyclic flows
2. **ChromaDB local-first** — Single-node persistent store; swappable to server mode for scale
3. **OpenAI gpt-4o-mini** — Cost/quality balance; configurable via .env
4. **Structured output for grading** — Pydantic with_structured_output, not text parsing
5. **InMemorySaver for dev** — SQLite/Postgres for production checkpointing (not yet swapped)
6. **Metadata-based access control** — Enforced at retriever layer, not in prompts
7. **Department from folder structure** — Loader infers department from first subfolder under root
8. **Content-hash deduplication** — SHA-256 of (source + start_index + content) prevents duplicate chunks
9. **Singleton pattern for vectorstore/embeddings** — Avoids re-creating clients per call

## Known Issues

1. `langchain-community` deprecation warning (no standalone replacement for TextLoader/PyPDFLoader yet)
2. RAGAS eval has only 1 Q/A pair — needs 50+ before Phase 2 can be considered done
3. Web search node is a stub (expected — Phase 5)
4. No critic node yet (Phase 4)
5. InMemorySaver means graph state is lost on restart (swap to SQLite in Phase 4)

## Next Steps (Recommended Order)

1. **Complete Phase 2** — author 50+ eval Q/A pairs, run RAGAS, record baseline scores
2. **Phase 3** — advanced retrieval strategies (one at a time, measured against baseline)
3. **Phase 4** — critic node, SQLite checkpointer, integration tests
4. **Phase 5-7** — multi-agent, observability, API/UI/Docker

---

## Sample Corpus

9 documents across 6 departments, 41 chunks total:
| Department | Files | Access Level |
|---|---|---|
| HR | handbook.md, onboarding_guide.md | internal |
| Legal | data_protection_policy.md, vendor_contract_terms.md | confidential |
| Engineering | api_guidelines.md, incident_response_runbook.md | internal |
| Finance | procurement_policy.md, quarterly_report_q1_2026.md | internal |
| Security | information_security_policy.md | confidential |
| Operations | business_continuity_plan.md, change_management_process.txt | internal |

---

## End-to-End Test Results (2026-06-01)

### Ingestion
- 11 documents loaded from 6 department folders
- 41 chunks created and persisted to ChromaDB
- Re-ingestion correctly skips all 41 as duplicates (0 new added)

### Naive RAG Queries
| Query | Result | Citation | Pass |
|---|---|---|---|
| "What is the remote work policy?" | Correct answer with details | handbook.md | YES |
| "What is the password policy and minimum length?" | Correct (14 chars + details) | information_security_policy.md | YES |
| "What is the capital of Mars?" | "I don't have enough information..." | N/A | YES |
| "What was Q1 2026 total revenue?" (--filter dept=finance) | "$47.3 million" | quarterly_report_q1_2026.md | YES |

### CRAG Graph Queries
| Query | Result | Path Taken | Pass |
|---|---|---|---|
| "What are the vendor payment terms and SLA requirements?" | Correct multi-part answer | retrieve → grade (relevant) → generate | YES |
| "What is the recipe for chocolate cake?" | "I don't have enough information..." | retrieve → grade (not relevant) → rewrite → retrieve → grade → rewrite → retrieve → grade → web_search → generate | YES |

### Unit Tests
- **58 tests passed, 0 failed** (pytest, 2.72s)

---

## Session History

### Session 1 — 2026-06-01
- **Actions:** Full codebase review, PRD/TRD analysis, gap identification
- **Findings:** Phase 1 is a working skeleton but not production-complete; significant gaps in error handling, logging, testing, and corpus coverage
- **Output:** Created context.md with comprehensive project status
- **Next:** Address Phase 1 gaps, then move to Phase 2 (eval harness)

### Session 2 — 2026-06-01
- **Actions:** Complete Phase 1 hardening
- **What was done:**
  - Expanded sample corpus from 1 file to 9 documents across 6 departments (41 chunks)
  - Enhanced config.py with centralized logging setup, LOG_LEVEL env var, extended validation
  - Enhanced loader with department/access_level metadata inferred from folder structure
  - Enhanced chunker with logging and empty-input handling
  - Enhanced ChromaDB store with content-hash dedup, singleton pattern, collection_stats()
  - Enhanced naive RAG with improved "I don't know" prompt, input validation, error handling
  - Enhanced all graph nodes with error handling, logging, improved prompts
  - Enhanced build_graph with cached compilation (singleton) and unique thread IDs
  - Enhanced CLI scripts with argparse (--mode, --filter, -k), error handling, logging
  - Created .venv and installed all dependencies
  - Wrote 58 unit tests across 7 test files
  - Added pytest.ini, pytest + pytest-cov to requirements
  - Updated .gitignore, .env.example
- **Test Results:** 58/58 tests pass; all 6 end-to-end queries verified
- **Next:** Phase 2 — author 50+ eval Q/A pairs, run RAGAS, record baseline scores
