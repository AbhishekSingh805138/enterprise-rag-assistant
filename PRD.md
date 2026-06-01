# Product Requirements Document (PRD)
### Enterprise Research & Compliance Assistant

| | |
|---|---|
| **Document owner** | Lead AI Architect |
| **Status** | Draft v1.0 |
| **Last updated** | 2026-05-30 |
| **Audience** | Engineering, eval/QA, stakeholders |
| **Related docs** | `TRD.md`, `IMPLEMENTATION_PLAN.md` |

---

## 1. Summary

An internal AI assistant that answers complex questions over a company's
private document corpus (policies, contracts, technical docs, handbooks) with
**grounded, source-cited answers**. Unlike a generic chatbot, it uses a
multi-agent retrieval pipeline that retrieves, grades, self-corrects, and
verifies claims against sources before answering — the pattern enterprise
products like Glean, Hebbia, and Harvey are built on.

The product doubles as a reference implementation of modern AI engineering:
advanced RAG, agentic orchestration, and evaluation-driven development.

## 2. Problem statement

Knowledge workers waste time hunting through scattered documents, and when they
do find an answer they can't trust it without verifying the source. Generic
LLM chat tools hallucinate, can't see private documents, and give no provenance.
Naive RAG reduces hallucination but still fails on multi-part questions, returns
irrelevant context, and offers no way to know when it's wrong.

**The core need:** trustworthy, traceable answers over private knowledge, with
a measurable confidence story — not vibes.

## 3. Goals and non-goals

### Goals
- G1 — Answer natural-language questions over a private corpus with **inline source citations**.
- G2 — Reduce hallucination via a self-correcting retrieval loop (grade → retry → verify).
- G3 — Support metadata-filtered retrieval (by department, doc type, access level).
- G4 — Provide a **measurable quality bar** (RAGAS) that gates every change.
- G5 — Expose the assistant via API and a simple UI with streaming responses.

### Non-goals (v1)
- N1 — Document *editing* or generation of new policy documents.
- N2 — Real-time ingestion / live sync with source systems (batch ingest only in v1).
- N3 — Fine-tuning a base LLM (reranker fine-tuning is a stretch goal only).
- N4 — Multi-language corpus (English-only in v1).
- N5 — Mobile-native apps.

## 4. Target users & personas

| Persona | Need | Key behaviour |
|---|---|---|
| **Operations / HR staff** | Quick, authoritative answers on policy | Asks short factual questions; needs the source to act |
| **Legal / compliance analyst** | Cross-reference clauses across contracts | Asks multi-part questions; demands verifiable provenance |
| **Engineer / IC** | Find technical-doc answers fast | Tolerates latency for depth; values accuracy over speed |
| **Knowledge admin** | Curate corpus & control access | Manages ingestion, access levels, monitors quality |

## 5. User stories

- US1 — *As an HR staffer*, I ask "What is the remote work policy?" and get a one-paragraph answer with the source document cited, so I can quote it confidently.
- US2 — *As a compliance analyst*, I ask a question spanning multiple contracts and get an answer that synthesizes across them with each claim attributed.
- US3 — *As any user*, when the corpus doesn't contain the answer, the assistant tells me it doesn't know rather than inventing one.
- US4 — *As a knowledge admin*, I ingest a folder of PDFs and tag them with department and access level so retrieval respects permissions.
- US5 — *As an engineer*, I see which sources were retrieved and why an answer was produced (trace), so I can debug poor answers.

## 6. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR1 | Ingest PDF, TXT, and Markdown; chunk; embed; persist to vector store | Must |
| FR2 | Attach metadata (source, filename, doc_type, department, access_level) per chunk | Must |
| FR3 | Answer queries with top-k retrieval + LLM generation, citing source filenames | Must |
| FR4 | Grade retrieved context for relevance before generating | Must |
| FR5 | On weak context, rewrite the query and re-retrieve (corrective loop, bounded retries) | Must |
| FR6 | Refuse / say "I don't know" when context is insufficient | Must |
| FR7 | Metadata-filtered retrieval (e.g. restrict to a department or access level) | Should |
| FR8 | Web-search fallback when the corpus is insufficient | Should |
| FR9 | Multi-agent decomposition for multi-part questions (planner → retrieve → synthesize → critic) | Should |
| FR10 | Streaming responses via API and UI | Should |
| FR11 | Per-query trace (retrieved docs, grader verdict, path taken) | Should |
| FR12 | Tool use: calculator, SQL over structured data | Could |

## 7. Experience requirements

- Answers must lead with the direct response, followed by cited sources.
- Latency expectation set per mode: fast (naive) vs. thorough (agentic).
- Citations must be clickable/identifiable back to the source document.
- "I don't know" is an acceptable and expected outcome — never a fabricated answer.

## 8. Success metrics (KPIs)

### Quality (gating — measured with RAGAS on a held-out eval set)
| Metric | Baseline (naive) target | v1 (agentic) target |
|---|---|---|
| Faithfulness | ≥ 0.65 | **≥ 0.90** |
| Answer relevancy | ≥ 0.70 | **≥ 0.85** |
| Context precision | ≥ 0.60 | **≥ 0.80** |
| Context recall | ≥ 0.70 | **≥ 0.85** |

### Performance & cost
- p95 latency: naive mode < 3s; agentic mode < 8s.
- Cost per query: < $0.02 at default models (`gpt-4o-mini` + `text-embedding-3-small`).

### Product
- Hallucination rate (manual audit, 50 queries): < 5%.
- "Don't know" correctly triggered on out-of-corpus questions: > 90%.

## 9. Assumptions, dependencies, constraints

- **Assumption:** corpus fits a single ChromaDB instance for v1 (≤ ~1M chunks).
- **Dependency:** OpenAI API access (key provided); optional LangSmith, Tavily.
- **Constraint:** English-only; batch ingestion; single-tenant deployment in v1.
- **Constraint:** cost-conscious defaults; premium models opt-in via config.

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Eval set too small/biased | Quality numbers misleading | Build 50+ curated Q/A pairs; review coverage |
| Retrieval misses relevant docs (recall) | Wrong/incomplete answers | Hybrid search + reranking; measure context recall |
| Corrective loop increases latency/cost | Poor UX, budget overrun | Bound retries; cache; fast/thorough mode split |
| Access-control leakage via retrieval | Compliance breach | Metadata pre-filtering enforced at retriever layer |
| LLM provider/model drift | Behaviour changes | Pin models in config; eval gate catches regressions |

## 11. Out of scope / future

Multi-tenancy with per-tenant isolation, access-control-aware retrieval at
scale, fine-tuned reranker, evaluation-driven CI, live document sync, and
multi-language support are explicitly deferred beyond v1.

## 12. High-level milestones

See `IMPLEMENTATION_PLAN.md` for detail. At a glance: Baseline → Eval harness →
Advanced retrieval → Agentic (LangGraph CRAG) → Multi-agent + tools →
Observability → Ship (API/UI/Docker).
