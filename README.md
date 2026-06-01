# Enterprise Research & Compliance Assistant

A multi-agent RAG system over a company document corpus, built to learn modern
AI engineering in depth: **advanced RAG, LangChain 1.0, LangGraph 1.0, ChromaDB**,
and evaluation-driven development.

This repo is the **Phase 1 scaffold + LangGraph skeleton**. It runs end-to-end
today (naive RAG baseline + a Corrective RAG graph) and is structured so you
layer in the advanced pieces one phase at a time, measuring each against a
RAGAS baseline.

## Why this architecture

- **LangChain** does the plumbing: loaders, splitters, model/embedding interfaces (LCEL).
- **ChromaDB** is the persistent vector store with metadata-filtered retrieval.
- **LangGraph** orchestrates a *stateful, cyclic* agent graph — retrieve → grade →
  (rewrite & retry | web fallback) → generate. The conditional loop is the thing
  a linear chain can't do, and the reason to learn LangGraph.
- **RAGAS** turns "feels better" into faithfulness / context-precision numbers.

## Setup

Requires **Python 3.11+** and an OpenAI API key.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then paste your OPENAI_API_KEY
```

## Run it

```bash
# 1. Ingest the sample doc (or point at your own folder / PDF)
python -m scripts.ingest ./data/sample_docs

# 2. Ask with the naive baseline (Phase 1)
python -m scripts.ask "What is the remote work policy?"

# 3. Ask with the Corrective RAG graph (LangGraph)
python -m scripts.ask --graph "What is the remote work policy?"

# 4. Measure the baseline (after adding real Q/A pairs to src/eval/ragas_eval.py)
python -m src.eval.ragas_eval
```

## Project layout

```
config.py                    # env-driven settings
scripts/
  ingest.py                  # load -> chunk -> embed -> persist
  ask.py                     # query (naive or --graph)
src/
  ingestion/loader.py        # pdf / txt / md loaders + metadata
  ingestion/chunker.py       # RecursiveCharacterTextSplitter (swap in Phase 3)
  vectorstore/chroma_store.py# persistent Chroma + metadata-filtered retriever
  rag/naive_rag.py           # Phase 1 LCEL baseline
  graph/state.py             # LangGraph shared state (TypedDict)
  graph/nodes.py             # retrieve / grade / rewrite / web / generate nodes
  graph/build_graph.py       # wires the CRAG cycle, compiles with a checkpointer
  eval/ragas_eval.py         # Phase 2 evaluation harness
```

## Roadmap (build in this order)

1. **[done] Naive RAG baseline** + a metric to beat.
2. **[done — fill in Q/A] Eval harness** (RAGAS). Record baseline scores *now*.
3. **Advanced retrieval**, one at a time, measuring each:
   hybrid (dense + BM25 + RRF), parent-document / small-to-big,
   query transformation (HyDE, multi-query, step-back), cross-encoder reranking.
4. **LangGraph CRAG** (skeleton present): grade → correct → loop. Add a critic
   node that verifies claims against sources.
5. **Multi-agent + tools**: planner/synthesizer agents, wire `web_search` to
   Tavily, add SQL-over-structured-data and a calculator tool.
6. **Observability**: turn on LangSmith tracing (uncomment in `.env`), track
   latency/cost, store eval datasets.
7. **Ship it**: FastAPI backend with streaming, a Streamlit/Next.js UI, Docker.

### Stretch
Evaluation-driven CI (evals on every commit), access-control-aware retrieval
(metadata filters by `department`/`access_level`), a fine-tuned reranker.

## The portfolio artifact

Don't just ship a chatbot. Produce the **benchmark table**: faithfulness and
context precision from naive (~0.6) to corrective multi-agent (~0.9), with the
cost/latency tradeoff for each step. That comparison is what demonstrates senior
AI-engineering judgment.

## Notes

- Built against LangChain 1.0 / LangGraph 1.0 (stable, GA Oct 2025). APIs in
  these examples use `langgraph.graph.StateGraph` and LCEL.
- `gpt-4o-mini` + `text-embedding-3-small` are the cheap defaults — change in `.env`.
- `chroma_db/` and `.env` are gitignored. Never commit your key.
