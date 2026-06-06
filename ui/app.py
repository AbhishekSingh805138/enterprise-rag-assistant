"""Streamlit chat UI for the Enterprise RAG Assistant.

Connects to the FastAPI backend for queries. Supports streaming responses
with citation highlighting.

Usage:
    streamlit run ui/app.py
    # or with custom API URL:
    API_URL=http://localhost:8000 streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Settings")

    mode = st.selectbox("Pipeline Mode", ["naive", "graph", "auto"], index=1)
    retriever = st.selectbox(
        "Retriever Strategy",
        ["dense", "hybrid", "multi_query", "rerank", "cross_rerank",
         "hybrid_cross_rerank", "knowledge_graph"],
        index=1,
    )

    st.divider()

    # Health check
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.ok:
            health = resp.json()
            st.success(f"API: {health['status']}")
            st.caption(f"Collection: {health['collection']} ({health['document_count']} chunks)")
            st.caption(f"Version: {health['version']}")
        else:
            st.error("API: unhealthy")
    except requests.ConnectionError:
        st.error(f"Cannot connect to API at {API_URL}")
    except Exception as e:
        st.warning(f"Health check failed: {e}")

    st.divider()

    # Document upload
    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload a document to the knowledge base",
        type=["pdf", "txt", "md"],
        help="Supported formats: PDF, TXT, MD",
    )
    department = st.selectbox(
        "Department",
        ["general", "hr", "legal", "engineering", "finance", "security", "operations"],
        help="Assign a department for metadata filtering",
    )

    if uploaded_file is not None and st.button("Ingest Document"):
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            try:
                resp = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")},
                    data={"department": department},
                    timeout=60,
                )
                if resp.ok:
                    result = resp.json()
                    st.success(
                        f"Ingested **{result['filename']}**: "
                        f"{result['chunks_added']} new chunks added "
                        f"({result['collection_total']} total in collection)"
                    )
                else:
                    st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
            except requests.ConnectionError:
                st.error(f"Cannot connect to API at {API_URL}")
            except Exception as e:
                st.error(f"Upload error: {e}")

    st.divider()

    # Session management
    st.subheader("Conversation")
    st.caption(f"Session: {st.session_state.session_id}")
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.rerun()

    st.divider()
    st.caption("Enterprise RAG Assistant v1.0")
    st.caption("Powered by LangChain + LangGraph + ChromaDB")


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

st.title("Enterprise RAG Assistant")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            parts = [
                f"Mode: {meta.get('mode', '?')}",
                f"Retriever: {meta.get('retriever', '?')}",
                f"Tokens: {meta.get('tokens', '?')}",
                f"Cost: ${meta.get('cost', 0):.5f}",
                f"Latency: {meta.get('latency', 0):.0f}ms",
            ]
            if meta.get("intent"):
                parts.append(f"Intent: {meta['intent']}")
            if meta.get("cache_hit"):
                parts.append("Cache: HIT")
            st.caption(" | ".join(parts))


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about enterprise documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream response from API
    with st.chat_message("assistant"):
        placeholder = st.empty()
        meta_placeholder = st.empty()

        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": prompt,
                    "mode": mode,
                    "retriever_strategy": retriever,
                    "stream": True,
                    "session_id": st.session_state.session_id,
                },
                stream=True,
                timeout=60,
            )
            resp.raise_for_status()

            full_text = ""
            meta = {}

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])

                if data.get("type") == "token":
                    content = data.get("content", "")
                    if mode == "graph":
                        # Graph mode sends full generation at once
                        full_text = content
                    else:
                        # Naive mode streams token by token
                        full_text += content
                    placeholder.markdown(full_text + "▌")

                elif data.get("type") == "status":
                    node = data.get("node", "")
                    meta_placeholder.caption(f"Processing: {node}...")

                elif data.get("type") == "done":
                    full_text = data.get("answer", full_text)
                    meta = {
                        "mode": mode,
                        "retriever": retriever,
                        "cost": data.get("cost_usd", 0),
                        "latency": data.get("latency_ms", 0),
                        "tokens": data.get("tokens_used", 0),
                        "intent": data.get("intent"),
                        "cache_hit": data.get("cache_hit", False),
                    }

            # Final display
            placeholder.markdown(full_text)
            if meta:
                parts = [
                    f"Mode: {meta.get('mode', '?')}",
                    f"Retriever: {meta.get('retriever', '?')}",
                    f"Tokens: {meta.get('tokens', '?')}",
                    f"Cost: ${meta.get('cost', 0):.5f}",
                    f"Latency: {meta.get('latency', 0):.0f}ms",
                ]
                if meta.get("intent"):
                    parts.append(f"Intent: {meta['intent']}")
                if meta.get("cache_hit"):
                    parts.append("Cache: HIT")
                meta_placeholder.caption(" | ".join(parts))

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text,
                "meta": meta,
            })

        except requests.ConnectionError:
            placeholder.error(
                f"Cannot connect to API at {API_URL}. "
                "Start the API server with: `uvicorn api.app:app`"
            )
        except requests.HTTPError as e:
            placeholder.error(f"API error: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            placeholder.error(f"Error: {e}")
