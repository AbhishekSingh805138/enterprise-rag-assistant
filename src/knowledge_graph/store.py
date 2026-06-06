"""Knowledge graph store backed by NetworkX.

Thread-safe singleton that persists to JSON. Supports adding triples,
querying neighbors, and searching entities by name.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import networkx as nx

from config import settings
from src.knowledge_graph.models import Triple

logger = logging.getLogger(__name__)


class KnowledgeGraphStore:
    """In-memory knowledge graph with JSON persistence."""

    def __init__(self, persist_path: str | None = None) -> None:
        self._graph = nx.DiGraph()
        self._lock = threading.Lock()
        self._persist_path = persist_path or settings.kg_persist_path
        self._load()

    def _load(self) -> None:
        """Load graph from JSON if it exists."""
        path = Path(self._persist_path)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self._graph = nx.node_link_graph(data, directed=True)
                logger.info("Loaded knowledge graph: %d nodes, %d edges",
                            self._graph.number_of_nodes(), self._graph.number_of_edges())
            except Exception:
                logger.warning("Failed to load knowledge graph from %s", path, exc_info=True)
                self._graph = nx.DiGraph()

    def _persist(self) -> None:
        """Save graph to JSON."""
        path = Path(self._persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = nx.node_link_data(self._graph)
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            logger.warning("Failed to persist knowledge graph", exc_info=True)

    def add_triples(self, triples: list[Triple], source_doc: str = "") -> int:
        """Add triples to the graph. Returns number added."""
        if not triples:
            return 0

        with self._lock:
            added = 0
            for t in triples:
                if not t.subject.strip() or not t.object.strip():
                    continue

                # Add nodes with metadata
                self._graph.add_node(t.subject, type="entity", source=source_doc)
                self._graph.add_node(t.object, type="entity", source=source_doc)

                # Add edge
                self._graph.add_edge(
                    t.subject, t.object,
                    predicate=t.predicate,
                    source=source_doc,
                )
                added += 1

            if added > 0:
                self._persist()

        logger.debug("Added %d triples (total: %d nodes, %d edges)",
                      added, self._graph.number_of_nodes(), self._graph.number_of_edges())
        return added

    def query_neighbors(self, entity: str, depth: int = 1) -> list[dict]:
        """Get neighboring entities and relationships up to a given depth.

        Returns a list of dicts: [{"subject", "predicate", "object", "source"}, ...]
        """
        max_depth = min(depth, settings.kg_max_depth)
        results: list[dict] = []

        with self._lock:
            if entity not in self._graph:
                # Try case-insensitive match
                matches = [n for n in self._graph.nodes if n.lower() == entity.lower()]
                if not matches:
                    return []
                entity = matches[0]

            # BFS up to max_depth
            visited: set[str] = set()
            queue: list[tuple[str, int]] = [(entity, 0)]

            while queue:
                node, d = queue.pop(0)
                if node in visited or d > max_depth:
                    continue
                visited.add(node)

                # Outgoing edges
                for _, target, data in self._graph.edges(node, data=True):
                    results.append({
                        "subject": node,
                        "predicate": data.get("predicate", "related_to"),
                        "object": target,
                        "source": data.get("source", ""),
                    })
                    if d < max_depth:
                        queue.append((target, d + 1))

                # Incoming edges
                for source, _, data in self._graph.in_edges(node, data=True):
                    results.append({
                        "subject": source,
                        "predicate": data.get("predicate", "related_to"),
                        "object": node,
                        "source": data.get("source", ""),
                    })
                    if d < max_depth:
                        queue.append((source, d + 1))

        return results

    def search_entities(self, query: str) -> list[str]:
        """Search for entities whose names contain the query string (case-insensitive)."""
        query_lower = query.lower()
        with self._lock:
            return [n for n in self._graph.nodes if query_lower in n.lower()]

    def stats(self) -> dict:
        """Return graph statistics."""
        with self._lock:
            return {
                "nodes": self._graph.number_of_nodes(),
                "edges": self._graph.number_of_edges(),
            }

    def clear(self) -> None:
        """Clear all nodes and edges."""
        with self._lock:
            self._graph.clear()
            self._persist()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: KnowledgeGraphStore | None = None
_store_lock = threading.Lock()


def get_kg_store(persist_path: str | None = None) -> KnowledgeGraphStore:
    """Return the singleton KnowledgeGraphStore. Thread-safe."""
    global _store
    with _store_lock:
        if _store is None:
            _store = KnowledgeGraphStore(persist_path)
            logger.info("KnowledgeGraphStore initialized")
        return _store


def reset_kg_store() -> None:
    """Discard the singleton (for testing)."""
    global _store
    with _store_lock:
        _store = None
