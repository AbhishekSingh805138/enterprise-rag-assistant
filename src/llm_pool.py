"""Thread-safe LLM instance pool.

Caches ChatOpenAI instances by (model, temperature) so the same
configuration is reused across nodes, avoiding redundant client
creation on every call.
"""
from __future__ import annotations

import logging
import threading

from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger(__name__)

_pool: dict[tuple[str, float], ChatOpenAI] = {}
_lock = threading.Lock()


def get_llm(temperature: float = 0, model: str | None = None) -> ChatOpenAI:
    """Return a cached ChatOpenAI instance for the given params.

    Thread-safe: concurrent calls with the same key return the same object.
    """
    mdl = model or settings.llm_model
    key = (mdl, temperature)
    with _lock:
        if key not in _pool:
            _pool[key] = ChatOpenAI(
                model=mdl,
                temperature=temperature,
                api_key=settings.openai_api_key,
                timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
            )
            logger.debug("LLM pool: created instance for %s", key)
        return _pool[key]


def reset_pool() -> None:
    """Clear the pool (for testing)."""
    with _lock:
        _pool.clear()
