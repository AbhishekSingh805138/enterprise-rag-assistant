"""Phase 10: Conversation Memory tests.

Tests for:
- ConversationStore CRUD operations
- Memory context builder (formatting, truncation)
- Graph memory nodes (load_memory, save_memory)
- Feature flag bypass when memory_enabled=False
"""
from __future__ import annotations

import pytest

from config import settings


def _set_setting(name: str, value):
    """Bypass frozen dataclass to set a setting for testing."""
    object.__setattr__(settings, name, value)


# ---------------------------------------------------------------------------
# ConversationStore tests
# ---------------------------------------------------------------------------


class TestConversationStore:
    """Unit tests for the SQLite-backed conversation store."""

    @pytest.fixture(autouse=True)
    def setup_store(self, tmp_path):
        from src.memory.conversation_store import ConversationStore
        self.db_path = str(tmp_path / "test_conv.db")
        self.store = ConversationStore(self.db_path)
        yield
        self.store.close()

    def test_add_and_get_history(self):
        self.store.add_message("s1", "user", "Hello")
        self.store.add_message("s1", "assistant", "Hi there!")
        history = self.store.get_history("s1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there!"

    def test_history_returns_chronological_order(self):
        self.store.add_message("s1", "user", "First")
        self.store.add_message("s1", "assistant", "Second")
        self.store.add_message("s1", "user", "Third")
        history = self.store.get_history("s1")
        contents = [m["content"] for m in history]
        assert contents == ["First", "Second", "Third"]

    def test_history_limit(self):
        for i in range(20):
            self.store.add_message("s1", "user", f"msg-{i}")
        history = self.store.get_history("s1", limit=5)
        assert len(history) == 5
        assert history[0]["content"] == "msg-15"
        assert history[-1]["content"] == "msg-19"

    def test_session_isolation(self):
        self.store.add_message("s1", "user", "Session 1")
        self.store.add_message("s2", "user", "Session 2")
        h1 = self.store.get_history("s1")
        h2 = self.store.get_history("s2")
        assert len(h1) == 1
        assert len(h2) == 1
        assert h1[0]["content"] == "Session 1"
        assert h2[0]["content"] == "Session 2"

    def test_clear_session(self):
        self.store.add_message("s1", "user", "msg1")
        self.store.add_message("s1", "assistant", "msg2")
        self.store.add_message("s2", "user", "other")
        deleted = self.store.clear_session("s1")
        assert deleted == 2
        assert len(self.store.get_history("s1")) == 0
        assert len(self.store.get_history("s2")) == 1

    def test_empty_session_returns_empty_list(self):
        assert self.store.get_history("nonexistent") == []

    def test_session_count(self):
        self.store.add_message("s1", "user", "a")
        self.store.add_message("s2", "user", "b")
        self.store.add_message("s1", "user", "c")
        assert self.store.session_count() == 2

    def test_created_at_is_set(self):
        self.store.add_message("s1", "user", "hello")
        history = self.store.get_history("s1")
        assert "created_at" in history[0]
        assert len(history[0]["created_at"]) > 0


# ---------------------------------------------------------------------------
# Memory context builder tests
# ---------------------------------------------------------------------------


class TestMemoryContextBuilder:
    """Unit tests for build_memory_context."""

    @pytest.fixture(autouse=True)
    def save_restore_settings(self):
        orig_enabled = settings.memory_enabled
        orig_tokens = settings.memory_max_tokens
        yield
        _set_setting("memory_enabled", orig_enabled)
        _set_setting("memory_max_tokens", orig_tokens)

    def test_empty_history_returns_empty_string(self):
        from src.memory.context_builder import build_memory_context
        assert build_memory_context([]) == ""

    def test_formats_history_correctly(self):
        _set_setting("memory_enabled", True)
        _set_setting("memory_max_tokens", 5000)
        from src.memory.context_builder import build_memory_context
        history = [
            {"role": "user", "content": "What is PTO?"},
            {"role": "assistant", "content": "PTO stands for Paid Time Off."},
        ]
        result = build_memory_context(history)
        assert "Previous conversation:" in result
        assert "User: What is PTO?" in result
        assert "Assistant: PTO stands for Paid Time Off." in result

    def test_truncates_to_token_budget(self):
        _set_setting("memory_enabled", True)
        from src.memory.context_builder import build_memory_context
        history = [
            {"role": "user", "content": "x" * 1000},
            {"role": "assistant", "content": "y" * 1000},
            {"role": "user", "content": "z" * 1000},
        ]
        result = build_memory_context(history, max_tokens=100)
        assert len(result) < 2000

    def test_disabled_returns_empty(self):
        _set_setting("memory_enabled", False)
        from src.memory.context_builder import build_memory_context
        history = [{"role": "user", "content": "hello"}]
        assert build_memory_context(history) == ""

    def test_preserves_chronological_order(self):
        _set_setting("memory_enabled", True)
        _set_setting("memory_max_tokens", 5000)
        from src.memory.context_builder import build_memory_context
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        result = build_memory_context(history)
        first_pos = result.index("first")
        second_pos = result.index("second")
        third_pos = result.index("third")
        assert first_pos < second_pos < third_pos


# ---------------------------------------------------------------------------
# Graph memory node tests
# ---------------------------------------------------------------------------


class TestMemoryNodes:
    """Unit tests for load_memory and save_memory graph nodes."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self._orig_enabled = settings.memory_enabled
        self._orig_turns = settings.memory_max_turns
        self._orig_tokens = settings.memory_max_tokens
        _set_setting("memory_enabled", True)
        _set_setting("memory_max_turns", 10)
        _set_setting("memory_max_tokens", 2000)
        self.db_path = str(tmp_path / "test_memory_nodes.db")
        import src.memory.conversation_store as cs_mod
        cs_mod._store = None
        self._cs_mod = cs_mod
        self._orig_default = cs_mod._default_db_path
        cs_mod._default_db_path = lambda: self.db_path
        yield
        cs_mod._store = None
        cs_mod._default_db_path = self._orig_default
        _set_setting("memory_enabled", self._orig_enabled)
        _set_setting("memory_max_turns", self._orig_turns)
        _set_setting("memory_max_tokens", self._orig_tokens)

    def test_load_memory_empty_session(self):
        from src.graph.memory_nodes import load_memory
        result = load_memory({"session_id": "new-session"})
        assert result["chat_history"] == []
        assert result["memory_context"] == ""

    def test_load_memory_with_history(self):
        from src.graph.memory_nodes import load_memory
        from src.memory.conversation_store import get_conversation_store
        store = get_conversation_store(self.db_path)
        store.add_message("s1", "user", "What is PTO?")
        store.add_message("s1", "assistant", "PTO is Paid Time Off.")

        result = load_memory({"session_id": "s1"})
        assert len(result["chat_history"]) == 2
        assert "PTO" in result["memory_context"]

    def test_load_memory_no_session_id(self):
        from src.graph.memory_nodes import load_memory
        result = load_memory({})
        assert result["chat_history"] == []
        assert result["memory_context"] == ""

    def test_save_memory_persists_qa(self):
        from src.graph.memory_nodes import save_memory
        from src.memory.conversation_store import get_conversation_store
        save_memory({
            "session_id": "s1",
            "question": "What is PTO?",
            "generation": "PTO is Paid Time Off.",
        })
        store = get_conversation_store(self.db_path)
        history = store.get_history("s1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_save_memory_uses_original_question(self):
        from src.graph.memory_nodes import save_memory
        from src.memory.conversation_store import get_conversation_store
        save_memory({
            "session_id": "s1",
            "original_question": "original Q",
            "question": "rewritten Q",
            "generation": "answer",
        })
        store = get_conversation_store(self.db_path)
        history = store.get_history("s1")
        assert history[0]["content"] == "original Q"

    def test_save_memory_no_session_is_noop(self):
        from src.graph.memory_nodes import save_memory
        result = save_memory({"question": "hello", "generation": "hi"})
        assert result == {}

    def test_disabled_memory_skips_load(self):
        _set_setting("memory_enabled", False)
        from src.graph.memory_nodes import load_memory
        result = load_memory({"session_id": "s1"})
        assert result["chat_history"] == []

    def test_disabled_memory_skips_save(self):
        _set_setting("memory_enabled", False)
        from src.graph.memory_nodes import save_memory
        from src.memory.conversation_store import get_conversation_store
        save_memory({
            "session_id": "s1",
            "question": "hello",
            "generation": "hi",
        })
        store = get_conversation_store(self.db_path)
        assert store.get_history("s1") == []


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestConversationStoreSingleton:
    """Test the module-level singleton pattern."""

    def test_get_and_reset(self, tmp_path):
        import src.memory.conversation_store as cs_mod
        cs_mod._store = None
        db = str(tmp_path / "singleton_test.db")
        store1 = cs_mod.get_conversation_store(db)
        store2 = cs_mod.get_conversation_store(db)
        assert store1 is store2
        cs_mod.reset_conversation_store()
        store3 = cs_mod.get_conversation_store(db)
        assert store3 is not store1
        cs_mod.reset_conversation_store()
