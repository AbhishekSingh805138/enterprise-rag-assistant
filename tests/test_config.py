"""Tests for configuration and validation."""
from __future__ import annotations

import pytest

from config import Settings


class TestSettingsValidation:
    def test_missing_api_key_raises(self):
        s = Settings(openai_api_key="")
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            s.validate()

    def test_valid_settings_pass(self):
        s = Settings(openai_api_key="sk-test-key")
        s.validate()  # should not raise

    def test_zero_chunk_size_raises(self):
        s = Settings(openai_api_key="sk-test", chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size"):
            s.validate()

    def test_negative_chunk_overlap_raises(self):
        s = Settings(openai_api_key="sk-test", chunk_overlap=-1)
        with pytest.raises(ValueError, match="chunk_overlap"):
            s.validate()

    def test_overlap_gte_size_raises(self):
        s = Settings(openai_api_key="sk-test", chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="chunk_overlap"):
            s.validate()

    def test_zero_top_k_raises(self):
        s = Settings(openai_api_key="sk-test", top_k=0)
        with pytest.raises(ValueError, match="top_k"):
            s.validate()

    def test_defaults_are_sensible(self):
        s = Settings(openai_api_key="sk-test")
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 200
        assert s.top_k == 4
        assert s.llm_model == "gpt-4o-mini"
