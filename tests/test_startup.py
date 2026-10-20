"""Startup logging and BM25 integrity-check messages."""

import hashlib
import logging

import pytest

from src.backend.main import _verify_bm25_hash, log_startup_config


def test_startup_log_lists_env_var_names_but_never_values(monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret-value")
    monkeypatch.setenv("APP_API_KEY", "app-very-secret-value")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:db-very-secret@host/db")
    monkeypatch.delenv("BM25_INDEX_SHA256", raising=False)
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abcdef1234567890")

    with caplog.at_level(logging.INFO, logger="src.backend.main"):
        log_startup_config()

    assert "Starting commit abcdef1" in caplog.text
    assert "OPENAI_API_KEY" in caplog.text and "DATABASE_URL" in caplog.text
    assert "Optional env vars not set:" in caplog.text and "BM25_INDEX_SHA256" in caplog.text
    assert "very-secret" not in caplog.text


def test_startup_log_warns_when_app_api_key_missing(monkeypatch, caplog):
    monkeypatch.delenv("APP_API_KEY", raising=False)
    with caplog.at_level(logging.INFO, logger="src.backend.main"):
        log_startup_config()
    assert "Required env var APP_API_KEY is not set." in caplog.text


def test_bm25_warning_mentions_environment_variable_not_dotenv(monkeypatch, caplog, tmp_path):
    monkeypatch.delenv("BM25_INDEX_SHA256", raising=False)
    with caplog.at_level(logging.WARNING, logger="src.backend.main"):
        _verify_bm25_hash(tmp_path / "bm25.pkl")
    assert "BM25_INDEX_SHA256 environment variable is not set" in caplog.text
    assert ".env" not in caplog.text


def test_bm25_hash_match_passes(monkeypatch, tmp_path):
    index = tmp_path / "bm25.pkl"
    index.write_bytes(b"index bytes")
    monkeypatch.setenv("BM25_INDEX_SHA256", hashlib.sha256(b"index bytes").hexdigest().upper())
    _verify_bm25_hash(index)  # case-insensitive match, no exception


def test_bm25_hash_mismatch_shows_prefixes_and_how_to_fix(monkeypatch, tmp_path):
    index = tmp_path / "bm25.pkl"
    index.write_bytes(b"rebuilt index")
    expected = "a" * 64
    actual = hashlib.sha256(b"rebuilt index").hexdigest()
    monkeypatch.setenv("BM25_INDEX_SHA256", expected)

    with pytest.raises(RuntimeError) as excinfo:
        _verify_bm25_hash(index)

    message = str(excinfo.value)
    assert f"Expected (BM25_INDEX_SHA256): {expected[:12]}..." in message
    assert f"Actual (file on disk):        {actual[:12]}..." in message
    assert actual not in message  # prefix only
    assert "Render dashboard" in message and "hashlib.sha256" in message
