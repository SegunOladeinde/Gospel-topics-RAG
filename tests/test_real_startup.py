"""Run the app's real startup against the real data/index files. No mocks.

Every other test replaces load_retriever with a stub (see conftest.py), which
is how a startup crash in index_chunk_counts reached production unnoticed:
nothing ever loaded the real indexes. These tests do. The index files are
committed, so this runs in CI too. Loading takes a few seconds and makes no
network calls (no embeddings are computed).
"""

import hashlib
import logging

import pytest
from fastapi.testclient import TestClient
from langchain.retrievers import EnsembleRetriever

import src.backend.main as backend
from src.backend.keyword_retriever import KeywordBM25Retriever
from src.backend.main import BM25_INDEX_PATH, INDEX_DIR, app, index_chunk_counts


def test_index_files_are_present():
    """If this fails, the tests below can't prove startup works."""
    for name in ("index.faiss", "index.pkl", "bm25.pkl"):
        assert (INDEX_DIR / name).is_file(), f"data/index/{name} is missing"


def test_real_startup_loads_both_indexes(monkeypatch, real_startup):
    monkeypatch.setenv("APP_API_KEY", "test-key")
    # Exercise the integrity check too, with the real file's digest.
    monkeypatch.setenv("BM25_INDEX_SHA256", hashlib.sha256(BM25_INDEX_PATH.read_bytes()).hexdigest())

    with TestClient(app) as client:
        health = client.get("/health")
        retriever = app.state.retriever

        assert health.status_code == 200
        assert health.json()["indexes_loaded"] is True
        assert isinstance(retriever, EnsembleRetriever)
        faiss_retriever, keyword = retriever.retrievers
        assert isinstance(keyword, KeywordBM25Retriever)

        counts = index_chunk_counts(retriever)
        assert counts["faiss"] > 0
        assert counts["faiss"] == counts["bm25"] == keyword.doc_count

        # Keyword search needs no embeddings call, so it can run here.
        hits = keyword.invoke("What is baptism?")
        assert hits and all("baptism" in doc.page_content.lower() for doc in hits)


def test_startup_survives_a_failing_index_count(monkeypatch, real_startup, caplog):
    """Counting chunks is only for the log; an error there must not stop the app."""
    monkeypatch.setenv("APP_API_KEY", "test-key")

    def broken(_retriever):
        raise AttributeError("simulated: wrapper has no such attribute")

    monkeypatch.setattr(backend, "index_chunk_counts", broken)
    with caplog.at_level(logging.WARNING, logger="src.backend.main"):
        with TestClient(app) as client:
            assert client.get("/health").json()["indexes_loaded"] is True
    assert "Could not count the loaded index chunks" in caplog.text


@pytest.mark.parametrize("sha", ["0" * 64])
def test_real_startup_refuses_a_tampered_index(monkeypatch, real_startup, sha):
    """The integrity check still guards the real file."""
    monkeypatch.setenv("APP_API_KEY", "test-key")
    monkeypatch.setenv("BM25_INDEX_SHA256", sha)
    with pytest.raises(Exception, match="integrity"):
        with TestClient(app):
            pass
