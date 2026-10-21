"""Tests for src/backend/retrieval: fusion, index files, vector search."""

import os
import pickle

import faiss
import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.backend.retrieval import HybridRetriever, Retriever, load_local_retriever
from src.backend.retrieval.hybrid import reciprocal_rank_fusion
from src.backend.retrieval.index_files import (
    load_keyword_index,
    load_vector_index,
    save_keyword_index,
    save_vector_index,
)
from src.backend.retrieval.vector import VectorSearch


@pytest.fixture
def anyio_backend():
    return "asyncio"


def doc(text, topic=None):
    return Document(page_content=text, metadata={"source": "topical_guide", "topic": topic or text})


class FakeEmbeddings(Embeddings):
    """Embeds by keyword: each text becomes a unit vector over a tiny vocabulary."""

    VOCAB = ["faith", "baptism", "tithing", "prayer"]

    def _embed(self, text):
        v = np.array([text.lower().count(w) for w in self.VOCAB], dtype=float) + 1e-3
        return (v / np.linalg.norm(v)).tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


# ---------------------------------------------------------------------------
# Reciprocal rank fusion (must match LangChain's EnsembleRetriever)
# ---------------------------------------------------------------------------

def test_fusion_rewards_passages_found_by_both():
    a, b, c = doc("a"), doc("b"), doc("c")
    fused = reciprocal_rank_fusion([[a, b], [b, c]], [0.5, 0.5])
    assert [d.page_content for d in fused] == ["b", "a", "c"]


def test_fusion_dedupes_by_text_and_keeps_first_copy():
    first, duplicate = doc("same", "first"), doc("same", "second")
    fused = reciprocal_rank_fusion([[first], [duplicate]], [0.5, 0.5])
    assert len(fused) == 1 and fused[0].metadata["topic"] == "first"


def test_fusion_ties_keep_first_seen_order():
    a, b = doc("a"), doc("b")
    assert [d.page_content for d in reciprocal_rank_fusion([[a], [b]], [0.5, 0.5])] == ["a", "b"]


def test_fusion_handles_an_empty_list():
    a = doc("a")
    assert reciprocal_rank_fusion([[a], []], [0.5, 0.5]) == [a]


# ---------------------------------------------------------------------------
# Index files
# ---------------------------------------------------------------------------

def _build(tmp_path, texts):
    docs = [doc(t) for t in texts]
    embeddings = FakeEmbeddings()
    index = faiss.IndexFlatL2(len(FakeEmbeddings.VOCAB))
    index.add(np.asarray(embeddings.embed_documents(texts), dtype=np.float32))
    save_vector_index(tmp_path, index, docs)
    save_keyword_index(tmp_path / "bm25.pkl", docs)
    return docs, embeddings


def test_saved_indexes_load_back(tmp_path):
    docs, _ = _build(tmp_path, ["faith and prayer", "baptism by immersion", "tithing is a tenth"])
    index, vector_docs = load_vector_index(tmp_path)
    vectorizer, keyword_docs = load_keyword_index(tmp_path / "bm25.pkl")
    assert index.ntotal == 3
    assert [d.page_content for d in vector_docs] == [d.page_content for d in keyword_docs] == [d.page_content for d in docs]
    assert vectorizer.corpus_size == 3


def test_unknown_classes_in_an_index_file_are_refused(tmp_path):
    """A tampered pickle can't make the loader build arbitrary objects (or run code)."""
    class Sneaky:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    (tmp_path / "bm25.pkl").write_bytes(pickle.dumps(Sneaky()))
    with pytest.raises(pickle.UnpicklingError, match="isn't allowed"):
        load_keyword_index(tmp_path / "bm25.pkl")


def test_vector_and_passage_counts_must_match(tmp_path):
    _build(tmp_path, ["faith", "baptism"])
    with open(tmp_path / "index.pkl", "wb") as f:
        pickle.dump({"format": "gospel-rag/vector-docs/v1", "docs": [doc("faith")]}, f)
    with pytest.raises(ValueError, match="2 vectors but index.pkl has 1"):
        load_vector_index(tmp_path)


# ---------------------------------------------------------------------------
# Vector search and the assembled retriever
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_vector_search_returns_nearest_first(tmp_path):
    docs, embeddings = _build(tmp_path, ["faith and prayer", "baptism by immersion", "tithing is a tenth"])
    index, vector_docs = load_vector_index(tmp_path)
    search = VectorSearch(index, vector_docs, embeddings, k=2)
    hits = await search.search_with_distances("baptism")
    assert hits[0][0].page_content == "baptism by immersion"
    assert len(hits) == 2
    assert VectorSearch.relevance(hits[0][1]) > VectorSearch.relevance(hits[1][1])


@pytest.mark.anyio
async def test_local_retriever_fuses_both_searches(tmp_path):
    _, embeddings = _build(tmp_path, ["faith and prayer", "baptism by immersion", "tithing is a tenth"])
    retriever = load_local_retriever(tmp_path, embeddings, k=2)
    assert isinstance(retriever, HybridRetriever)
    assert isinstance(retriever, Retriever)
    assert retriever.chunk_counts() == {"vector": 3, "keyword": 3}
    hits = await retriever.ainvoke("tithing")
    assert hits[0].page_content == "tithing is a tenth"
