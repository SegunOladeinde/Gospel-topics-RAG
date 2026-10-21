"""Build the retriever from the FAISS + BM25 files in data/index/."""

from pathlib import Path

from langchain_core.embeddings import Embeddings

from src.backend.retrieval.hybrid import HybridRetriever
from src.backend.retrieval.index_files import load_keyword_index, load_vector_index
from src.backend.retrieval.keyword import KeywordSearch
from src.backend.retrieval.vector import VectorSearch


def load_local_retriever(index_dir: Path, embeddings: Embeddings, k: int = 4) -> HybridRetriever:
    """Semantic search over index.faiss plus keyword search over bm25.pkl.

    Callers should check bm25.pkl's integrity (BM25_INDEX_SHA256) first.
    """
    index, vector_docs = load_vector_index(index_dir)
    vectorizer, keyword_docs = load_keyword_index(index_dir / "bm25.pkl")
    return HybridRetriever(
        vector=VectorSearch(index, vector_docs, embeddings, k=k),
        keyword=KeywordSearch(vectorizer, keyword_docs, k=k),
    )
