"""Passage retrieval for the answer pipeline.

main.py only depends on the Retriever interface (base.py). The current
implementation searches local index files (local_index.py): semantic
search over FAISS (vector.py) and keyword search over BM25 (keyword.py),
fused by reciprocal rank (hybrid.py). Moving to Supabase pgvector + Postgres
full-text search means adding one module that implements Retriever and
loading it in main.load_retriever(); nothing else needs to change.
"""

from src.backend.retrieval.base import Retriever
from src.backend.retrieval.hybrid import HybridRetriever
from src.backend.retrieval.local_index import load_local_retriever

__all__ = ["HybridRetriever", "Retriever", "load_local_retriever"]
