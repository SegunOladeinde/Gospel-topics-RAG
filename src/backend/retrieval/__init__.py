"""Passage retrieval for the answer pipeline.

main.py only depends on the Retriever interface (base.py). The current
implementation searches local index files (local_index.py): semantic
search over FAISS (vector.py) and keyword search over BM25 (keyword.py),
fused by reciprocal rank (hybrid.py). Moving to Supabase pgvector + Postgres
full-text search means adding one module that implements Retriever and
loading it in main.load_retriever(); nothing else needs to change.
pgvector.py is that module: hybrid search in Supabase Postgres, chosen with
RETRIEVER_BACKEND=pgvector.
"""

from src.backend.retrieval.base import Retriever
from src.backend.retrieval.hybrid import HybridRetriever
from src.backend.retrieval.local_index import load_local_retriever
from src.backend.retrieval.pgvector import PgvectorRetriever, RestTransport, SqlTransport

__all__ = ["HybridRetriever", "PgvectorRetriever", "RestTransport", "Retriever", "SqlTransport", "load_local_retriever"]
