"""The one interface the answer pipeline uses to find passages.

Everything in main.py talks to retrieval through Retriever. Today it's
HybridRetriever over local FAISS + BM25 files (local_index.py); a Supabase
pgvector + Postgres full-text implementation only has to provide the same
two methods to drop in.
"""

from typing import Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class Retriever(Protocol):
    async def ainvoke(self, query: str) -> list[Document]:
        """The most relevant passages for the query, best first.

        Each Document's metadata has "source" (the collection, e.g.
        "topical_guide") and "topic" (the page slug), which describe_source()
        and the source links are built from.
        """
        ...

    def chunk_counts(self) -> dict[str, int]:
        """How many passages each underlying index holds, for the startup log."""
        ...
