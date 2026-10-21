"""Hybrid search in Supabase Postgres: pgvector + full-text, fused in SQL.

All the searching happens in the hybrid_search() SQL function
(sql/004_rag_store.sql); this module embeds the question, builds the
full-text query (keyword.full_text_query) and turns the rows into
Documents. Nothing is loaded into the app's memory, unlike the FAISS/BM25
backend, and newly ingested collections are searchable immediately.

Two ways to reach the function, with identical results:
- SqlTransport: the backend's own Postgres connection (DATABASE_URL), used on Render
- RestTransport: Supabase's REST API with the service-role key, for scripts
  on machines without a direct database connection
"""

import asyncio
from typing import Protocol

import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import text

from src.backend.retrieval.keyword import full_text_query


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.8g}" for x in vector) + "]"


class Transport(Protocol):
    async def search(self, embedding: list[float], query_text: str, k: int) -> list[dict]: ...
    def count_chunks(self) -> int: ...


class SqlTransport:
    """Calls hybrid_search() over the backend's SQLAlchemy session factory."""

    _SEARCH = text(
        "SELECT chunk_id, content, collection, slug, title, url, score, semantic_rank, full_text_rank "
        "FROM hybrid_search(CAST(:embedding AS extensions.halfvec), :query_text, :k)"
    )

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def _search(self, embedding, query_text, k) -> list[dict]:
        with self.session_factory() as db:
            rows = db.execute(
                self._SEARCH, {"embedding": _vector_literal(embedding), "query_text": query_text, "k": k}
            ).mappings().all()
        return [dict(row) for row in rows]

    async def search(self, embedding, query_text, k) -> list[dict]:
        return await asyncio.to_thread(self._search, embedding, query_text, k)

    def count_chunks(self) -> int:
        with self.session_factory() as db:
            return db.execute(text("SELECT count(*) FROM rag_chunks")).scalar_one()


class RestTransport:
    """Calls hybrid_search() through Supabase's REST API (rpc) with the service-role key."""

    def __init__(self, supabase_url: str, service_key: str, client: httpx.AsyncClient | None = None):
        self.base = f"{supabase_url.rstrip('/')}/rest/v1"
        self.headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}"}
        # One client for the app's lifetime, so connections are reused.
        self.client = client or httpx.AsyncClient(headers=self.headers, timeout=30)

    async def search(self, embedding, query_text, k) -> list[dict]:
        response = await self.client.post(
            f"{self.base}/rpc/hybrid_search",
            json={"query_embedding": _vector_literal(embedding), "query_text": query_text, "match_count": k},
        )
        response.raise_for_status()
        return response.json()

    def count_chunks(self) -> int:
        response = httpx.get(
            f"{self.base}/rag_chunks",
            params={"select": "id"},
            headers=self.headers | {"Prefer": "count=exact", "Range": "0-0"},
            timeout=30,
        )
        response.raise_for_status()
        return int(response.headers["content-range"].rsplit("/", 1)[1])


class PgvectorRetriever:
    name = "pgvector"

    def __init__(self, transport: Transport, embeddings: Embeddings, k: int = 4, chunk_total: int | None = None):
        self.transport = transport
        self.embeddings = embeddings
        self.k = k
        self._chunk_total = transport.count_chunks() if chunk_total is None else chunk_total

    async def search_rows(self, query: str) -> list[dict]:
        embedding = await self.embeddings.aembed_query(query)
        return await self.transport.search(embedding, full_text_query(query), self.k)

    async def ainvoke(self, query: str) -> list[Document]:
        return [
            Document(
                page_content=row["content"],
                metadata={
                    "source": row["collection"],
                    "topic": row["slug"],
                    "title": row["title"],
                    "url": row["url"],
                },
            )
            for row in await self.search_rows(query)
        ]

    def chunk_counts(self) -> dict[str, int]:
        # One table serves both searches.
        return {"vector": self._chunk_total, "keyword": self._chunk_total}
