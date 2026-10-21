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

from rank_bm25 import BM25Okapi

from src.backend.retrieval.hybrid import reciprocal_rank_fusion
from src.backend.retrieval.keyword import KeywordSearch, full_text_query


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.8g}" for x in vector) + "]"


class Transport(Protocol):
    async def search(self, embedding: list[float], query_text: str, k: int) -> list[dict]: ...
    async def match(self, embedding: list[float], k: int, candidate_ids: list[int]) -> list[dict]: ...
    def count_chunks(self) -> int: ...
    def load_passages(self) -> list[dict]: ...
    def signature(self) -> tuple: ...


# What signature() returns: (passage count, highest passage id, latest document
# update). Ingestion replaces a page's passages by deleting and re-inserting
# them, and ids only ever grow, so any added or replaced passage raises the
# highest id and any removed one lowers the count. The document timestamp
# catches title/URL-only changes (a trigger keeps updated_at current).


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

    _MATCH = text(
        "SELECT chunk_id, content, collection, slug, title, url, similarity, semantic_rank "
        "FROM match_chunks(CAST(:embedding AS extensions.halfvec), :k, CAST(:ids AS BIGINT[]))"
    )
    _PASSAGES = text(
        "SELECT c.id AS chunk_id, c.content, d.collection, d.slug, d.title, d.url "
        "FROM rag_chunks c JOIN rag_documents d ON d.id = c.document_id ORDER BY c.id"
    )

    def _match(self, embedding, k, candidate_ids) -> list[dict]:
        with self.session_factory() as db:
            rows = db.execute(
                self._MATCH, {"embedding": _vector_literal(embedding), "k": k, "ids": list(candidate_ids)}
            ).mappings().all()
        return [dict(row) for row in rows]

    async def match(self, embedding, k, candidate_ids) -> list[dict]:
        return await asyncio.to_thread(self._match, embedding, k, candidate_ids)

    def count_chunks(self) -> int:
        with self.session_factory() as db:
            return db.execute(text("SELECT count(*) FROM rag_chunks")).scalar_one()

    def load_passages(self) -> list[dict]:
        with self.session_factory() as db:
            return [dict(row) for row in db.execute(self._PASSAGES).mappings().all()]

    _SIGNATURE = text(
        "SELECT count(*), coalesce(max(id), 0), (SELECT max(updated_at) FROM rag_documents) FROM rag_chunks"
    )

    def signature(self) -> tuple:
        with self.session_factory() as db:
            count, max_id, updated = db.execute(self._SIGNATURE).one()
        return (count, max_id, updated.isoformat() if updated else None)


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

    async def match(self, embedding, k, candidate_ids) -> list[dict]:
        response = await self.client.post(
            f"{self.base}/rpc/match_chunks",
            json={"query_embedding": _vector_literal(embedding), "match_count": k,
                  "candidate_ids": list(candidate_ids)},
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

    def signature(self) -> tuple:
        with httpx.Client(headers=self.headers, timeout=30) as client:
            newest = client.get(f"{self.base}/rag_chunks", params={"select": "id", "order": "id.desc", "limit": 1})
            updated = client.get(
                f"{self.base}/rag_documents", params={"select": "updated_at", "order": "updated_at.desc", "limit": 1}
            )
            newest.raise_for_status()
            updated.raise_for_status()
        rows, docs = newest.json(), updated.json()
        return (self.count_chunks(), rows[0]["id"] if rows else 0, docs[0]["updated_at"] if docs else None)

    def load_passages(self, page: int = 1000) -> list[dict]:
        passages, start = [], 0
        with httpx.Client(headers=self.headers, timeout=60) as client:
            while True:
                response = client.get(
                    f"{self.base}/rag_chunks",
                    params={"select": "id,content,rag_documents(collection,slug,title,url)", "order": "id"},
                    headers={"Range": f"{start}-{start + page - 1}"},
                )
                response.raise_for_status()
                rows = response.json()
                passages.extend(
                    {"chunk_id": r["id"], "content": r["content"], **r["rag_documents"]} for r in rows
                )
                if len(rows) < page:
                    return passages
                start += page


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


def _document(row: dict) -> Document:
    return Document(
        page_content=row["content"],
        metadata={
            "source": row["collection"],
            "topic": row["slug"],
            "title": row["title"],
            "url": row["url"],
            "chunk_id": row["chunk_id"],
        },
    )


def fuse_with_cutoff(
    semantic: list[Document],
    keyword: list[Document],
    similarity: dict[int, float],
    weights: tuple[float, float],
    min_similarity: float | None,
    c: int = 60,
) -> list[Document]:
    """Weighted reciprocal rank fusion, after dropping keyword-only hits whose
    embedding similarity to the question is below min_similarity.

    A keyword hit that the semantic search also found is always kept: both
    agree it's relevant. One found by keywords alone has to be at least
    somewhat about the question, which removes pages that merely share a
    word ("Topical Guide: Ten" for "why ten percent").
    """
    semantic_ids = {doc.metadata["chunk_id"] for doc in semantic}
    if min_similarity is not None:
        keyword = [
            doc for doc in keyword
            if doc.metadata["chunk_id"] in semantic_ids
            or similarity.get(doc.metadata["chunk_id"], 0.0) >= min_similarity
        ]
    return reciprocal_rank_fusion([semantic, keyword], list(weights), c)


class PgvectorBm25Retriever:
    """Semantic search in Supabase (pgvector) + BM25 keyword search in memory.

    BM25 is built at startup from the passage text in rag_chunks (the same
    scorer and query rules as the FAISS backend's keyword search, about 30 MB
    of RAM), because Postgres full-text ranking chose noticeably worse
    keyword matches: on the 30-question eval it was worse than FAISS on 9
    questions, this backend on none (eval/retrieval_comparison.json). The
    store holds exactly the FAISS passages (scripts/ingest.py), so without
    the cutoff the two backends return the same sources. Each question makes one database call (match_chunks):
    the semantic top-k plus the similarity of the BM25 hits, used to drop
    keyword-only hits that aren't about the question.

    Passages ingested after startup are found by semantic search right away,
    and by keyword search once rebuild_keyword_index() has run (POST
    /admin/reindex, or the automatic check in refresh.py). The rebuild
    happens beside the live index, which keeps answering until it's swapped.
    """

    name = "pgvector_bm25"

    def __init__(
        self,
        transport: Transport,
        embeddings: Embeddings,
        k: int = 4,
        weights: tuple[float, float] = (0.5, 0.5),
        min_similarity: float | None = None,
        passages: list[dict] | None = None,
        drop_phrases: tuple[str, ...] = (),
        drop_words: frozenset[str] = frozenset(),
    ):
        self.transport = transport
        self.embeddings = embeddings
        self.k = k
        self.weights = weights
        self.min_similarity = min_similarity
        self.drop_phrases = drop_phrases
        self.drop_words = drop_words
        self.signature = None
        if passages is None:
            # Taken before loading, so a change made during the load shows up
            # as stale at the next check rather than being missed.
            self.signature = self._current_signature()
            passages = transport.load_passages()
        # (KeywordSearch or None, passage count), replaced as one value so a
        # query never sees one index's search with another's count.
        self._index = self._build_index(passages)

    def _build_index(self, rows: list[dict]) -> tuple[KeywordSearch | None, int]:
        docs = [_document(row) for row in rows]
        if not docs:
            return None, 0
        vectorizer = BM25Okapi([doc.page_content.split() for doc in docs])
        search = KeywordSearch(vectorizer, docs, k=self.k, drop_phrases=self.drop_phrases, drop_words=self.drop_words)
        return search, len(docs)

    def _current_signature(self) -> tuple | None:
        signature = getattr(self.transport, "signature", None)
        return signature() if signature else None

    @property
    def keyword(self) -> KeywordSearch | None:
        return self._index[0]

    def is_stale(self) -> bool:
        """Whether rag_chunks / rag_documents changed since the index was built (one small query)."""
        return self._current_signature() != self.signature

    def rebuild_keyword_index(self) -> int:
        """Reload the passages and swap in a new BM25 index; returns the passage count.

        Blocking (run it in a thread). Queries keep using the old index until
        the single assignment at the end, so none waits or sees half an index.
        """
        signature = self._current_signature()
        index = self._build_index(self.transport.load_passages())
        self._index = index
        self.signature = signature
        return index[1]

    async def search(self, query: str) -> tuple[list[Document], list[Document], dict[int, float]]:
        """The semantic hits, keyword hits and similarities before fusion (for evals)."""
        keyword = self.keyword  # read once: a rebuild may swap it meanwhile
        embedding, keyword_docs = await asyncio.gather(
            self.embeddings.aembed_query(query),
            keyword.ainvoke(query) if keyword else asyncio.sleep(0, result=[]),
        )
        rows = await self.transport.match(embedding, self.k, [d.metadata["chunk_id"] for d in keyword_docs])
        semantic = [_document(r) for r in sorted(
            (r for r in rows if r.get("semantic_rank")), key=lambda r: r["semantic_rank"])]
        similarity = {r["chunk_id"]: r["similarity"] for r in rows}
        return semantic, keyword_docs, similarity

    async def ainvoke(self, query: str) -> list[Document]:
        semantic, keyword_docs, similarity = await self.search(query)
        return fuse_with_cutoff(semantic, keyword_docs, similarity, self.weights, self.min_similarity)

    def chunk_counts(self) -> dict[str, int]:
        total = self._index[1]
        return {"vector": total, "keyword": total}
