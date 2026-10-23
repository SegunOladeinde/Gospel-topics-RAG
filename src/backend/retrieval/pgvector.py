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
import ctypes
import gc
from collections.abc import Iterable, Iterator
from typing import Protocol

import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import text

from src.backend.retrieval.hybrid import reciprocal_rank_fusion
from src.backend.retrieval.keyword import full_text_query
from src.backend.retrieval.sparse_bm25 import SparseBm25


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.8g}" for x in vector) + "]"


class Transport(Protocol):
    async def search(self, embedding: list[float], query_text: str, k: int) -> list[dict]: ...
    async def match(self, embedding: list[float], k: int, candidate_ids: list[int]) -> list[dict]: ...
    def count_chunks(self) -> int: ...
    def load_passages(self) -> list[dict]: ...
    def iter_passages(self) -> Iterator[tuple[int, str]]: ...
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
        "FROM match_chunks(CAST(:embedding AS extensions.halfvec), :k, CAST(:ids AS BIGINT[]), "
        "CAST(:collections AS TEXT[]))"
    )
    # With a collection filter, HNSW would otherwise filter only its first
    # ef_search (40) candidates, and return fewer or different passages when
    # most of those are in other collections. pgvector 0.8 keeps scanning
    # until the filtered top-k is complete, and a wider ef_search made the
    # filtered top-4 match an exact search on every eval question that
    # differed (40 missed e.g. Topical Guide "Nephites" for "Who was Nephi?").
    # SET LOCAL lasts one transaction.
    _FILTERED_SCAN = (
        text("SET LOCAL hnsw.iterative_scan = strict_order"),
        text("SET LOCAL hnsw.ef_search = 200"),
    )
    _PASSAGES = text(
        "SELECT c.id AS chunk_id, c.content, d.collection, d.slug, d.title, d.url "
        "FROM rag_chunks c JOIN rag_documents d ON d.id = c.document_id ORDER BY c.id"
    )

    def _match(self, embedding, k, candidate_ids, collections=None) -> list[dict]:
        with self.session_factory() as db:
            if collections:
                for setting in self._FILTERED_SCAN:
                    db.execute(setting)
            rows = db.execute(
                self._MATCH,
                {"embedding": _vector_literal(embedding), "k": k, "ids": list(candidate_ids),
                 "collections": list(collections) if collections else None},
            ).mappings().all()
        return [dict(row) for row in rows]

    async def match(self, embedding, k, candidate_ids, collections=None) -> list[dict]:
        return await asyncio.to_thread(self._match, embedding, k, candidate_ids, collections)

    def count_chunks(self) -> int:
        with self.session_factory() as db:
            return db.execute(text("SELECT count(*) FROM rag_chunks")).scalar_one()

    def load_passages(self) -> list[dict]:
        with self.session_factory() as db:
            return [dict(row) for row in db.execute(self._PASSAGES).mappings().all()]

    _PASSAGE_TEXT = text("SELECT id, content FROM rag_chunks ORDER BY id")
    _PASSAGE_TEXT_IN = text("SELECT id, content FROM rag_chunks WHERE collection = ANY(:c) ORDER BY id")

    def iter_passages(self, batch: int = 500, collections=None) -> Iterator[tuple[int, str]]:
        """(chunk id, text) in id order, streamed batch by batch (server-side
        cursor), so the passages are never all in memory at once. Only the
        given collections' passages when collections is set."""
        with self.session_factory() as db:
            if collections:
                query = self._PASSAGE_TEXT_IN.execution_options(yield_per=batch)
                result = db.execute(query, {"c": list(collections)})
            else:
                result = db.execute(self._PASSAGE_TEXT.execution_options(yield_per=batch))
            for chunk_id, content in result:
                yield chunk_id, content

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

    async def match(self, embedding, k, candidate_ids, collections=None) -> list[dict]:
        # REST can't SET hnsw.iterative_scan, so a filtered search here is
        # approximate the way SqlTransport's isn't (scripts only, not the app).
        response = await self.client.post(
            f"{self.base}/rpc/match_chunks",
            json={"query_embedding": _vector_literal(embedding), "match_count": k,
                  "candidate_ids": list(candidate_ids), "collections": list(collections) if collections else None},
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

    def iter_passages(self, page: int = 1000, collections=None) -> Iterator[tuple[int, str]]:
        """(chunk id, text) in id order, a page at a time."""
        start = 0
        params = {"select": "id,content", "order": "id"}
        if collections:
            params["collection"] = "in.(" + ",".join(f'"{c}"' for c in collections) + ")"
        with httpx.Client(headers=self.headers, timeout=60) as client:
            while True:
                response = client.get(
                    f"{self.base}/rag_chunks",
                    params=params,
                    headers={"Range": f"{start}-{start + page - 1}"},
                )
                response.raise_for_status()
                rows = response.json()
                for row in rows:
                    yield row["id"], row["content"]
                if len(rows) < page:
                    return
                start += page

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


def release_memory() -> None:
    """Hand freed memory back to the OS after a big build: collect garbage, then
    ask glibc to trim its heaps (Linux; elsewhere just the gc). Without the
    trim, the memory a build used stays in the process's RSS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


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

    BM25 is built at startup from the passage text in rag_chunks, streamed
    from the database, into a SparseBm25 (sparse_bm25.py: the same scores and
    query rules as the FAISS backend's keyword search in about 10 MB, holding
    chunk ids only). Postgres full-text ranking chose noticeably worse
    keyword matches: on the 30-question eval it was worse than FAISS on 9
    questions, this backend on none (eval/retrieval_comparison.json). The
    store holds exactly the FAISS passages (scripts/ingest.py), so without
    the cutoff the two backends return the same sources. Each question makes
    one database call (match_chunks): the semantic top-k plus the BM25 hits'
    text, metadata and similarity, used to drop keyword-only hits that
    aren't about the question.

    Passages ingested after startup are found by semantic search right away,
    and by keyword search once rebuild_keyword_index() has run (POST
    /admin/reindex, or the automatic check in refresh.py). The rebuild
    happens beside the live index, which keeps answering until it's swapped;
    the live index is small, so the two together stay small.
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
        collections: tuple[str, ...] | None = None,
    ):
        """collections: search only these (semantic search filtered in SQL,
        BM25 built from their passages alone); None searches everything."""
        self.transport = transport
        self.collections = tuple(collections) if collections else None
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
            self._index = self._build_index(self._stream_passages())
        else:
            self._index = self._build_index(
                (row["chunk_id"], row["content"]) for row in passages if self._wanted(row.get("collection"))
            )

    def _wanted(self, collection: str | None) -> bool:
        return self.collections is None or collection in self.collections

    def _stream_passages(self) -> Iterable[tuple[int, str]]:
        stream = getattr(self.transport, "iter_passages", None)
        if stream:
            return stream(collections=self.collections) if self.collections else stream()
        return ((row["chunk_id"], row["content"]) for row in self.transport.load_passages()
                if self._wanted(row.get("collection")))

    def _build_index(self, passages: Iterable[tuple[int, str]]) -> SparseBm25 | None:
        index = SparseBm25.build(passages, k=self.k, drop_phrases=self.drop_phrases, drop_words=self.drop_words)
        release_memory()
        return index if index.size else None

    def _current_signature(self) -> tuple | None:
        signature = getattr(self.transport, "signature", None)
        return signature() if signature else None

    @property
    def keyword(self) -> SparseBm25 | None:
        return self._index

    def is_stale(self) -> bool:
        """Whether rag_chunks / rag_documents changed since the index was built (one small query)."""
        return self._current_signature() != self.signature

    def rebuild_keyword_index(self) -> int:
        """Reload the passages and swap in a new BM25 index; returns the passage count.

        Blocking (run it in a thread). Queries keep using the old index until
        the single assignment at the end, so none waits or sees half an index.
        """
        signature = self._current_signature()
        index = self._build_index(self._stream_passages())
        self._index = index  # one assignment: the swap
        self.signature = signature
        release_memory()  # the old index, now unreferenced
        return index.size if index else 0

    async def search(
        self, query: str, embedding: list[float] | None = None
    ) -> tuple[list[Document], list[Document], dict[int, float]]:
        """The semantic hits, keyword hits and similarities before fusion (for
        evals and GroupedRetriever). embedding: the query's, if already known."""
        keyword = self.keyword  # read once: a rebuild may swap it meanwhile
        embedding, keyword_ids = await asyncio.gather(
            self.embeddings.aembed_query(query) if embedding is None else asyncio.sleep(0, result=embedding),
            keyword.ainvoke_ids(query) if keyword else asyncio.sleep(0, result=[]),
        )
        # The keyword hits' text and metadata come back with the semantic
        # search; a passage deleted since the index was built is just skipped.
        if self.collections:
            rows = await self.transport.match(embedding, self.k, keyword_ids, collections=self.collections)
        else:
            rows = await self.transport.match(embedding, self.k, keyword_ids)
        by_id = {r["chunk_id"]: r for r in rows}
        semantic = [_document(r) for r in sorted(
            (r for r in rows if r.get("semantic_rank")), key=lambda r: r["semantic_rank"])]
        keyword_docs = [_document(by_id[i]) for i in keyword_ids if i in by_id]
        similarity = {r["chunk_id"]: r["similarity"] for r in rows}
        return semantic, keyword_docs, similarity

    async def ainvoke(self, query: str, embedding: list[float] | None = None) -> list[Document]:
        semantic, keyword_docs, similarity = await self.search(query, embedding)
        return fuse_with_cutoff(semantic, keyword_docs, similarity, self.weights, self.min_similarity)

    async def ainvoke_scored(self, query: str, embedding: list[float] | None = None) -> list[tuple[Document, float]]:
        """ainvoke(), with each passage's similarity to the question."""
        semantic, keyword_docs, similarity = await self.search(query, embedding)
        fused = fuse_with_cutoff(semantic, keyword_docs, similarity, self.weights, self.min_similarity)
        return [(doc, similarity.get(doc.metadata["chunk_id"], 0.0)) for doc in fused]

    def chunk_counts(self) -> dict[str, int]:
        index = self._index
        total = index.size if index else 0
        return {"vector": total, "keyword": total}


class GroupedRetriever:
    """Two PgvectorBm25Retrievers over separate groups of collections.

    The core group (the Bible Dictionary, Topical Guide and Gospel Topics)
    is searched exactly as it was on its own: its own semantic top-k and its
    own BM25, so collections added later can't crowd out its sources or
    change its BM25 weights. The added group (the Institute manuals and
    Church History Topics) is searched the same way, and up to max_added of
    its passages follow the core ones when their similarity to the question
    is at least min_similarity, or (relative_to_core) at least that of the
    best core passage: when the core collections have nothing close, a
    manual section that is the closest match still gets in. With one shared top-k instead, manual
    passages displaced older sources on 16 of 27 existing eval questions
    (scripts/eval_collections.py).

    One embedding per question, shared by both searches, which run together.
    """

    name = "pgvector_bm25"

    def __init__(self, core: PgvectorBm25Retriever, added: PgvectorBm25Retriever,
                 max_added: int, min_similarity: float, relative_to_core: bool = False):
        self.core = core
        self.added = added
        self.max_added = max_added
        self.min_similarity = min_similarity
        self.relative_to_core = relative_to_core
        self.embeddings = core.embeddings

    async def ainvoke(self, query: str) -> list[Document]:
        embedding = await self.embeddings.aembed_query(query)
        core, added = await asyncio.gather(
            self.core.ainvoke_scored(query, embedding), self.added.ainvoke_scored(query, embedding)
        )
        floor = self.min_similarity
        if self.relative_to_core and core:
            floor = min(floor, max(similarity for _, similarity in core))
        extra = [doc for doc, similarity in added if similarity >= floor][: self.max_added]
        return [doc for doc, _ in core] + extra

    def is_stale(self) -> bool:
        return self.core.is_stale() or self.added.is_stale()

    def rebuild_keyword_index(self) -> int:
        """Rebuild both groups' BM25 indexes, one after the other (each is
        swapped in as soon as it's built); returns the passage count."""
        return self.core.rebuild_keyword_index() + self.added.rebuild_keyword_index()

    def chunk_counts(self) -> dict[str, int]:
        core, added = self.core.chunk_counts(), self.added.chunk_counts()
        return {key: core[key] + added[key] for key in core}
