"""Unit tests for the pgvector backend (no database needed).

The SQL side (hybrid_search, SqlTransport, SqlRagStore) runs against a real
pgvector Postgres in tests/test_pgvector_postgres.py.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import src.backend.main as backend
from src.backend.history import _link, _source_id, _stored_source
from src.backend.main import QueryResponse, collect_unique_links, load_retriever
from src.backend.retrieval import PgvectorRetriever, RestTransport, Retriever, SqlTransport
from src.backend.retrieval.keyword import full_text_query
from src.backend.source_links import SourceLink

ROW = {
    "chunk_id": 1, "content": "Aaron's descendants held the priesthood.", "collection": "topical_guide",
    "slug": "aaron-descendants-of", "title": "Aaron, Descendants of",
    "url": "https://www.churchofjesuschrist.org/study/scriptures/tg/aaron-descendants-of?lang=eng",
    "score": 0.03, "semantic_rank": 1, "full_text_rank": 1,
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0] * 1536 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 1536


class FakeTransport:
    def __init__(self, rows=(ROW,), total=9956):
        self.rows, self.total, self.calls = list(rows), total, []

    async def search(self, embedding, query_text, k):
        self.calls.append((len(embedding), query_text, k))
        return self.rows

    def count_chunks(self):
        return self.total


# ---------------------------------------------------------------------------
# Full-text query
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("question", "query"),
    [
        ("What does the Bible Dictionary say about grace?", "grace"),  # collection name dropped
        ("What does the Topical Guide say about faith?", "faith"),
        ("How can I increase my faith?", "faith"),  # generic verb dropped
        ("Introduce me to the plan of salvation", "plan | salvation"),
        ("What is the Bible?", "bible"),  # "Bible" alone is a topic, kept
        ("List the steps of repentance", "repentance"),  # instruction words still dropped
        ("What is Lehi's dream?", "lehi | dream"),  # only [a-z0-9]: always valid tsquery
        ("What is it?", ""),  # nothing to search: semantic only
        ("Faith, faith & FAITH?!", "faith"),
    ],
)
def test_full_text_query(question, query):
    assert full_text_query(question) == query


# ---------------------------------------------------------------------------
# PgvectorRetriever
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_retriever_returns_documents_with_title_and_url():
    transport = FakeTransport()
    retriever = PgvectorRetriever(transport, FakeEmbeddings(), k=4)
    docs = await retriever.ainvoke("How can I increase my faith in Aaron's priesthood?")
    assert docs[0].page_content == ROW["content"]
    assert docs[0].metadata == {
        "source": "topical_guide", "topic": "aaron-descendants-of",
        "title": "Aaron, Descendants of", "url": ROW["url"],
    }
    assert transport.calls == [(1536, "faith | aaron | priesthood", 4)]


def test_retriever_reports_one_table_for_both_searches():
    retriever = PgvectorRetriever(FakeTransport(total=9956), FakeEmbeddings())
    assert isinstance(retriever, Retriever)
    assert retriever.chunk_counts() == {"vector": 9956, "keyword": 9956}
    assert retriever.name == "pgvector"


def test_links_use_real_titles_and_canonical_urls():
    docs = [
        Document(page_content="a", metadata={"source": "topical_guide", "topic": "aaron-descendants-of",
                                             "title": "Aaron, Descendants of", "url": ROW["url"]}),
        Document(page_content="b", metadata={"source": "topics_and_questions", "topic": "priest",
                                             "title": "Aaronic Priesthood",
                                             "url": "https://www.churchofjesuschrist.org/study/manual/gospel-topics/aaronic-priesthood?lang=eng"}),
        Document(page_content="c", metadata={"source": "topical_guide", "topic": "faith"}),  # FAISS-style
        Document(page_content="d", metadata={"source": "topical_guide", "topic": "aaron-descendants-of",
                                             "title": "Aaron, Descendants of", "url": ROW["url"]}),  # duplicate
    ]
    links = collect_unique_links(docs)
    assert [link.label for link in links] == [
        "Topical Guide: Aaron, Descendants of",
        "Gospel Topics: Aaronic Priesthood",
        "Topical Guide: Faith",
    ]
    assert links[1].url.endswith("/gospel-topics/aaronic-priesthood?lang=eng")  # the moved page's new URL


def test_query_response_keeps_given_links_and_builds_missing_ones():
    link = SourceLink(id="topical_guide/x", label="Topical Guide: The X", url=None)
    assert QueryResponse(answer="a", sources=["topical_guide/x"], source_links=[link]).source_links == [link]
    built = QueryResponse(answer="a", sources=["topical_guide/faith"]).source_links
    assert built[0].label == "Topical Guide: Faith"


def test_history_stores_links_and_reads_old_id_only_rows():
    link = SourceLink(id="topical_guide/aaron-descendants-of", label="Topical Guide: Aaron, Descendants of", url=ROW["url"])
    stored = _stored_source(link)
    assert stored == link.model_dump()
    assert _source_id(stored) == link.id and _link(stored) == link
    # Chats saved before links were stored:
    assert _source_id("topical_guide/faith") == "topical_guide/faith"
    assert _link("topical_guide/faith").label == "Topical Guide: Faith"


# ---------------------------------------------------------------------------
# Choosing the backend
# ---------------------------------------------------------------------------

def test_unknown_backend_is_refused(monkeypatch):
    monkeypatch.setenv("RETRIEVER_BACKEND", "elastic")
    with pytest.raises(RuntimeError, match="faiss, pgvector"):
        backend.retriever_backend()


def test_default_backend_is_faiss(monkeypatch):
    monkeypatch.delenv("RETRIEVER_BACKEND", raising=False)
    assert backend.retriever_backend() == "faiss"


def test_pgvector_needs_a_database_or_rest_credentials(monkeypatch):
    monkeypatch.setenv("RETRIEVER_BACKEND", "pgvector")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        load_retriever(None)


def test_pgvector_uses_the_database_connection_when_there_is_one(monkeypatch):
    monkeypatch.setenv("RETRIEVER_BACKEND", "pgvector")
    monkeypatch.setattr(SqlTransport, "count_chunks", lambda self: 7)
    retriever = load_retriever(MagicMock())
    assert isinstance(retriever, PgvectorRetriever)
    assert isinstance(retriever.transport, SqlTransport)
    assert retriever.chunk_counts()["vector"] == 7


def test_pgvector_falls_back_to_rest(monkeypatch):
    monkeypatch.setenv("RETRIEVER_BACKEND", "pgvector")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-key")
    monkeypatch.setattr(RestTransport, "count_chunks", lambda self: 3)
    retriever = load_retriever(None)
    assert isinstance(retriever.transport, RestTransport)


# ---------------------------------------------------------------------------
# REST transport
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_rest_transport_calls_the_rpc_with_the_service_key():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("apikey")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=[ROW])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                               headers={"apikey": "test-service-key"})
    transport = RestTransport("https://example.supabase.co", "test-service-key", client=client)
    rows = await transport.search([0.5, 0.25], "faith", 4)
    assert rows == [ROW]
    assert seen["url"] == "https://example.supabase.co/rest/v1/rpc/hybrid_search"
    assert seen["key"] == "test-service-key"
    assert seen["body"] == {"query_embedding": "[0.5,0.25]", "query_text": "faith", "match_count": 4}


def test_rest_transport_counts_chunks_from_content_range(monkeypatch):
    def fake_get(url, params, headers, timeout):
        assert headers["Prefer"] == "count=exact"
        return httpx.Response(206, headers={"content-range": "0-0/9956"},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert RestTransport("https://example.supabase.co", "k").count_chunks() == 9956


# ---------------------------------------------------------------------------
# pgvector_bm25: semantic search in Postgres + BM25 in memory
# ---------------------------------------------------------------------------

from src.backend.retrieval import PgvectorBm25Retriever  # noqa: E402
from src.backend.retrieval.pgvector import fuse_with_cutoff  # noqa: E402

PASSAGES = [
    {"chunk_id": 1, "content": "Tithing is one tenth of our increase.", "collection": "topical_guide",
     "slug": "tithing", "title": "Tithing", "url": "https://www.churchofjesuschrist.org/tg/tithing"},
    {"chunk_id": 2, "content": "The Ten Commandments were given to Moses.", "collection": "bible_dictionary",
     "slug": "ten-commandments", "title": "Ten Commandments", "url": "https://www.churchofjesuschrist.org/bd/ten"},
    {"chunk_id": 3, "content": "Fast offerings help the poor.", "collection": "topics_and_questions",
     "slug": "fasting", "title": "Fasting", "url": "https://www.churchofjesuschrist.org/gt/fasting"},
]


def doc_for(chunk_id):
    row = next(p for p in PASSAGES if p["chunk_id"] == chunk_id)
    return Document(page_content=row["content"], metadata={"source": row["collection"], "topic": row["slug"],
                                                           "chunk_id": chunk_id})


class Bm25Transport:
    """match(): chunk 1 is the semantic hit; similarities as given."""

    def __init__(self, similarity):
        self.similarity, self.asked = similarity, []

    async def match(self, embedding, k, candidate_ids):
        self.asked.append(list(candidate_ids))
        rows = [{**PASSAGES[0], "similarity": self.similarity[1], "semantic_rank": 1}]
        rows += [{**next(p for p in PASSAGES if p["chunk_id"] == i), "similarity": self.similarity[i], "semantic_rank": None}
                 for i in candidate_ids if i != 1]
        return rows

    def load_passages(self):
        return PASSAGES

    def count_chunks(self):
        return len(PASSAGES)


def test_cutoff_drops_keyword_only_hits_that_are_off_topic():
    semantic = [doc_for(1)]
    keyword = [doc_for(1), doc_for(2), doc_for(3)]
    fused = fuse_with_cutoff(semantic, keyword, {1: 0.6, 2: 0.12, 3: 0.41}, (0.5, 0.5), 0.30)
    assert [d.metadata["chunk_id"] for d in fused] == [1, 3]  # 2 shares only the word "ten"


def test_no_cutoff_keeps_every_keyword_hit():
    fused = fuse_with_cutoff([doc_for(1)], [doc_for(2)], {1: 0.6, 2: 0.1}, (0.5, 0.5), None)
    assert {d.metadata["chunk_id"] for d in fused} == {1, 2}


def test_keyword_hit_also_found_semantically_is_kept_whatever_its_similarity():
    fused = fuse_with_cutoff([doc_for(2)], [doc_for(2)], {2: 0.05}, (0.5, 0.5), 0.30)
    assert [d.metadata["chunk_id"] for d in fused] == [2]


@pytest.mark.anyio
async def test_pgvector_bm25_retriever_end_to_end():
    transport = Bm25Transport({1: 0.62, 2: 0.10, 3: 0.20})
    retriever = PgvectorBm25Retriever(transport, FakeEmbeddings(), k=4, min_similarity=0.30)
    assert retriever.name == "pgvector_bm25"
    assert retriever.chunk_counts() == {"vector": 3, "keyword": 3}

    docs = await retriever.ainvoke("Why is tithing a tenth and not ten commandments?")
    # BM25 found the tithing and Ten Commandments passages; the database was
    # asked for their similarity in the same call as the semantic search.
    assert set(transport.asked[0]) >= {1, 2}
    assert [d.metadata["topic"] for d in docs] == ["tithing"]
    assert docs[0].metadata["title"] == "Tithing" and docs[0].metadata["url"].endswith("/tg/tithing")


def test_load_retriever_builds_pgvector_bm25(monkeypatch):
    monkeypatch.setenv("RETRIEVER_BACKEND", "pgvector_bm25")
    monkeypatch.setattr(SqlTransport, "load_passages", lambda self: PASSAGES)
    monkeypatch.setattr(SqlTransport, "signature", lambda self: (3, 3, None))
    retriever = load_retriever(MagicMock())
    assert isinstance(retriever, PgvectorBm25Retriever)
    assert retriever.min_similarity == backend.PGVECTOR_BM25_MIN_SIMILARITY == 0.30
    assert retriever.chunk_counts()["keyword"] == 3


@pytest.mark.anyio
async def test_rest_transport_match_and_load_passages(monkeypatch):
    def handler(request: httpx.Request):
        if request.url.path.endswith("/rpc/match_chunks"):
            body = json.loads(request.content)
            assert body["candidate_ids"] == [7, 9] and body["match_count"] == 4
            return httpx.Response(200, json=[{"chunk_id": 7, "similarity": 0.5}])
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = RestTransport("https://example.supabase.co", "k", client=client)
    assert await transport.match([0.1], 4, [7, 9]) == [{"chunk_id": 7, "similarity": 0.5}]

    pages = [
        [{"id": i, "content": f"c{i}", "rag_documents": {"collection": "topical_guide", "slug": f"s{i}",
                                                         "title": f"T{i}", "url": None}} for i in range(2)],
        [],
    ]

    class SyncClient:
        def __init__(self, *a, **kw):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, params, headers):
            rows = pages[min(self.calls, 1)]
            self.calls += 1
            return httpx.Response(200, json=rows, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "Client", SyncClient)
    loaded = RestTransport("https://example.supabase.co", "k", client=client).load_passages(page=2)
    assert [p["chunk_id"] for p in loaded] == [0, 1]
    assert loaded[0]["slug"] == "s0" and loaded[0]["collection"] == "topical_guide"
