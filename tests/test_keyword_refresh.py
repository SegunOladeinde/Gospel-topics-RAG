"""pgvector_bm25: the BM25 query filter, rebuilding the in-memory index
without a restart (src/backend/retrieval/refresh.py) and POST /admin/reindex."""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

import src.backend.main as backend
from src.backend.main import app
from src.backend.retrieval import PgvectorBm25Retriever, SqlTransport
from src.backend.retrieval.keyword import (
    COLLECTION_PHRASES,
    GENERIC_VERBS,
    KeywordSearch,
    collection_phrases,
    remove_phrases,
)
from src.backend.retrieval.pgvector import _document
from src.backend.retrieval.refresh import KeywordIndexRefresher

VALID_KEY = "test-key"


def row(chunk_id, content, slug, collection="topical_guide"):
    return {"chunk_id": chunk_id, "content": content, "collection": collection, "slug": slug,
            "title": slug.title(), "url": f"https://example.org/{slug}"}


PASSAGES = [
    row(1, "Grace is divine help through the Atonement.", "grace"),
    row(2, "Bible Dictionary introduction to this Dictionary of the Bible.", "introduction", "bible_dictionary"),
    row(3, "Increase in learning and wisdom.", "increase"),
    row(4, "Faith in Jesus Christ is a principle of action.", "faith"),
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0, 1.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 1.0]


class Store:
    """A transport over a mutable passage list; signature() as SqlTransport computes it."""

    def __init__(self, passages):
        self.passages = list(passages)
        self.updated = "2026-10-21T00:00:00+00:00"
        self.loads = 0

    def load_passages(self):
        self.loads += 1
        return list(self.passages)

    def signature(self):
        return (len(self.passages), max((p["chunk_id"] for p in self.passages), default=0), self.updated)

    async def match(self, embedding, k, candidate_ids):
        return [{**p, "similarity": 0.9, "semantic_rank": None} for p in self.passages if p["chunk_id"] in candidate_ids]


def keyword_search(drop_phrases=(), drop_words=frozenset()):
    docs = [_document(r) for r in PASSAGES]
    return KeywordSearch(BM25Okapi([d.page_content.split() for d in docs]), docs, k=4,
                         drop_phrases=drop_phrases, drop_words=drop_words)


# --- query filter -----------------------------------------------------------

def test_collection_phrases_add_yaml_names_and_labels_longest_first():
    phrases = collection_phrases(["church_history", "Church History", "Gospel Topics"])
    assert "church history" in phrases
    assert set(COLLECTION_PHRASES) <= set(phrases)
    assert phrases.index("gospel topics") < phrases.index("gospel topic")


def test_remove_phrases_matches_whole_words_only():
    assert remove_phrases("What does the Bible Dictionary say?", ("bible dictionary",)).split() == ["What", "does", "the", "say?"]
    # "the Bible" alone is still a topic.
    assert "Bible" in remove_phrases("What is the Bible?", ("bible dictionary",))


def test_unfiltered_search_is_unchanged():
    topics = [d.metadata["topic"] for d in keyword_search().search("What does the Bible Dictionary say about grace?")]
    assert "introduction" in topics  # the problem the filter fixes


def test_filter_drops_collection_names_and_generic_verbs():
    search = keyword_search(collection_phrases(), GENERIC_VERBS)
    assert [d.metadata["topic"] for d in search.search("What does the Bible Dictionary say about grace?")] == ["grace"]
    assert [d.metadata["topic"] for d in search.search("How can I increase my faith?")] == ["faith"]


def test_load_retriever_gives_pgvector_bm25_the_filter(real_startup, monkeypatch):
    monkeypatch.setenv("RETRIEVER_BACKEND", "pgvector_bm25")
    monkeypatch.setattr(SqlTransport, "iter_passages", lambda self: ((r["chunk_id"], r["content"]) for r in PASSAGES))
    monkeypatch.setattr(SqlTransport, "signature", lambda self: (4, 4, None))
    retriever = backend.load_retriever(MagicMock())
    assert isinstance(retriever, PgvectorBm25Retriever)
    # From data/collections.yaml as well as the built-in list.
    assert {"bible dictionary", "topics and questions", "gospel topics"} <= set(retriever.keyword.drop_phrases)
    assert retriever.keyword.drop_words == GENERIC_VERBS


# --- rebuilding the index ---------------------------------------------------

def test_rebuild_swaps_in_new_passages_and_signature():
    store = Store(PASSAGES[:2])
    retriever = PgvectorBm25Retriever(store, FakeEmbeddings(), k=4)
    assert retriever.chunk_counts()["keyword"] == 2 and not retriever.is_stale()

    store.passages.append(PASSAGES[3])
    assert retriever.is_stale()
    old_index = retriever.keyword
    assert retriever.rebuild_keyword_index() == 3
    assert retriever.keyword is not old_index
    assert retriever.chunk_counts() == {"vector": 3, "keyword": 3}
    assert not retriever.is_stale()
    assert retriever.keyword.search_ids("faith") == [4]


def test_replaced_passages_and_document_edits_count_as_changes():
    store = Store(PASSAGES)
    retriever = PgvectorBm25Retriever(store, FakeEmbeddings())
    store.passages[-1] = {**PASSAGES[-1], "chunk_id": 5}  # re-ingested: same count, new id
    assert retriever.is_stale()
    retriever.rebuild_keyword_index()
    store.updated = "2026-10-22T00:00:00+00:00"  # a title or URL changed
    assert retriever.is_stale()


@pytest.mark.anyio
async def test_queries_keep_working_during_a_rebuild():
    store = Store(PASSAGES)
    retriever = PgvectorBm25Retriever(store, FakeEmbeddings(), k=4)
    started, release = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()
    real_load = store.load_passages

    def slow_load():
        loop.call_soon_threadsafe(started.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return real_load()

    store.load_passages = slow_load
    refresher = KeywordIndexRefresher(retriever, interval_seconds=0)
    assert refresher.start_rebuild()
    await started.wait()
    assert refresher.rebuilding
    assert not refresher.start_rebuild()  # one at a time
    # The old index answers meanwhile.
    _, keyword, _ = await retriever.search("grace")
    assert [d.metadata["topic"] for d in keyword] == ["grace"]
    release.set()
    while refresher.rebuilding:
        await asyncio.sleep(0.01)
    assert refresher.refreshed_at is not None
    await refresher.stop()


@pytest.mark.anyio
async def test_check_once_rebuilds_only_when_the_store_changed():
    store = Store(PASSAGES[:3])
    retriever = PgvectorBm25Retriever(store, FakeEmbeddings())
    refresher = KeywordIndexRefresher(retriever, interval_seconds=0)
    assert await refresher.check_once() is False
    assert store.loads == 1
    store.passages.append(PASSAGES[3])
    assert await refresher.check_once() is True
    assert retriever.chunk_counts()["keyword"] == 4


@pytest.mark.anyio
async def test_watcher_checks_on_its_interval_and_survives_errors():
    store = Store(PASSAGES[:3])
    retriever = PgvectorBm25Retriever(store, FakeEmbeddings())
    calls = []
    real = store.signature

    def flaky():
        calls.append(1)
        if len(calls) == 2:  # the first check (after the build's own call) fails
            raise ConnectionError("db blip")
        return real()

    store.signature = flaky
    store.passages.append(PASSAGES[3])
    refresher = KeywordIndexRefresher(retriever, interval_seconds=0.02)
    refresher.start()
    for _ in range(200):
        if retriever.chunk_counts()["keyword"] == 4:
            break
        await asyncio.sleep(0.01)
    await refresher.stop()
    assert retriever.chunk_counts()["keyword"] == 4


def test_refresh_interval_env(monkeypatch):
    monkeypatch.delenv("BM25_REFRESH_SECONDS", raising=False)
    assert backend.bm25_refresh_seconds() == 600
    monkeypatch.setenv("BM25_REFRESH_SECONDS", "0")
    assert backend.bm25_refresh_seconds() == 0
    monkeypatch.setenv("BM25_REFRESH_SECONDS", "soon")
    assert backend.bm25_refresh_seconds() == 600


# --- POST /admin/reindex ----------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("APP_API_KEY", VALID_KEY)
    with TestClient(app) as c:
        yield c


ADMIN = {"X-API-Key": VALID_KEY, "X-User-Id": "user_1", "X-User-Role": "admin"}


def test_reindex_needs_the_api_key_and_an_admin(client):
    assert client.post("/admin/reindex", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/admin/reindex", headers={"X-API-Key": VALID_KEY, "X-User-Id": "user_1"}).status_code == 403
    # A guest can't be an admin, whatever the role header says.
    assert client.post("/admin/reindex", headers={"X-API-Key": VALID_KEY, "X-User-Role": "admin"}).status_code == 403


def test_reindex_is_refused_for_backends_without_a_memory_index(client):
    response = client.post("/admin/reindex", headers=ADMIN)
    assert response.status_code == 409
    assert "faiss" in response.json()["detail"]


def test_reindex_starts_a_background_rebuild(client, monkeypatch):
    store = Store(PASSAGES[:2])
    retriever = PgvectorBm25Retriever(store, FakeEmbeddings())
    store.passages = list(PASSAGES)
    monkeypatch.setattr(app.state, "retriever", retriever)
    monkeypatch.setattr(app.state, "keyword_refresher", KeywordIndexRefresher(retriever, 0), raising=False)
    response = client.post("/admin/reindex", headers=ADMIN)
    assert response.status_code == 202
    assert response.json()["status"] == "started"
    for _ in range(200):
        health = client.get("/health").json()["retriever"]
        if health["keyword_index"]["refreshed_at"]:
            break
    assert health["chunks"] == 4


# --- memory: chunk ids only in memory, text from the database ---------------

@pytest.mark.anyio
async def test_keyword_hit_text_comes_from_match_and_deleted_passages_are_skipped():
    store = Store(PASSAGES)
    retriever = PgvectorBm25Retriever(store, FakeEmbeddings(), k=4)
    store.passages = [p for p in PASSAGES if p["chunk_id"] != 4]  # deleted after the build
    _, keyword, _ = await retriever.search("grace faith")
    assert [d.metadata["topic"] for d in keyword] == ["grace"]
    assert keyword[0].page_content == PASSAGES[0]["content"]


def test_rest_transport_streams_passage_text_page_by_page(monkeypatch):
    import httpx

    from src.backend.retrieval import RestTransport

    seen = []

    def handler(request):
        seen.append(request.headers["Range"])
        start = int(request.headers["Range"].split("-")[0])
        rows = [{"id": i, "content": f"text {i}"} for i in range(start, min(start + 2, 5))]
        return httpx.Response(200, json=rows)

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    transport = RestTransport("https://example.supabase.co", "k")
    assert list(transport.iter_passages(page=2)) == [(i, f"text {i}") for i in range(5)]
    assert seen == ["0-1", "2-3", "4-5"]


def test_release_memory_is_safe_everywhere():
    from src.backend.retrieval.pgvector import release_memory

    release_memory()  # malloc_trim on Linux, just gc elsewhere


def test_pgvector_backends_do_not_import_faiss():
    import subprocess
    import sys

    code = ("import os, sys; os.environ['RETRIEVER_BACKEND'] = 'pgvector_bm25'; import src.backend.main; "
            "print('faiss' in sys.modules, 'rank_bm25' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "False False"


def test_malloc_arena_cap_is_safe_everywhere(monkeypatch):
    import sys as _sys

    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
    applied = backend.limit_malloc_arenas()
    assert applied is _sys.platform.startswith("linux")  # glibc on Linux (CI, Render)
    monkeypatch.setenv("MALLOC_ARENA_MAX", "4")
    assert backend.limit_malloc_arenas() is False  # the env var wins
