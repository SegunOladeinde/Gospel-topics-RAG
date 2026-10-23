"""pgvector backend against a real Postgres with the vector extension.

Opt-in like the other `postgres` tests: skipped unless TEST_DATABASE_URL is
set. CI runs them against the pgvector/pgvector:pg16 service. Uses fake,
deterministic embeddings, so no OpenAI calls.
"""

import hashlib
import os
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text

from src.backend.database import get_session_factory, init_db
from src.backend.ingestion.config import CollectionConfig, IngestConfig
from src.backend.ingestion.documents import load_documents
from src.backend.ingestion.pipeline import run_ingestion
from src.backend.ingestion.stores import SqlRagStore
import src.backend.main as backend
from src.backend.main import app
from src.backend.retrieval import PgvectorRetriever, SqlTransport
from src.backend.retrieval.keyword import full_text_query

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to run against a real Postgres"),
]

TOPICS = {"faith": "Faith in Jesus Christ is a principle of action and power.",
          "baptism": "Baptism by immersion is the first ordinance of the gospel.",
          "tithing": "Tithing is one tenth of our increase, paid to the Lord."}


def fake_vector(text_: str) -> list[float]:
    """Deterministic 1536-dim vector: one hot dimension per topic word it mentions."""
    vector = [0.0] * 1536
    vector[0] = 0.01  # never all zeros (cosine distance needs a direction)
    for word in TOPICS:
        if word in text_.lower():
            vector[1 + int(hashlib.sha256(word.encode()).hexdigest(), 16) % 1500] = 1.0
    return vector


def fake_embed(texts):
    return [fake_vector(t) for t in texts]


class FakeEmbeddings:
    async def aembed_query(self, text_):
        return fake_vector(text_)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    get_session_factory.cache_clear()
    factory = init_db()
    yield factory
    get_session_factory.cache_clear()


@pytest.fixture
def collection(tmp_path, database):
    """A uniquely named collection, removed again afterwards."""
    name = f"test_{uuid.uuid4().hex[:8]}"
    cfg = CollectionConfig(name=name, label="Test", raw_path=tmp_path / name,
                           url_pattern=f"https://www.churchofjesuschrist.org/{name}/{{slug}}", chunk_size=200, chunk_overlap=20)
    cfg.raw_path.mkdir()
    for slug, body in TOPICS.items():
        (cfg.raw_path / f"{slug}.txt").write_text(f"{slug.title()}\nSource: https://www.churchofjesuschrist.org/{name}/{slug}\n\n{body}\n")
    yield cfg
    with database() as db:
        db.execute(text("DELETE FROM rag_documents WHERE collection = :c"), {"c": name})
        db.commit()


def ingest(cfg, store):
    config = IngestConfig(collections=[cfg], embedding_model="fake", tokenizer="cl100k_base", usd_per_million_tokens=0.02)
    return run_ingestion(config, load_documents(cfg), store, fake_embed, dry_run=False)


@pytest.fixture(autouse=True)
def offline_tokens(monkeypatch):
    import src.backend.ingestion.pipeline as pipeline

    monkeypatch.setattr(pipeline, "count_tokens", lambda texts, tokenizer: sum(len(t.split()) for t in texts))


def test_init_db_creates_the_passage_store(database):
    engine = database.kw["bind"]
    tables = set(inspect(engine).get_table_names())
    assert {"rag_documents", "rag_chunks"} <= tables
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")).scalar() == 1
        assert connection.execute(text("SELECT count(*) FROM pg_proc WHERE proname = 'hybrid_search'")).scalar() == 1
        rls = dict(connection.execute(text(
            "SELECT relname, relrowsecurity FROM pg_class WHERE relname IN ('rag_documents', 'rag_chunks')"
        )).all())
    assert rls == {"rag_documents": True, "rag_chunks": True}
    # Re-running is safe (init_db does it on every startup).
    get_session_factory.cache_clear()
    assert init_db() is not None


@pytest.mark.anyio
async def test_ingest_then_hybrid_search_finds_the_right_page(database, collection):
    report = ingest(collection, SqlRagStore(database))
    assert (report.new, report.written) == (3, True)

    retriever = PgvectorRetriever(SqlTransport(database), FakeEmbeddings(), k=4)
    rows = await retriever.search_rows("What is tithing?")
    mine = [r for r in rows if r["collection"] == collection.name]
    assert mine[0]["slug"] == "tithing"
    assert mine[0]["title"] == "Tithing"
    assert mine[0]["url"].endswith("/tithing")
    assert mine[0]["semantic_rank"] == 1 and mine[0]["full_text_rank"] == 1  # found by both searches

    docs = await retriever.ainvoke("What is tithing?")
    assert docs[0].metadata["topic"] == "tithing"


@pytest.mark.anyio
async def test_full_text_alone_finds_a_page_by_its_words(database, collection):
    ingest(collection, SqlRagStore(database))
    transport = SqlTransport(database)
    # A query vector pointing nowhere in particular, so only full text decides.
    rows = await transport.search([0.0] * 1535 + [1.0], full_text_query("immersion ordinance"), 4)
    mine = [r for r in rows if r["collection"] == collection.name and r["full_text_rank"]]
    assert mine and mine[0]["slug"] == "baptism"


def test_incremental_runs_and_deletions(database, collection):
    store = SqlRagStore(database)
    ingest(collection, store)

    again = ingest(collection, store)
    assert (again.unchanged, again.chunks_to_embed) == (3, 0)

    (collection.raw_path / "faith.txt").write_text("Faith\nSource: https://www.churchofjesuschrist.org/x/faith\n\nFaith, edited.\n")
    (collection.raw_path / "baptism.txt").unlink()
    third = ingest(collection, store)
    assert (third.changed, third.removed, third.unchanged) == (1, 1, 1)

    with database() as db:
        slugs = db.execute(text("SELECT slug FROM rag_documents WHERE collection = :c ORDER BY slug"),
                           {"c": collection.name}).scalars().all()
        orphan_chunks = db.execute(text(
            "SELECT count(*) FROM rag_chunks c LEFT JOIN rag_documents d ON d.id = c.document_id WHERE d.id IS NULL"
        )).scalar()
        faith_text = db.execute(text(
            "SELECT string_agg(c.content, ' ') FROM rag_chunks c JOIN rag_documents d ON d.id = c.document_id "
            "WHERE d.collection = :c AND d.slug = 'faith'"
        ), {"c": collection.name}).scalar()
    assert slugs == ["faith", "tithing"]
    assert orphan_chunks == 0  # deleting a document removed its passages
    assert "edited" in faith_text


def test_real_startup_with_the_pgvector_backend(monkeypatch, real_startup, database, collection):
    """The app starts with RETRIEVER_BACKEND=pgvector and /health reports it."""
    ingest(collection, SqlRagStore(database))
    monkeypatch.setenv("RETRIEVER_BACKEND", "pgvector")
    monkeypatch.setenv("APP_API_KEY", "test-key")
    get_session_factory.cache_clear()

    with TestClient(app) as client:
        health = client.get("/health").json()
    with database() as db:
        total = db.execute(text("SELECT count(*) FROM rag_chunks")).scalar()

    assert health["indexes_loaded"] is True
    assert health["db"] == "ok"
    assert health["retriever"] == {"backend": "pgvector", "chunks": total}
    assert total >= 3


@pytest.mark.anyio
async def test_match_chunks_returns_semantic_hits_and_candidate_similarities(database, collection):
    ingest(collection, SqlRagStore(database))
    transport = SqlTransport(database)
    passages = [p for p in transport.load_passages() if p["collection"] == collection.name]
    baptism = next(p for p in passages if p["slug"] == "baptism")
    rows = await transport.match(fake_vector("tithing"), 4, [baptism["chunk_id"]])
    by_id = {r["chunk_id"]: r for r in rows}
    semantic = sorted((r for r in rows if r["semantic_rank"]), key=lambda r: r["semantic_rank"])
    assert semantic[0]["slug"] == "tithing"
    assert by_id[baptism["chunk_id"]]["similarity"] < semantic[0]["similarity"]


@pytest.mark.anyio
async def test_match_chunks_filtered_to_collections_with_an_exact_scan(database, collection):
    """The grouped retriever's filtered search: SET LOCAL hnsw.iterative_scan and
    ef_search are accepted, and only the given collections come back."""
    ingest(collection, SqlRagStore(database))
    transport = SqlTransport(database)
    rows = await transport.match(fake_vector("tithing"), 4, [], collections=[collection.name])
    assert rows and {r["collection"] for r in rows} == {collection.name}
    assert sorted((r for r in rows if r["semantic_rank"]), key=lambda r: r["semantic_rank"])[0]["slug"] == "tithing"
    assert await transport.match(fake_vector("tithing"), 4, [], collections=["no_such_collection"]) == []
    passages = list(transport.iter_passages(collections=[collection.name]))
    assert passages and len(passages) == sum(1 for p in transport.load_passages() if p["collection"] == collection.name)


def test_real_startup_with_the_pgvector_bm25_backend(monkeypatch, real_startup, database, collection):
    ingest(collection, SqlRagStore(database))
    monkeypatch.setenv("RETRIEVER_BACKEND", "pgvector_bm25")
    # The test collection isn't in data/collections.yaml's groups: search everything as one.
    monkeypatch.setattr(backend, "collection_groups", lambda: ((), ()))
    monkeypatch.setenv("APP_API_KEY", "test-key")
    get_session_factory.cache_clear()

    with TestClient(app) as client:
        health = client.get("/health").json()
        keyword_ids = app.state.retriever.keyword.search_ids("immersion ordinance")
    with database() as db:
        total = db.execute(text("SELECT count(*) FROM rag_chunks")).scalar()

    assert {k: health["retriever"][k] for k in ("backend", "chunks")} == {"backend": "pgvector_bm25", "chunks": total}
    assert health["retriever"]["keyword_index"] == {"rebuilding": False, "refreshed_at": None}
    with database() as db:
        topics = db.execute(text("SELECT d.slug FROM rag_chunks c JOIN rag_documents d ON d.id = c.document_id "
                                 "WHERE c.id = ANY(:ids)"), {"ids": keyword_ids}).scalars().all()
    assert "baptism" in topics


def test_copy_batches_store_the_same_vectors_as_halfvec(database, collection):
    ingest(collection, SqlRagStore(database))
    with database() as db:
        stored = db.execute(text(
            "SELECT c.embedding::text FROM rag_chunks c JOIN rag_documents d ON d.id = c.document_id "
            "WHERE d.collection = :c AND d.slug = 'tithing'"
        ), {"c": collection.name}).scalar()
        sha = db.execute(text("SELECT content_sha256 FROM rag_documents WHERE collection = :c AND slug = 'tithing'"),
                         {"c": collection.name}).scalar()
    values = [float(x) for x in stored.strip("[]").split(",")]
    # What float16 keeps of each value (0.01 isn't exact in float16).
    assert np.allclose(values, np.asarray(fake_vector("tithing"), dtype=np.float16), rtol=1e-3, atol=0)
    assert len(sha) == 64


def test_signature_moves_on_every_kind_of_change(database, collection):
    store = SqlRagStore(database)
    ingest(collection, store)
    transport = SqlTransport(database)
    before = transport.signature()

    (collection.raw_path / "faith.txt").write_text("Faith\nSource: https://www.churchofjesuschrist.org/x/faith\n\nFaith, edited.\n")
    ingest(collection, store)  # same passage count, new ids
    after_edit = transport.signature()
    assert after_edit != before and after_edit[1] > before[1]

    with database() as db:  # a title-only change: the trigger bumps updated_at
        db.execute(text("UPDATE rag_documents SET title = 'Faith!' WHERE collection = :c AND slug = 'faith'"),
                   {"c": collection.name})
        db.commit()
    assert transport.signature()[2] > after_edit[2]
