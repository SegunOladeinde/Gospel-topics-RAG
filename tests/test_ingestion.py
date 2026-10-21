"""Tests for src/backend/ingestion: config, parsing, planning and incremental runs.

Uses an in-memory store and fake embeddings, so nothing touches a database
or OpenAI. The Postgres version (SqlRagStore + hybrid_search) is tested in
tests/test_pgvector_postgres.py.
"""

from dataclasses import replace
from pathlib import Path

import pytest

import src.backend.ingestion.pipeline as pipeline
from src.backend.ingestion.config import CollectionConfig, IngestConfig, load_config
from src.backend.ingestion.documents import load_documents, parse_document
from src.backend.ingestion.pipeline import StoredDocument, embed_with_retries, plan_run, run_ingestion

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_COUNT_TOKENS = pipeline.count_tokens


@pytest.fixture(autouse=True)
def no_tiktoken(monkeypatch):
    # Token counting is exercised separately; keep these tests offline.
    monkeypatch.setattr(pipeline, "count_tokens", lambda texts, tokenizer: sum(len(t.split()) for t in texts))


def collection(tmp_path, name="topical_guide", resolve=False, size=200, overlap=20):
    return CollectionConfig(
        name=name,
        label="Topical Guide",
        raw_path=tmp_path / name,
        url_pattern=f"https://www.churchofjesuschrist.org/{name}/{{slug}}?lang=eng",
        resolve_redirects=resolve,
        chunk_size=size,
        chunk_overlap=overlap,
    )


def write(cfg, slug, title, body, url=True):
    cfg.raw_path.mkdir(parents=True, exist_ok=True)
    source = f"Source: https://www.churchofjesuschrist.org/real/{slug}?lang=eng\n" if url else ""
    (cfg.raw_path / f"{slug}.txt").write_text(f"{title}\n{source}\n{body}\n", encoding="utf-8")


class FakeStore:
    def __init__(self):
        self.documents: dict[tuple, dict] = {}
        self.chunks: dict[int, list[dict]] = {}
        self.calls: list[tuple] = []
        self.next_id = 1

    def existing_documents(self, collections):
        return {k: StoredDocument(v["id"], v["sha256"]) for k, v in self.documents.items() if k[0] in collections}

    def upsert_documents(self, documents, sha256, chunk_counts):
        ids = {}
        for doc in documents:
            row = self.documents.get(doc.key)
            if row is None:
                row = self.documents[doc.key] = {"id": self.next_id}
                self.next_id += 1
            row.update(sha256=sha256[doc.key], title=doc.title, url=doc.url, chunks=chunk_counts[doc.key])
            ids[doc.key] = row["id"]
            self.calls.append(("upsert", doc.slug, sha256[doc.key]))
        return ids

    def replace_chunks(self, document_ids, rows):
        for doc_id in document_ids:
            self.chunks[doc_id] = [r for r in rows if r["document_id"] == doc_id]
        self.calls.append(("chunks", len(rows)))

    def delete_documents(self, document_ids):
        for key in [k for k, v in self.documents.items() if v["id"] in document_ids]:
            self.chunks.pop(self.documents.pop(key)["id"], None)
        self.calls.append(("delete", sorted(document_ids)))


class FakeEmbedder:
    def __init__(self):
        self.embedded: list[str] = []

    def __call__(self, texts):
        self.embedded.extend(texts)
        return [[0.1] * 3 for _ in texts]


def ingest_config(*collections):
    return IngestConfig(collections=list(collections), embedding_model="m", tokenizer="t", usd_per_million_tokens=0.02)


def docs_for(*collections):
    return [doc for c in collections for doc in load_documents(c)]


# ---------------------------------------------------------------------------
# Config and parsing
# ---------------------------------------------------------------------------

def test_repo_config_lists_the_three_collections():
    config = load_config(PROJECT_ROOT / "data" / "collections.yaml", PROJECT_ROOT)
    assert [c.name for c in config.collections] == ["bible_dictionary", "topical_guide", "topics_and_questions"]
    gospel_topics = config.collections[2]
    assert gospel_topics.resolve_redirects is True
    assert gospel_topics.chunk_size == 1000 and gospel_topics.chunk_overlap == 200


def test_config_rejects_a_pattern_without_slug(tmp_path):
    bad = tmp_path / "c.yaml"
    bad.write_text("collections:\n  - {name: x, label: X, raw_path: raw, url_pattern: 'https://example.org'}\n")
    with pytest.raises(ValueError, match="slug"):
        load_config(bad, tmp_path)


def test_parse_takes_the_real_title_and_url(tmp_path):
    cfg = collection(tmp_path)
    raw = b"Aaron, Descendants of\nSource: https://www.churchofjesuschrist.org/study/scriptures/tg/aaron-descendants-of?lang=eng\n\nSee also Aaron."
    doc = parse_document(cfg, "aaron-descendants-of", raw)
    assert doc.title == "Aaron, Descendants of"
    assert doc.url.endswith("/tg/aaron-descendants-of?lang=eng")
    # The passages are cut from the raw text unchanged, like the FAISS index.
    assert doc.text == raw.decode("utf-8")


def test_parse_falls_back_to_the_url_pattern_and_slug_title(tmp_path):
    cfg = replace(collection(tmp_path), title_source="slug")
    doc = parse_document(cfg, "word-of-wisdom", b"Some text without a header.")
    assert doc.title == "Word Of Wisdom"
    assert doc.url == "https://www.churchofjesuschrist.org/topical_guide/word-of-wisdom?lang=eng"


def test_sha_changes_with_chunk_settings(tmp_path):
    raw = b"Faith\nSource: https://www.churchofjesuschrist.org/x\n\nFaith is..."
    a = parse_document(collection(tmp_path, size=1000), "faith", raw)
    b = parse_document(collection(tmp_path, size=500), "faith", raw)
    assert a.sha256 != b.sha256


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def test_plan_sorts_documents_by_what_changed(tmp_path):
    cfg = collection(tmp_path)
    for slug in ("faith", "hope", "charity"):
        write(cfg, slug, slug.title(), f"About {slug}.")
    docs = {d.slug: d for d in docs_for(cfg)}
    stored = {
        docs["faith"].key: StoredDocument(1, docs["faith"].sha256),  # unchanged
        docs["hope"].key: StoredDocument(2, "old-sha"),  # changed
        ("topical_guide", "gone"): StoredDocument(3, "x"),  # removed from disk
    }
    plan = plan_run(list(docs.values()), stored)
    assert [d.slug for d in plan.new] == ["charity"]
    assert [d.slug for d in plan.changed] == ["hope"]
    assert [d.slug for d in plan.unchanged] == ["faith"]
    assert list(plan.removed) == [("topical_guide", "gone")]


def test_half_written_documents_are_redone(tmp_path):
    cfg = collection(tmp_path)
    write(cfg, "faith", "Faith", "About faith.")
    doc = docs_for(cfg)[0]
    plan = plan_run([doc], {doc.key: StoredDocument(1, "")})
    assert plan.changed == [doc]


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def test_dry_run_counts_without_writing_or_embedding(tmp_path):
    cfg = collection(tmp_path)
    write(cfg, "faith", "Faith", "word " * 300)
    store, embed = FakeStore(), FakeEmbedder()
    report = run_ingestion(ingest_config(cfg), docs_for(cfg), store, embed, dry_run=True)
    assert report.new == 1 and report.chunks_to_embed > 1
    assert report.tokens_to_embed > 0
    assert report.estimated_usd == pytest.approx(report.tokens_to_embed / 1_000_000 * 0.02)
    assert store.calls == [] and embed.embedded == []


def test_first_run_then_incremental_runs(tmp_path):
    cfg = collection(tmp_path)
    for slug in ("faith", "hope", "charity"):
        write(cfg, slug, slug.title(), f"About {slug}. " * 30)
    store, embed = FakeStore(), FakeEmbedder()
    first = run_ingestion(ingest_config(cfg), docs_for(cfg), store, embed, dry_run=False)
    assert first.new == 3 and first.written
    assert len(store.documents) == 3
    assert all(row["sha256"] for row in store.documents.values())
    first_embedded = len(embed.embedded)

    # Nothing changed: nothing embedded, nothing written.
    store.calls.clear()
    again = run_ingestion(ingest_config(cfg), docs_for(cfg), store, embed, dry_run=False)
    assert (again.unchanged, again.chunks_to_embed) == (3, 0)
    assert len(embed.embedded) == first_embedded
    assert store.calls == []

    # One page edited, one deleted: only the edit is re-embedded.
    write(cfg, "hope", "Hope", "A brand new text about hope. " * 30)
    (cfg.raw_path / "charity.txt").unlink()
    third = run_ingestion(ingest_config(cfg), docs_for(cfg), store, embed, dry_run=False)
    assert (third.changed, third.removed, third.unchanged) == (1, 1, 1)
    assert set(k[1] for k in store.documents) == {"faith", "hope"}
    assert all("hope" in text.lower() for text in embed.embedded[first_embedded:])


def test_writes_sha_only_after_passages(tmp_path):
    """A crash mid-run must leave the document marked as needing work."""
    cfg = collection(tmp_path)
    write(cfg, "faith", "Faith", "About faith.")
    store = FakeStore()
    run_ingestion(ingest_config(cfg), docs_for(cfg), store, FakeEmbedder(), dry_run=False)
    doc = docs_for(cfg)[0]
    assert store.calls == [("upsert", "faith", ""), ("chunks", 1), ("upsert", "faith", doc.sha256)]


def test_passage_rows_carry_collection_and_order(tmp_path):
    cfg = collection(tmp_path)
    write(cfg, "faith", "Faith", "word " * 300)
    store = FakeStore()
    run_ingestion(ingest_config(cfg), docs_for(cfg), store, FakeEmbedder(), dry_run=False)
    rows = next(iter(store.chunks.values()))
    assert [r["chunk_index"] for r in rows] == list(range(len(rows)))
    assert {r["collection"] for r in rows} == {"topical_guide"}
    assert rows[0]["content"].startswith("Faith\nSource:")


def test_redirects_are_resolved_only_where_configured(tmp_path):
    resolving = collection(tmp_path, name="topics_and_questions", resolve=True)
    plain = collection(tmp_path, name="topical_guide")
    write(resolving, "priest", "Aaronic Priesthood", "Text.")
    write(plain, "faith", "Faith", "Text.")
    store = FakeStore()
    asked = []

    def resolve(doc):
        asked.append(doc.slug)
        return "https://www.churchofjesuschrist.org/moved/aaronic-priesthood?lang=eng"

    run_ingestion(ingest_config(resolving, plain), docs_for(resolving, plain), store, FakeEmbedder(),
                  dry_run=False, resolve_url=resolve)
    assert asked == ["priest"]
    assert store.documents[("topics_and_questions", "priest")]["url"].endswith("/moved/aaronic-priesthood?lang=eng")
    assert store.documents[("topical_guide", "faith")]["url"].endswith("/real/faith?lang=eng")


def test_real_run_needs_an_embedder(tmp_path):
    cfg = collection(tmp_path)
    write(cfg, "faith", "Faith", "Text.")
    with pytest.raises(ValueError):
        run_ingestion(ingest_config(cfg), docs_for(cfg), FakeStore(), None, dry_run=False)


# ---------------------------------------------------------------------------
# Embedding retries
# ---------------------------------------------------------------------------

def test_embedding_batches_and_retries_with_backoff():
    calls, sleeps = [], []

    def flaky(texts):
        calls.append(len(texts))
        if len(calls) == 2:
            raise RuntimeError("429 rate limited")
        return [[1.0] for _ in texts]

    vectors = embed_with_retries([f"t{i}" for i in range(5)], flaky, batch_size=2, sleep=sleeps.append)
    assert len(vectors) == 5
    assert calls == [2, 2, 2, 1]  # the second batch failed once and was retried
    assert sleeps == [2]


def test_embedding_gives_up_after_max_attempts():
    def broken(texts):
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        embed_with_retries(["a"], broken, max_attempts=3, sleep=lambda s: None)


def test_count_tokens_uses_the_tokenizer():
    try:
        assert REAL_COUNT_TOKENS(["hello world"], "cl100k_base") == 2
    except Exception as exc:  # noqa: BLE001 - tiktoken downloads its table on first use
        pytest.skip(f"tiktoken tokenizer unavailable offline: {exc}")


def test_windows_line_endings_are_normalised_like_read_text(tmp_path):
    """The FAISS index was built from read_text (universal newlines); match it."""
    cfg = collection(tmp_path)
    unix = parse_document(cfg, "faith", b"Faith\nSource: https://www.churchofjesuschrist.org/x\n\nText.")
    windows = parse_document(cfg, "faith", b"Faith\r\nSource: https://www.churchofjesuschrist.org/x\r\n\r\nText.")
    assert windows.text == unix.text
    assert windows.url == unix.url


def test_rest_store_retries_dropped_connections(monkeypatch):
    import httpx

    from src.backend.ingestion import stores

    monkeypatch.setattr(stores.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        if calls["n"] == 2:
            return httpx.Response(503)
        return httpx.Response(200, json=[])

    store = stores.RestRagStore("https://example.supabase.co", "k",
                                client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert store.existing_documents(["topical_guide"]) == {}
    assert calls["n"] == 3


def test_rest_store_does_not_retry_client_errors(monkeypatch):
    import httpx

    from src.backend.ingestion import stores

    monkeypatch.setattr(stores.time, "sleep", lambda s: None)
    store = stores.RestRagStore("https://example.supabase.co", "k",
                                client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400))))
    with pytest.raises(httpx.HTTPStatusError):
        store.existing_documents(["topical_guide"])


def test_max_documents_caps_a_run_and_the_next_run_continues(tmp_path):
    cfg = collection(tmp_path)
    for slug in ("a", "b", "c"):
        write(cfg, slug, slug.upper(), f"About {slug}.")
    store = FakeStore()
    first = run_ingestion(ingest_config(cfg), docs_for(cfg), store, FakeEmbedder(), dry_run=False, max_documents=2)
    assert first.new == 2 and len(store.documents) == 2
    second = run_ingestion(ingest_config(cfg), docs_for(cfg), store, FakeEmbedder(), dry_run=False, max_documents=2)
    assert (second.new, second.unchanged) == (1, 2) and len(store.documents) == 3
