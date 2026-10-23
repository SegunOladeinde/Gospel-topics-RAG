"""Tests for GroupedRetriever: core collections searched on their own, a few
study passages (Institute manuals, Church History Topics) added after them.

No database: a fake transport answers match() from in-memory passages, with
the similarities given per test. The SQL side of the collection filter is in
test_sql_transport_filters_by_collection_with_an_exact_scan.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.embeddings import Embeddings

import src.backend.main as backend
from src.backend.ingestion.config import load_config
from src.backend.retrieval import GroupedRetriever, PgvectorBm25Retriever, SqlTransport

CORE = ("topical_guide", "topics_and_questions")
STUDY = ("institute_bom", "church_history_topics")


def row(chunk_id, content, collection, slug):
    return {"chunk_id": chunk_id, "content": content, "collection": collection, "slug": slug,
            "title": slug.title(), "url": f"https://www.churchofjesuschrist.org/{collection}/{slug}"}


PASSAGES = [
    row(1, "Faith is a hope in things not seen which are true.", "topical_guide", "faith"),
    row(2, "Faith in Jesus Christ leads to repentance.", "topics_and_questions", "faith-in-jesus-christ"),
    row(3, "Alma 32 compares the word to a seed: faith grows as we nourish it.", "institute_bom", "alma-32--seed"),
    row(4, "Alma 32:21. Faith is not a perfect knowledge.", "institute_bom", "alma-32--faith-and-hope"),
    row(5, "Liberty Jail held Joseph Smith through the winter of 1838.", "church_history_topics", "liberty-jail"),
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        self.calls += 1
        return [0.0]

    async def aembed_query(self, text):
        return self.embed_query(text)


class Transport:
    """Semantic hits: the filtered passages in id order, with the given similarities."""

    def __init__(self, similarity):
        self.similarity = similarity
        self.matched = []

    async def match(self, embedding, k, candidate_ids, collections=None):
        self.matched.append(collections)
        wanted = [p for p in PASSAGES if not collections or p["collection"] in collections]
        semantic = sorted(wanted, key=lambda p: -self.similarity[p["chunk_id"]])[:k]
        rows = [{**p, "similarity": self.similarity[p["chunk_id"]], "semantic_rank": rank}
                for rank, p in enumerate(semantic, start=1)]
        seen = {p["chunk_id"] for p in semantic}
        rows += [{**p, "similarity": self.similarity[p["chunk_id"]], "semantic_rank": None}
                 for p in wanted if p["chunk_id"] in candidate_ids and p["chunk_id"] not in seen]
        return rows

    def load_passages(self):
        return PASSAGES

    def count_chunks(self):
        return len(PASSAGES)


def grouped(similarity, max_added=2, min_similarity=0.60, relative=False, k=2):
    transport, embeddings = Transport(similarity), CountingEmbeddings()
    make = lambda cols: PgvectorBm25Retriever(transport, embeddings, k=k, min_similarity=0.30, collections=cols)  # noqa: E731
    return GroupedRetriever(make(CORE), make(STUDY), max_added, min_similarity, relative), transport, embeddings


def ids(docs):
    return [d.metadata["chunk_id"] for d in docs]


def test_each_group_indexes_only_its_own_passages():
    retriever, _, _ = grouped({i: 0.5 for i in range(1, 6)})
    assert retriever.core.chunk_counts()["keyword"] == 2
    assert retriever.added.chunk_counts()["keyword"] == 3
    assert retriever.chunk_counts() == {"vector": 5, "keyword": 5}


@pytest.mark.anyio
async def test_core_passages_come_first_and_study_ones_need_the_threshold():
    # The study passages are far more similar, but can't push out the core ones.
    retriever, transport, embeddings = grouped({1: 0.50, 2: 0.45, 3: 0.80, 4: 0.70, 5: 0.40})
    docs = await retriever.ainvoke("What does Alma 32 teach about faith?")
    assert ids(docs)[:2] == [1, 2]
    assert ids(docs)[2:] == [3, 4]  # both study passages reach 0.60; Liberty Jail (0.40) doesn't
    assert set(transport.matched) == {CORE, STUDY}  # each search filtered to its group
    assert embeddings.calls == 1  # one embedding, shared by both searches


@pytest.mark.anyio
async def test_study_passages_are_capped():
    retriever, _, _ = grouped({1: 0.5, 2: 0.5, 3: 0.9, 4: 0.8, 5: 0.7}, max_added=1)
    assert ids(await retriever.ainvoke("faith"))[2:] == [3]


@pytest.mark.anyio
async def test_relative_floor_lets_in_study_passages_closer_than_any_core_one():
    similarity = {1: 0.40, 2: 0.35, 3: 0.52, 4: 0.38, 5: 0.10}
    absolute, _, _ = grouped(similarity)
    assert ids(await absolute.ainvoke("How does the seed comparison work?"))[2:] == []
    relative, _, _ = grouped(similarity, relative=True)
    # 0.52 beats the best core passage (0.40); 0.38 doesn't.
    assert ids(await relative.ainvoke("How does the seed comparison work?"))[2:] == [3]


def test_staleness_and_rebuild_cover_both_groups():
    retriever, _, _ = grouped({i: 0.5 for i in range(1, 6)})
    retriever.core.is_stale = lambda: False
    retriever.added.is_stale = lambda: True
    assert retriever.is_stale()
    assert retriever.rebuild_keyword_index() == 5


def test_build_groups_collections_from_the_config(monkeypatch):
    monkeypatch.setattr(SqlTransport, "iter_passages", lambda self, batch=500, collections=None: (
        (p["chunk_id"], p["content"]) for p in PASSAGES if not collections or p["collection"] in collections))
    monkeypatch.setattr(SqlTransport, "signature", lambda self: (5, 5, None))
    retriever = backend.build_pgvector_bm25(SqlTransport(MagicMock()), CountingEmbeddings())
    assert isinstance(retriever, GroupedRetriever)
    assert retriever.core.collections == ("bible_dictionary", "topical_guide", "topics_and_questions")
    assert retriever.added.collections == ("institute_bom", "institute_restoration", "church_history_topics")
    assert (retriever.max_added, retriever.min_similarity, retriever.relative_to_core) == (2, 0.60, True)


def test_collection_groups_fall_back_to_one_group_when_the_config_is_unreadable(monkeypatch):
    import src.backend.ingestion.config as config

    monkeypatch.setattr(config, "load_config", MagicMock(side_effect=OSError("missing")))
    assert backend.collection_groups() == ((), ())


def test_config_rejects_an_unknown_group(tmp_path):
    path = tmp_path / "collections.yaml"
    path.write_text("collections:\n  - {name: a, label: A, raw_path: a, url_pattern: 'x/{slug}', group: extra}\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="group"):
        load_config(path, tmp_path)


class Session:
    def __init__(self, executed):
        self.executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        self.executed.append((str(statement), params))
        result = MagicMock()
        result.mappings.return_value.all.return_value = []
        return result


def test_sql_transport_filters_by_collection_with_an_exact_scan():
    executed = []
    transport = SqlTransport(lambda: Session(executed))
    transport._match([0.1], 4, [7], collections=CORE)
    assert [sql for sql, _ in executed[:2]] == [
        "SET LOCAL hnsw.iterative_scan = strict_order", "SET LOCAL hnsw.ef_search = 200"]
    assert executed[2][1]["collections"] == list(CORE)

    executed.clear()
    transport._match([0.1], 4, [7])  # unfiltered: the default HNSW settings, no filter
    assert len(executed) == 1 and executed[0][1]["collections"] is None
