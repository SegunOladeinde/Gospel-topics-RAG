"""SparseBm25 must score exactly like rank_bm25's BM25Okapi (and so rank like
KeywordSearch, which the FAISS backend uses) while holding only arrays."""

import random

import numpy as np
import pytest
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.backend.retrieval.keyword import GENERIC_VERBS, KeywordSearch, collection_phrases, query_tokens
from src.backend.retrieval.sparse_bm25 import SparseBm25

CORPUS = [
    "Faith in Jesus Christ is a principle of action and power.",
    "Faith, hope and charity. Faith. faith? FAITH",
    "Baptism by immersion for the remission of sins.",
    "Tithing is one tenth of our increase. Tithing blesses.",
    "The Ten Commandments were given to Moses on Sinai.",
    "Grace is divine help: the Atonement of Jesus Christ.",
    "Bible Dictionary introduction to this Dictionary.",
    "Lehi's dream of the tree of life. Nephi, Lehi's son.",
    "Café naïve résumé — unicode tokens sort by bytes.",
    "",  # an empty passage
    "the the the and of a",  # stopword-only: negative IDFs get the epsilon floor
]
QUESTIONS = [
    "What is faith?", "Why do we pay tithing?", "Tell me about Lehi's dream", "baptism immersion remission",
    "What does the Bible Dictionary say about grace?", "How can I increase my faith?", "café résumé",
    "the and of", "nothing matches xyzzy",
]


def sparse(corpus, **options):
    return SparseBm25.build(((100 + i, text) for i, text in enumerate(corpus)), **options)


@pytest.mark.parametrize("question", QUESTIONS)
def test_scores_equal_bm25okapi_exactly(question):
    reference = BM25Okapi([text.split() for text in CORPUS])
    tokens = query_tokens(question) or ["the", "and", "of"]
    assert np.array_equal(sparse(CORPUS).scores(tokens), reference.get_scores(tokens))


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.parametrize("filtered", [False, True])
def test_top_k_ids_equal_keyword_search(question, filtered):
    options = {"drop_phrases": collection_phrases(), "drop_words": GENERIC_VERBS} if filtered else {}
    docs = [Document(page_content=text, metadata={"chunk_id": 100 + i}) for i, text in enumerate(CORPUS)]
    reference = KeywordSearch(BM25Okapi([d.page_content.split() for d in docs]), docs, k=4, **options)
    expected = [d.metadata["chunk_id"] for d in reference.search(question)]
    assert sparse(CORPUS, k=4, **options).search_ids(question) == expected


def test_random_corpus_matches_bm25okapi():
    rng = random.Random(7)
    words = [f"w{i}" for i in range(300)] + ["Faith", "faith.", "the", "of"]
    corpus = [" ".join(rng.choices(words, k=rng.randint(0, 60))) for _ in range(400)]
    reference = BM25Okapi([text.split() for text in corpus])
    index = sparse(corpus)
    for _ in range(25):
        tokens = rng.choices(words, k=rng.randint(1, 12))
        assert np.array_equal(index.scores(tokens), reference.get_scores(tokens))


def test_holds_only_arrays_and_returns_chunk_ids():
    index = sparse(CORPUS, k=2)
    assert index.size == len(CORPUS)
    assert index.search_ids("faith") == [101, 100]
    assert index.nbytes < 10_000
    assert not hasattr(index, "docs_text")


def test_empty_store():
    index = SparseBm25.build(iter(()))
    assert index.size == 0 and index.search_ids("faith") == []


def test_real_store_matches_keyword_search():
    """All stored passages, if the local FAISS index files are present."""
    from pathlib import Path

    index_dir = Path(__file__).resolve().parent.parent / "data" / "index"
    if not (index_dir / "index.faiss").exists():
        pytest.skip("data/index not present")
    from src.backend.retrieval.index_files import load_vector_index

    _, docs = load_vector_index(index_dir)
    texts = [d.page_content for d in docs]
    options = {"drop_phrases": collection_phrases(), "drop_words": GENERIC_VERBS}
    keyed = [Document(page_content=t, metadata={"chunk_id": i}) for i, t in enumerate(texts)]
    reference = KeywordSearch(BM25Okapi([t.split() for t in texts]), keyed, k=4, **options)
    index = SparseBm25.build(enumerate(texts), k=4, **options)
    for question in QUESTIONS + ["Who was Moroni?", "Why ten percent?", "What happens after we die?"]:
        assert index.search_ids(question) == [d.metadata["chunk_id"] for d in reference.search(question)], question
