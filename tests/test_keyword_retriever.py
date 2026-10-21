"""Tests for src/backend/retrieval/keyword.py.

Builds a tiny BM25 index the same way scripts/ingest.py does (a plain
str.split tokenizer), so the corpus has the same quirks as the real one:
case and punctuation stay attached to words.
"""

import numpy as np
import pytest
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.backend.retrieval.keyword import KeywordSearch, content_words, query_tokens

CORPUS = {
    "baptism": "Baptism by immersion is the first ordinance. We are baptized for the remission of sins.",
    "sacrament": "The sacrament is bread and water taken each week to remember Jesus Christ.",
    "filler": "What is it that we do? This is what it is and what it was.",
    "faith": "Faith in Jesus Christ is a principle of action and power.",
    "tithing": "Tithing is one tenth of our increase, paid to the Lord.",
}


@pytest.fixture(scope="module")
def retriever():
    docs = [Document(page_content=text, metadata={"topic": topic}) for topic, text in CORPUS.items()]
    return KeywordSearch(BM25Okapi([doc.page_content.split() for doc in docs]), docs, k=3)


def topics(documents):
    return [doc.metadata["topic"] for doc in documents]


def test_content_words_drop_stopwords_and_punctuation():
    assert content_words("What is baptism?") == ["baptism"]
    assert content_words("Why do we partake of the sacrament?") == ["partake", "sacrament"]
    assert content_words("Can you explain the Word of Wisdom, please?") == ["word", "wisdom"]


def test_content_words_deduplicate_case_insensitively():
    assert content_words("Faith, faith and FAITH") == ["faith"]


def test_content_words_keep_apostrophes():
    assert content_words("What is Lehi's dream?") == ["lehi's", "dream"]


def test_question_of_only_stopwords_has_no_content():
    assert content_words("What is it?") == []
    assert query_tokens("Who was he?") == []


def test_query_tokens_cover_the_corpus_spellings():
    tokens = query_tokens("baptism")
    for form in ("baptism", "Baptism", "BAPTISM", "baptism.", "Baptism,", "baptism?"):
        assert form in tokens


def test_matches_the_topic_not_the_question_words(retriever):
    # The raw index would rank "filler" first: it's full of "What", "is", "it".
    assert topics(retriever.search("What is baptism?"))[0] == "baptism"
    assert "filler" not in topics(retriever.search("What is baptism?"))


def test_finds_capitalised_and_punctuated_corpus_tokens(retriever):
    # The corpus has "Baptism" and "Tithing" capitalised, "sins." with a period.
    assert topics(retriever.search("what about tithing"))[0] == "tithing"
    assert topics(retriever.search("remission of sins?"))[0] == "baptism"


def test_only_documents_with_a_real_match_are_returned(retriever):
    # Only one document mentions the sacrament, so k=3 doesn't pad with others.
    assert topics(retriever.search("What is the sacrament?")) == ["sacrament"]


def test_stopword_only_question_returns_nothing(retriever):
    assert retriever.search("What is it?") == []


def test_unknown_word_returns_nothing(retriever):
    assert retriever.search("What is zyzzyva?") == []


def test_k_limits_results(retriever):
    assert len(retriever.search("Jesus Christ faith sacrament baptism tithing")) == 3


@pytest.mark.parametrize(
    ("question", "words"),
    [
        ("List the steps of repentance", ["repentance"]),
        ("What is the purpose of the sacrament?", ["sacrament"]),
        ("What does baptism mean?", ["baptism"]),
        ("Explain the meaning of the Atonement", ["atonement"]),
        ("Give me examples of faith", ["faith"]),
        ("Summarize the Word of Wisdom", ["word", "wisdom"]),
        ("Tell me the importance of prayer", ["prayer"]),
        ("Define charity", ["charity"]),
    ],
)
def test_instruction_words_are_not_searched(question, words):
    """Words saying how to answer ("list", "purpose") aren't topics to match."""
    assert content_words(question) == words


def test_instruction_word_only_question_has_no_keyword_hits(retriever):
    assert retriever.search("Explain the purpose, list the steps") == []


@pytest.mark.parametrize(
    "question",
    ["What is baptism?", "Jesus Christ faith", "remission of sins?", "what about tithing", "zyzzyva", "the the the"],
)
def test_scores_match_rank_bm25_exactly(question):
    """The inverted index computes the same floats as BM25Okapi.get_scores."""
    docs = [Document(page_content=text) for text in CORPUS.values()]
    vectorizer = BM25Okapi([doc.page_content.split() for doc in docs])
    tokens = query_tokens(question) or ["the", "the", "the"]  # also check stopword-only scoring
    assert np.array_equal(KeywordSearch(vectorizer, docs).scores(tokens), vectorizer.get_scores(tokens))


def test_mismatched_passages_are_refused():
    vectorizer = BM25Okapi([["a"], ["b"]])
    with pytest.raises(ValueError):
        KeywordSearch(vectorizer, [Document(page_content="a")])
