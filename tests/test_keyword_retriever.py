"""Tests for src/backend/keyword_retriever.py.

Builds a tiny BM25 index the same way scripts/ingest.py does (LangChain's
default str.split tokenizer), so the corpus has the same quirks as the real
one: case and punctuation stay attached to words.
"""

import pytest
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.backend.keyword_retriever import KeywordBM25Retriever, content_words, query_tokens

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
    return KeywordBM25Retriever(bm25=BM25Retriever.from_documents(docs), k=3)


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
    assert topics(retriever.invoke("What is baptism?"))[0] == "baptism"
    assert "filler" not in topics(retriever.invoke("What is baptism?"))


def test_finds_capitalised_and_punctuated_corpus_tokens(retriever):
    # The corpus has "Baptism" and "Tithing" capitalised, "sins." with a period.
    assert topics(retriever.invoke("what about tithing"))[0] == "tithing"
    assert topics(retriever.invoke("remission of sins?"))[0] == "baptism"


def test_only_documents_with_a_real_match_are_returned(retriever):
    # Only one document mentions the sacrament, so k=3 doesn't pad with others.
    assert topics(retriever.invoke("What is the sacrament?")) == ["sacrament"]


def test_stopword_only_question_returns_nothing(retriever):
    assert retriever.invoke("What is it?") == []


def test_unknown_word_returns_nothing(retriever):
    assert retriever.invoke("What is zyzzyva?") == []


def test_k_limits_results(retriever):
    assert len(retriever.invoke("Jesus Christ faith sacrament baptism tithing")) == 3
