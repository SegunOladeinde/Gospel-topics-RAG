"""BM25 keyword search that ignores stopwords and returns only real matches.

The pickled BM25Retriever (scripts/ingest.py) was built with LangChain's
default tokenizer, a plain str.split(): no lowercasing, punctuation left
attached, stopwords kept. Querying it the same way meant "What is
baptism?" was searched as ["What", "is", "baptism?"], and chunks matching
only "What is" (e.g. "Religion vs Violence", "Topical Guide: Mean") filled
the remaining slots, because BM25 always returns k results. BM25Okapi's IDF
floor even gives words like "the" a positive weight.

KeywordBM25Retriever wraps the same index (nothing is rebuilt) and changes
only how it's queried:
- stopwords are dropped from the question;
- each remaining word is looked up in the forms the corpus tokens actually
  take (faith, Faith, faith., faith, ...), since the corpus wasn't normalised;
- only chunks with a positive score are returned, so a question with no
  content words gets no keyword hits instead of k arbitrary ones.
"""

import re

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Common English function words plus question scaffolding ("tell me", "explain").
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most
    my myself no nor not now of off on once only or other our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves
    please tell explain describe mean means meaning know learn understand
    """.split()
)

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
# Trailing punctuation the corpus tokens most often carry.
_SUFFIXES = ("", ".", ",", "?", ";", ":")


def content_words(question: str) -> list[str]:
    """The question's words minus stopwords, de-duplicated, in order."""
    seen: dict[str, None] = {}
    for word in _WORD.findall(question):
        if word.lower() not in STOPWORDS:
            seen.setdefault(word.lower(), None)
    return list(seen)


def query_tokens(question: str) -> list[str]:
    """BM25 tokens for the question: every content word in each case/punctuation form."""
    tokens: list[str] = []
    for word in content_words(question):
        for form in dict.fromkeys((word, word.capitalize(), word.upper())):
            tokens.extend(form + suffix for suffix in _SUFFIXES)
    return tokens


class KeywordBM25Retriever(BaseRetriever):
    """Top-k BM25 matches on the question's content words (see module docstring)."""

    bm25: BM25Retriever
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        tokens = query_tokens(query)
        if not tokens:
            return []
        scores = self.bm25.vectorizer.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.bm25.docs[i] for i in ranked[: self.k] if scores[i] > 0]
