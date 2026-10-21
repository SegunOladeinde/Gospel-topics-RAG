"""BM25 keyword search that ignores stopwords and returns only real matches.

The BM25 index (data/index/bm25.pkl) was built with a plain str.split()
tokenizer: no lowercasing, punctuation left attached, stopwords kept.
Querying it the same way meant "What is baptism?" was searched as
["What", "is", "baptism?"], and chunks matching only "What is" (e.g.
"Religion vs Violence", "Topical Guide: Mean") filled the remaining slots,
because BM25 always returns k results. BM25Okapi's IDF floor even gives
words like "the" a positive weight.

So KeywordSearch changes only how the index is queried:
- stopwords and instruction words are dropped from the question;
- each remaining word is looked up in the forms the corpus tokens actually
  take (faith, Faith, faith., faith, ...), since the corpus wasn't normalised;
- only chunks with a positive score are returned, so a question with no
  content words gets no keyword hits instead of k arbitrary ones.

Scoring uses an inverted index built once at startup, so each query term
only touches the chunks that contain it. It computes exactly what
rank_bm25's BM25Okapi.get_scores does (same terms, same order, so the same
floating-point sums) without looping over all 10,000+ chunks per term; a
long question with many words went from seconds to milliseconds.
"""

import asyncio
import re
from collections import defaultdict

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Common English function words.
FUNCTION_WORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most
    my myself no nor not now of off on once only or other our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves
    """.split()
)

# Words that say how to answer, not what about. Several are also Topical
# Guide entries ("List", "Purpose", "Mean"), so leaving them in made keyword
# search cite those entries: "List the steps of repentance" returned
# "Topical Guide: List". The embedding search still sees the whole question.
INSTRUCTION_WORDS = frozenset(
    """
    please tell explain explanation describe description define definition summarize summarise
    summary overview outline list lists step steps purpose purposes mean means meaning meanings
    give show example examples importance important significance way ways kind kinds type types
    main key basic simple simply briefly detail details elaborate
    know learn understand
    """.split()
)

STOPWORDS = FUNCTION_WORDS | INSTRUCTION_WORDS

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


class KeywordSearch:
    """Top-k BM25 matches on the question's content words (see module docstring)."""

    def __init__(self, vectorizer: BM25Okapi, docs: list[Document], k: int = 4):
        if len(docs) != vectorizer.corpus_size:
            raise ValueError(f"BM25 index has {vectorizer.corpus_size} documents but {len(docs)} passages")
        self.docs = docs
        self.k = k
        self._idf = vectorizer.idf
        doc_len = np.asarray(vectorizer.doc_len, dtype=float)
        # The per-document part of BM25's denominator, and the numerator factor.
        self._norm = vectorizer.k1 * (1 - vectorizer.b + vectorizer.b * doc_len / vectorizer.avgdl)
        self._k1_plus_1 = vectorizer.k1 + 1
        postings: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for doc_id, frequencies in enumerate(vectorizer.doc_freqs):
            for term, count in frequencies.items():
                ids, counts = postings[term]
                ids.append(doc_id)
                counts.append(count)
        self._postings = {
            term: (np.asarray(ids), np.asarray(counts, dtype=float)) for term, (ids, counts) in postings.items()
        }

    @property
    def size(self) -> int:
        return len(self.docs)

    def scores(self, tokens: list[str]) -> np.ndarray:
        """BM25Okapi.get_scores(tokens), computed from the inverted index."""
        score = np.zeros(len(self.docs))
        for token in tokens:
            idf = self._idf.get(token) or 0
            posting = self._postings.get(token)
            if not idf or posting is None:
                continue  # contributes 0 to every document
            ids, freq = posting
            score[ids] += idf * (freq * self._k1_plus_1 / (freq + self._norm[ids]))
        return score

    def search(self, query: str) -> list[Document]:
        tokens = query_tokens(query)
        if not tokens:
            return []
        scores = self.scores(tokens)
        matching = np.flatnonzero(scores > 0)
        # Highest score first; ties keep index order, like a stable sort.
        ranked = sorted(matching.tolist(), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in ranked[: self.k]]

    async def ainvoke(self, query: str) -> list[Document]:
        # CPU work; keep it off the event loop.
        return await asyncio.to_thread(self.search, query)
