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


def content_words(question: str, extra_stopwords: frozenset[str] = frozenset()) -> list[str]:
    """The question's words minus stopwords, de-duplicated, in order."""
    seen: dict[str, None] = {}
    for word in _WORD.findall(question):
        if word.lower() not in STOPWORDS and word.lower() not in extra_stopwords:
            seen.setdefault(word.lower(), None)
    return list(seen)


def query_tokens(question: str, extra_stopwords: frozenset[str] = frozenset()) -> list[str]:
    """BM25 tokens for the question: every content word in each case/punctuation form."""
    tokens: list[str] = []
    for word in content_words(question, extra_stopwords):
        for form in dict.fromkeys((word, word.capitalize(), word.upper())):
            tokens.extend(form + suffix for suffix in _SUFFIXES)
    return tokens


class KeywordSearch:
    """Top-k BM25 matches on the question's content words (see module docstring)."""

    def __init__(
        self,
        vectorizer: BM25Okapi,
        docs: list[Document],
        k: int = 4,
        drop_phrases: tuple[str, ...] = (),
        drop_words: frozenset[str] = frozenset(),
    ):
        """drop_phrases / drop_words: also left out of the query (see
        remove_phrases); the FAISS backend uses neither, pgvector_bm25 drops
        collection names and GENERIC_VERBS."""
        if len(docs) != vectorizer.corpus_size:
            raise ValueError(f"BM25 index has {vectorizer.corpus_size} documents but {len(docs)} passages")
        self.docs = docs
        self.k = k
        self.drop_phrases = drop_phrases
        self.drop_words = drop_words
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
        if self.drop_phrases:
            query = remove_phrases(query, self.drop_phrases)
        tokens = query_tokens(query, self.drop_words)
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


# Extra filtering for the Postgres full-text query (pgvector backend) and the
# BM25 query of pgvector_bm25. Naming
# a collection ("what does the Bible Dictionary say about grace") says where
# to look, not what about, and matched pages like "Bible Dictionary:
# Introduction"; generic verbs ("increase my faith") matched "Topical Guide:
# Increase". Collection names are removed as whole phrases, so a question
# about "the Bible" still searches for it.
COLLECTION_PHRASES = ("bible dictionary", "topical guide", "gospel topics", "gospel topic", "study guide")
GENERIC_VERBS = frozenset(
    """
    increase increases increasing introduce introduces introduced introduction improve improves
    help helps get gets gain gains make makes become becomes find finds use uses share shares
    want wants need needs feel feels say says said talk talks teach teaches taught
    """.split()
)
_NOT_TERM = re.compile(r"[^a-z0-9]")


def collection_phrases(names_and_labels: list[str] = ()) -> tuple[str, ...]:
    """COLLECTION_PHRASES plus each collection's name and label from
    data/collections.yaml ("church_history" -> "church history"), longest
    first so "gospel topics" goes before "gospel topic"."""
    phrases = set(COLLECTION_PHRASES)
    for value in names_and_labels:
        phrase = " ".join(value.lower().replace("_", " ").split())
        if phrase:
            phrases.add(phrase)
    return tuple(sorted(phrases, key=lambda p: (-len(p), p)))


def remove_phrases(question: str, phrases: tuple[str, ...]) -> str:
    """The question with each phrase removed as whole words, case-insensitively."""
    for phrase in phrases:
        question = re.sub(rf"\b{re.escape(phrase)}\b", " ", question, flags=re.IGNORECASE)
    return question


def full_text_query(question: str) -> str:
    """An OR query of the question's content words for to_tsquery, e.g. "grace | works".

    Postgres stems and lowercases the words itself (to_tsquery('english',
    ...)), so unlike BM25 no spelling variants are needed. Only [a-z0-9]
    survive, so the result is always valid tsquery syntax. "" if nothing's left.
    """
    text = question.lower()
    for phrase in COLLECTION_PHRASES:
        text = re.sub(rf"\b{phrase}\b", " ", text)
    terms: dict[str, None] = {}
    for word in content_words(text):
        term = _NOT_TERM.sub("", word.split("'")[0])
        if term and term not in GENERIC_VERBS and term not in STOPWORDS:
            terms.setdefault(term, None)
    return " | ".join(terms)
