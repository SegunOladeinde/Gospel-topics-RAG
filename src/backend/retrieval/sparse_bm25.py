"""BM25 keyword search for pgvector_bm25 in a few compact arrays.

The FAISS backend's KeywordSearch wraps a rank_bm25 BM25Okapi: a dict of
term frequencies per passage, a dict of IDFs and one pair of numpy arrays per
term, plus every passage's text and metadata. For 10,160 passages that was
about 120 MB of RAM, and a rebuild briefly needed twice that.

SparseBm25 keeps only what scoring needs:
- the vocabulary as one sorted UTF-8 blob plus offsets, searched by bisection
- per term: its IDF and where its postings start (CSR, sorted by term)
- per posting: the passage index (int32) and term frequency (uint32)
- per passage: its chunk id and BM25 length norm
which is about 10 MB for today's store. Text and metadata stay in Postgres:
the search returns chunk ids, and match_chunks() (already called for the
semantic search) returns those passages' text and metadata.

Scores are computed exactly as rank_bm25's BM25Okapi (k1 1.5, b 0.75,
epsilon 0.25) does: the same IDF formula and negative-IDF floor, summed in the
same order, and the same per-term float64 arithmetic as KeywordSearch.scores,
so the rankings are identical (tests/test_sparse_bm25.py checks it against
BM25Okapi.get_scores). The index is built from a stream of (chunk id, text)
pairs, so the passages never need to be in memory all at once.
"""

import asyncio
import bisect
import math
from array import array
from collections import Counter
from collections.abc import Iterable

import numpy as np

from src.backend.retrieval.keyword import query_tokens, remove_phrases

# rank_bm25.BM25Okapi's defaults.
K1, B, EPSILON = 1.5, 0.75, 0.25


class _Terms:
    """A sorted list of byte strings stored as one blob, for bisect."""

    def __init__(self, blob: bytes, offsets: np.ndarray):
        self.blob = blob
        self.offsets = offsets

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> bytes:
        return self.blob[self.offsets[i] : self.offsets[i + 1]]

    def index(self, term: bytes) -> int | None:
        i = bisect.bisect_left(self, term)
        return i if i < len(self) and self[i] == term else None


class SparseBm25:
    def __init__(
        self,
        chunk_ids: np.ndarray,
        norm: np.ndarray,
        terms: _Terms,
        idf: np.ndarray,
        starts: np.ndarray,
        docs: np.ndarray,
        freqs: np.ndarray,
        k: int = 4,
        drop_phrases: tuple[str, ...] = (),
        drop_words: frozenset[str] = frozenset(),
    ):
        self.chunk_ids = chunk_ids
        self.norm = norm
        self.terms = terms
        self.idf = idf
        self.starts = starts
        self.docs = docs
        self.freqs = freqs
        self.k = k
        self.drop_phrases = drop_phrases
        self.drop_words = drop_words

    @classmethod
    def build(cls, passages: Iterable[tuple[int, str]], **options) -> "SparseBm25":
        """Index (chunk id, text) pairs, tokenised with str.split() like the FAISS BM25 file."""
        term_ids: dict[str, int] = {}  # in order of first appearance, like BM25Okapi's
        doc_freq = array("q")
        chunk_ids, doc_len = array("q"), array("q")
        posting_term, posting_doc, posting_freq = array("i"), array("i"), array("I")
        for doc, (chunk_id, content) in enumerate(passages):
            tokens = content.split()
            chunk_ids.append(chunk_id)
            doc_len.append(len(tokens))
            for term, freq in Counter(tokens).items():
                term_id = term_ids.get(term)
                if term_id is None:
                    term_id = term_ids[term] = len(term_ids)
                    doc_freq.append(0)
                doc_freq[term_id] += 1
                posting_term.append(term_id)
                posting_doc.append(doc)
                posting_freq.append(freq)

        corpus_size = len(chunk_ids)
        # BM25Okapi._calc_idf, term by term in the same order (the average
        # depends on the summation order).
        idf_by_id = np.empty(len(term_ids))
        idf_sum, negative = 0.0, []
        for term_id, n in enumerate(doc_freq):
            value = math.log(corpus_size - n + 0.5) - math.log(n + 0.5)
            idf_by_id[term_id] = value
            idf_sum += value
            if value < 0:
                negative.append(term_id)
        if negative:
            idf_by_id[negative] = EPSILON * (idf_sum / len(term_ids))

        lengths = np.frombuffer(doc_len, dtype=np.int64).astype(float)
        avgdl = sum(doc_len) / corpus_size if corpus_size else 0.0
        norm = K1 * (1 - B + B * lengths / avgdl) if corpus_size else lengths

        # Vocabulary sorted by its UTF-8 bytes; postings regrouped by term in
        # that order (a stable sort keeps each term's passages in index order).
        encoded = [term.encode("utf-8") for term in term_ids]
        del term_ids
        order = sorted(range(len(encoded)), key=encoded.__getitem__)
        rank = np.empty(len(order), dtype=np.int32)
        rank[order] = np.arange(len(order), dtype=np.int32)
        sorted_terms = [encoded[i] for i in order]
        del encoded
        offsets = np.zeros(len(sorted_terms) + 1, dtype=np.int64)
        np.cumsum([len(t) for t in sorted_terms], out=offsets[1:])
        blob = b"".join(sorted_terms)
        del sorted_terms

        term_of_posting = rank[np.frombuffer(posting_term, dtype=np.int32)]
        del posting_term
        by_term = np.argsort(term_of_posting, kind="stable")
        docs = np.frombuffer(posting_doc, dtype=np.int32)[by_term]
        freqs = np.frombuffer(posting_freq, dtype=np.uint32)[by_term]
        del posting_doc, posting_freq
        starts = np.searchsorted(term_of_posting[by_term], np.arange(len(order) + 1)).astype(np.int64)
        del by_term, term_of_posting

        return cls(
            chunk_ids=np.frombuffer(chunk_ids, dtype=np.int64).copy(),
            norm=norm,
            terms=_Terms(blob, offsets),
            idf=idf_by_id[order],
            starts=starts,
            docs=docs,
            freqs=freqs,
            **options,
        )

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    @property
    def nbytes(self) -> int:
        arrays = (self.chunk_ids, self.norm, self.terms.offsets, self.idf, self.starts, self.docs, self.freqs)
        return sum(a.nbytes for a in arrays) + len(self.terms.blob)

    def scores(self, tokens: list[str]) -> np.ndarray:
        """BM25Okapi.get_scores(tokens), one float per passage."""
        score = np.zeros(self.size)
        k1_plus_1 = K1 + 1
        for token in tokens:
            term = self.terms.index(token.encode("utf-8"))
            if term is None:
                continue  # unknown word: contributes 0 everywhere
            idf = self.idf[term]
            if not idf:
                continue
            start, end = self.starts[term], self.starts[term + 1]
            ids = self.docs[start:end]
            freq = self.freqs[start:end].astype(float)
            score[ids] += idf * (freq * k1_plus_1 / (freq + self.norm[ids]))
        return score

    def search_ids(self, query: str) -> list[int]:
        """Chunk ids of the top-k passages that match a content word, best first."""
        if self.drop_phrases:
            query = remove_phrases(query, self.drop_phrases)
        tokens = query_tokens(query, self.drop_words)
        if not tokens or not self.size:
            return []
        scores = self.scores(tokens)
        matching = np.flatnonzero(scores > 0)
        # Highest score first; ties keep index order, like KeywordSearch.
        ranked = sorted(matching.tolist(), key=lambda i: scores[i], reverse=True)
        return [int(self.chunk_ids[i]) for i in ranked[: self.k]]

    async def ainvoke_ids(self, query: str) -> list[int]:
        return await asyncio.to_thread(self.search_ids, query)
