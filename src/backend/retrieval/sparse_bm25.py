"""BM25 keyword search for pgvector_bm25 in a few compact arrays.

The FAISS backend's KeywordSearch wraps a rank_bm25 BM25Okapi: a dict of
term frequencies per passage, a dict of IDFs and one pair of numpy arrays per
term, plus every passage's text and metadata. For 10,160 passages that was
about 120 MB of RAM, and a rebuild briefly needed twice that.

SparseBm25 keeps only what scoring needs, about 6 MB for today's store:
- per posting (one per distinct word in a passage): the word's id (int32)
  and how often it occurs there (uint16), in passage order
- per passage: its chunk id, where its postings start, its BM25 length norm
- per word: its IDF, and the vocabulary as one sorted UTF-8 blob (searched
  by bisection) mapped back to word ids
Text and metadata stay in Postgres: the search returns chunk ids, and
match_chunks() (already called for the semantic search) returns those
passages' text and metadata.

Postings stay in the order they were read, so building needs no sort and no
second copy: the arrays filled while streaming the passages (array.array)
are the index, viewed through numpy. A query word's postings are found with
one vectorized comparison over the postings (well under a millisecond).

Scores are computed exactly as rank_bm25's BM25Okapi (k1 1.5, b 0.75,
epsilon 0.25) does: the same IDF formula and negative-IDF floor, summed in the
same order, and the same per-word float64 arithmetic as KeywordSearch.scores,
so the rankings are identical (tests/test_sparse_bm25.py checks it against
BM25Okapi.get_scores).
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
_MAX_FREQ = 2**16 - 1  # uint16; a 1,000-character passage can't get near it


class _Terms:
    """The vocabulary sorted by UTF-8 bytes, stored as one blob, for bisect."""

    def __init__(self, blob: bytes, offsets: np.ndarray, ids: np.ndarray):
        self.blob = blob
        self.offsets = offsets
        self.ids = ids  # word id of each sorted entry

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> bytes:
        return self.blob[self.offsets[i] : self.offsets[i + 1]]

    def id_of(self, term: str) -> int | None:
        key = term.encode("utf-8")
        i = bisect.bisect_left(self, key)
        return int(self.ids[i]) if i < len(self) and self[i] == key else None

    @property
    def nbytes(self) -> int:
        return len(self.blob) + self.offsets.nbytes + self.ids.nbytes


class SparseBm25:
    def __init__(
        self,
        chunk_ids: np.ndarray,
        starts: np.ndarray,
        norm: np.ndarray,
        terms: _Terms,
        idf: np.ndarray,
        posting_terms: np.ndarray,
        posting_freqs: np.ndarray,
        k: int = 4,
        drop_phrases: tuple[str, ...] = (),
        drop_words: frozenset[str] = frozenset(),
    ):
        self.chunk_ids = chunk_ids
        self.starts = starts
        self.norm = norm
        self.terms = terms
        self.idf = idf
        self.posting_terms = posting_terms
        self.posting_freqs = posting_freqs
        self.k = k
        self.drop_phrases = drop_phrases
        self.drop_words = drop_words

    @classmethod
    def build(cls, passages: Iterable[tuple[int, str]], **options) -> "SparseBm25":
        """Index (chunk id, text) pairs, tokenised with str.split() like the FAISS BM25 file."""
        term_ids: dict[str, int] = {}  # in order of first appearance, like BM25Okapi's
        doc_freq = array("q")
        chunk_ids, starts, doc_len = array("q"), array("q", [0]), array("q")
        posting_terms, posting_freqs = array("i"), array("H")
        for chunk_id, content in passages:
            tokens = content.split()
            chunk_ids.append(chunk_id)
            doc_len.append(len(tokens))
            for term, freq in Counter(tokens).items():
                term_id = term_ids.get(term)
                if term_id is None:
                    term_id = term_ids[term] = len(term_ids)
                    doc_freq.append(0)
                doc_freq[term_id] += 1
                posting_terms.append(term_id)
                posting_freqs.append(min(freq, _MAX_FREQ))
            starts.append(len(posting_terms))

        corpus_size = len(chunk_ids)
        # BM25Okapi._calc_idf, word by word in the same order (the average
        # depends on the summation order).
        idf = np.empty(len(term_ids))
        idf_sum, negative = 0.0, []
        for term_id, n in enumerate(doc_freq):
            value = math.log(corpus_size - n + 0.5) - math.log(n + 0.5)
            idf[term_id] = value
            idf_sum += value
            if value < 0:
                negative.append(term_id)
        if negative:
            idf[negative] = EPSILON * (idf_sum / len(term_ids))
        del doc_freq

        lengths = np.frombuffer(doc_len, dtype=np.int64).astype(float) if corpus_size else np.zeros(0)
        avgdl = sum(doc_len) / corpus_size if corpus_size else 0.0
        norm = K1 * (1 - B + B * lengths / avgdl) if corpus_size else lengths

        # The vocabulary as one sorted blob; the dict goes once it's built.
        ordered = sorted(term_ids, key=lambda term: term.encode("utf-8"))
        ids = np.fromiter((term_ids[term] for term in ordered), dtype=np.int32, count=len(ordered))
        del term_ids
        encoded = [term.encode("utf-8") for term in ordered]
        del ordered
        offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
        if encoded:
            np.cumsum(np.fromiter(map(len, encoded), dtype=np.int64, count=len(encoded)), out=offsets[1:])
        blob = b"".join(encoded)
        del encoded

        def view(buffer: array, dtype) -> np.ndarray:
            # Zero-copy over the array the build filled (it stays alive through the view).
            return np.frombuffer(buffer, dtype=dtype) if len(buffer) else np.zeros(0, dtype=dtype)

        return cls(
            chunk_ids=view(chunk_ids, np.int64),
            starts=view(starts, np.int64),
            norm=norm,
            terms=_Terms(blob, offsets, ids),
            idf=idf,
            posting_terms=view(posting_terms, np.int32),
            posting_freqs=view(posting_freqs, np.uint16),
            **options,
        )

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    @property
    def nbytes(self) -> int:
        arrays = (self.chunk_ids, self.starts, self.norm, self.idf, self.posting_terms, self.posting_freqs)
        return sum(a.nbytes for a in arrays) + self.terms.nbytes

    def scores(self, tokens: list[str]) -> np.ndarray:
        """BM25Okapi.get_scores(tokens), one float per passage."""
        score = np.zeros(self.size)
        k1_plus_1 = K1 + 1
        for token in tokens:
            term = self.terms.id_of(token)
            if term is None:
                continue  # unknown word: contributes 0 everywhere
            idf = self.idf[term]
            if not idf:
                continue
            positions = np.flatnonzero(self.posting_terms == term)  # ascending, so passages in order
            ids = np.searchsorted(self.starts, positions, side="right") - 1
            freq = self.posting_freqs[positions].astype(float)
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
