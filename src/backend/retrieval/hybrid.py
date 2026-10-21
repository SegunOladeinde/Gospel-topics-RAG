"""Combine semantic and keyword search with weighted reciprocal rank fusion.

Each passage scores sum(weight / (rank + c)) over the lists it appears in,
deduplicated by its text, highest first. This is exactly what LangChain's
EnsembleRetriever did (same formula, c=60, same tie-breaking), so swapping
it out didn't change which sources come back.
"""

import asyncio
from collections import defaultdict
from typing import Protocol

from langchain_core.documents import Document


class _Search(Protocol):
    size: int

    async def ainvoke(self, query: str) -> list[Document]: ...


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], weights: list[float], c: int = 60
) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    for docs, weight in zip(ranked_lists, weights, strict=True):
        for rank, doc in enumerate(docs, start=1):
            scores[doc.page_content] += weight / (rank + c)
    unique: dict[str, Document] = {}
    for docs in ranked_lists:
        for doc in docs:
            unique.setdefault(doc.page_content, doc)
    # sorted() is stable: equal scores keep first-seen order.
    return sorted(unique.values(), key=lambda doc: scores[doc.page_content], reverse=True)


class HybridRetriever:
    """Runs semantic and keyword search together and fuses the rankings."""

    def __init__(self, vector: _Search, keyword: _Search, weights: tuple[float, float] = (0.5, 0.5), c: int = 60):
        self.vector = vector
        self.keyword = keyword
        self.weights = list(weights)
        self.c = c

    async def ainvoke(self, query: str) -> list[Document]:
        vector_docs, keyword_docs = await asyncio.gather(self.vector.ainvoke(query), self.keyword.ainvoke(query))
        return reciprocal_rank_fusion([vector_docs, keyword_docs], self.weights, self.c)

    def chunk_counts(self) -> dict[str, int]:
        return {"vector": self.vector.size, "keyword": self.keyword.size}
