"""Semantic search: embed the question, find the nearest passages in FAISS.

Does what LangChain's FAISS vector store did for us (a plain L2 search over
an un-normalised IndexFlatL2; OpenAI embeddings are already unit length),
without the langchain_community dependency.
"""

import asyncio
import math

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorSearch:
    def __init__(self, index: faiss.Index, docs: list[Document], embeddings: Embeddings, k: int = 4):
        self.index = index
        self.docs = docs
        self.embeddings = embeddings
        self.k = k

    @property
    def size(self) -> int:
        return self.index.ntotal

    @staticmethod
    def relevance(distance: float) -> float:
        """L2 distance between unit vectors (0..sqrt 2) as a 0..1 relevance score."""
        return 1.0 - distance / math.sqrt(2)

    async def search_with_distances(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        embedding = await self.embeddings.aembed_query(query)
        vector = np.asarray([embedding], dtype=np.float32)
        distances, positions = await asyncio.to_thread(self.index.search, vector, k or self.k)
        return [
            (self.docs[position], float(distance))
            for distance, position in zip(distances[0], positions[0])
            if position != -1
        ]

    async def ainvoke(self, query: str) -> list[Document]:
        return [doc for doc, _ in await self.search_with_distances(query)]
