"""Build the local FAISS + BM25 index files (RETRIEVER_BACKEND=faiss).

The pgvector store is filled by scripts/ingest.py instead; this script and
data/index/ go away once the pgvector backend has run stably (see README).

Chunks the scraped doctrinal text and builds the retrieval indexes.

Reads every .txt file under data/raw/ (recursively), splits each into
overlapping chunks, embeds them with OpenAI, and writes two indexes to
data/index/:
- index.faiss + index.pkl: the vectors and their passages (semantic search)
- bm25.pkl: a BM25 index over the same passages (keyword search)

Both are loaded together by src/backend/retrieval at startup (see
CLAUDE.md), so they're built from the exact same chunks/metadata here.
After rebuilding, update BM25_INDEX_SHA256 on Render (see README).
"""

import logging
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backend.retrieval.index_files import save_keyword_index, save_vector_index  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_source_documents() -> tuple[list[str], list[dict]]:
    """Return (texts, metadatas) for every .txt file under data/raw/."""
    texts = []
    metadatas = []
    for path in sorted(RAW_DIR.rglob("*.txt")):
        texts.append(path.read_text(encoding="utf-8"))
        metadatas.append({"source": path.parent.name, "topic": path.stem})
    return texts, metadatas


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Check your .env file.")

    logger.info("Reading .txt files from %s ...", RAW_DIR)
    texts, metadatas = load_source_documents()
    logger.info("  %d source documents found", len(texts))

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.create_documents(texts, metadatas=metadatas)
    logger.info("  split into %d chunks", len(chunks))

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Embedding chunks and building FAISS index...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = np.asarray(embeddings.embed_documents([c.page_content for c in chunks]), dtype=np.float32)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    save_vector_index(INDEX_DIR, index, chunks)
    logger.info("  saved FAISS index to %s", INDEX_DIR)

    logger.info("Building BM25 index...")
    bm25_path = INDEX_DIR / "bm25.pkl"
    save_keyword_index(bm25_path, chunks)
    logger.info("  saved BM25 index to %s", bm25_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
