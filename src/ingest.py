"""Phase 3: chunk the scraped doctrinal text and build the retrieval indexes.

Reads every .txt file under data/raw/ (recursively), splits each into
overlapping chunks, embeds them with OpenAI, and writes two indexes to
data/index/:
- a FAISS vector store (semantic search)
- a BM25Retriever, pickled (keyword search)

Both are loaded together into an EnsembleRetriever at query time (see
CLAUDE.md), so they're built from the exact same chunks/metadata here.
"""

import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    print(f"Reading .txt files from {RAW_DIR} ...")
    texts, metadatas = load_source_documents()
    print(f"  {len(texts)} source documents found")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.create_documents(texts, metadatas=metadatas)
    print(f"  split into {len(chunks)} chunks")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Embedding chunks and building FAISS index...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"  saved FAISS index to {INDEX_DIR}")

    print("Training BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_path = INDEX_DIR / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"  saved BM25 index to {bm25_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
