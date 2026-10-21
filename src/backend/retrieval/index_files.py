"""Reading and writing the index files in data/index/.

    index.faiss  the FAISS vectors (read with faiss itself)
    index.pkl    the passages, in the same order as the vectors
    bm25.pkl     the BM25 keyword index and its passages

The files committed today were written by LangChain 0.3 (langchain_community's
FAISS and BM25Retriever). Rather than depend on langchain_community just to
read them, load them with a restricted unpickler: it only builds the handful
of classes these files are known to contain, mapping LangChain's to small
stand-ins here. Anything else in a pickle is refused, which also means a
tampered file can't run arbitrary code on load.

Files written by scripts/ingest.py now use the plain dict format of
save_keyword_index / save_vector_index; the loaders read both.
"""

import pickle
from pathlib import Path

import faiss
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

KEYWORD_FORMAT = "gospel-rag/keyword-index/v1"
VECTOR_FORMAT = "gospel-rag/vector-docs/v1"


def split_on_whitespace(text: str) -> list[str]:
    """The tokenizer the BM25 index was built with (LangChain's default)."""
    return text.split()


class _LegacyBM25Retriever:
    """Stand-in for langchain_community's BM25Retriever: just keeps its state."""

    def __setstate__(self, state):
        fields = state.get("__dict__", state)
        self.vectorizer = fields["vectorizer"]
        self.docs = fields["docs"]


class _LegacyDocstore:
    """Stand-in for langchain_community's InMemoryDocstore."""

    def __setstate__(self, state):
        fields = state.get("__dict__", state)
        self.passages = fields["_dict"]


_ALLOWED = {
    ("langchain_core.documents.base", "Document"): Document,
    ("rank_bm25", "BM25Okapi"): BM25Okapi,
    ("langchain_community.retrievers.bm25", "BM25Retriever"): _LegacyBM25Retriever,
    ("langchain_community.retrievers.bm25", "default_preprocessing_func"): split_on_whitespace,
    ("langchain_community.docstore.in_memory", "InMemoryDocstore"): _LegacyDocstore,
}


class _IndexUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(
                f"{module}.{name} isn't allowed in an index file; "
                "rebuild the indexes with scripts/ingest.py"
            ) from None


def _load(path: Path):
    with open(path, "rb") as file:
        return _IndexUnpickler(file).load()


def load_keyword_index(path: Path) -> tuple[BM25Okapi, list[Document]]:
    """The BM25 scorer and its passages (passage i is document i of the scorer)."""
    data = _load(path)
    if isinstance(data, _LegacyBM25Retriever):
        return data.vectorizer, list(data.docs)
    if isinstance(data, dict) and data.get("format") == KEYWORD_FORMAT:
        return data["vectorizer"], list(data["docs"])
    raise ValueError(f"{path} isn't a keyword index this app can read")


def load_vector_index(directory: Path) -> tuple[faiss.Index, list[Document]]:
    """The FAISS index and its passages (passage i is vector i)."""
    index = faiss.read_index(str(directory / "index.faiss"))
    data = _load(directory / "index.pkl")
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], _LegacyDocstore):
        docstore, position_to_id = data
        docs = [docstore.passages[position_to_id[i]] for i in range(index.ntotal)]
    elif isinstance(data, dict) and data.get("format") == VECTOR_FORMAT:
        docs = list(data["docs"])
    else:
        raise ValueError(f"{directory / 'index.pkl'} isn't a vector index this app can read")
    if len(docs) != index.ntotal:
        raise ValueError(f"index.faiss has {index.ntotal} vectors but index.pkl has {len(docs)} passages")
    return index, docs


def save_keyword_index(path: Path, docs: list[Document]) -> None:
    corpus = [split_on_whitespace(doc.page_content) for doc in docs]
    with open(path, "wb") as file:
        pickle.dump({"format": KEYWORD_FORMAT, "vectorizer": BM25Okapi(corpus), "docs": docs}, file)


def save_vector_index(directory: Path, index: faiss.Index, docs: list[Document]) -> None:
    faiss.write_index(index, str(directory / "index.faiss"))
    with open(directory / "index.pkl", "wb") as file:
        pickle.dump({"format": VECTOR_FORMAT, "docs": docs}, file)
