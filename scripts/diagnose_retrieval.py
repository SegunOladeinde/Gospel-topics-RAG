"""Show what each retriever returns for a set of sample questions.

For every question, prints the keyword (BM25) hits and which query tokens
they matched, the FAISS hits with their relevance score, and the final
source list the API would return. Used to investigate off-topic sources;
makes one embeddings call per question per retriever.

Usage:
    uv run python scripts/diagnose_retrieval.py
    uv run python scripts/diagnose_retrieval.py "What is baptism?" "Who was Moroni?"
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.backend.keyword_retriever import content_words, query_tokens  # noqa: E402
from src.backend.main import collect_unique_sources, describe_source, load_retriever  # noqa: E402

SAMPLE_QUESTIONS = [
    "What is baptism?",
    "What is the sacrament?",
    "Why do we partake of the sacrament?",
    "What is faith?",
    "What is tithing?",
    "How do I repent?",
    "What is the Word of Wisdom?",
    "Who was Moroni?",
    "What happens after we die?",
    "Why do we build temples?",
]


async def main() -> None:
    questions = sys.argv[1:] or SAMPLE_QUESTIONS
    ensemble = load_retriever()
    faiss_retriever, keyword = ensemble.retrievers
    vectorstore = faiss_retriever.vectorstore
    print(f"keyword retriever: {type(keyword).__name__}, k={keyword.k}, weights={ensemble.weights}\n")

    for question in questions:
        print(f"=== {question!r}  content words: {content_words(question)}")

        tokens = set(query_tokens(question))
        for doc in keyword.invoke(question):
            matched = sorted(tokens & set(doc.page_content.split()))  # the index's own tokenizer
            print(f"  BM25  {describe_source(doc):<55} matched={matched}")

        for doc, score in vectorstore.similarity_search_with_relevance_scores(question, k=keyword.k):
            print(f"  FAISS {describe_source(doc):<55} relevance={score:.3f}")

        documents = await ensemble.ainvoke(question)
        print(f"  -> sources: {collect_unique_sources(documents)}\n")


if __name__ == "__main__":
    asyncio.run(main())
