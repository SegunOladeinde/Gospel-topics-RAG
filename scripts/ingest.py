"""Load the collections in data/collections.yaml into the pgvector store.

Incremental: only new or changed pages (by sha256) are chunked and embedded,
and pages removed from disk are deleted with their passages. Always run
--dry-run first: it prints document/passage/token counts and the estimated
embedding cost without spending anything. A real run refuses to spend more
than --max-cost (default $1) unless you pass --approve-cost.

Writes through Supabase's REST API when SUPABASE_URL and
SUPABASE_SERVICE_ROLE_KEY are set, otherwise through DATABASE_URL (choose
with --store). Secrets are read from .env and never printed.

Usage:
    uv run python scripts/ingest.py --dry-run
    uv run python scripts/ingest.py
    uv run python scripts/ingest.py --collection topical_guide
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.backend.ingestion.config import load_config  # noqa: E402
from src.backend.ingestion.documents import SourceDocument, load_documents  # noqa: E402
from src.backend.ingestion.pipeline import run_ingestion  # noqa: E402
from src.backend.ingestion.stores import RestRagStore, SqlRagStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("ingest")


def make_store(kind: str):
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if kind == "rest" or (kind == "auto" and url and key):
        if not (url and key):
            sys.exit("--store rest needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
        return "Supabase REST (service role)", RestRagStore(url, key)
    if os.getenv("DATABASE_URL", "").strip():
        from src.backend.database import get_session_factory

        return "postgres (DATABASE_URL)", SqlRagStore(get_session_factory())
    sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY, or DATABASE_URL, in .env")


def load_known_embeddings() -> dict[str, list[float]]:
    """Passage text -> embedding, from the FAISS index (same model, same passages)."""
    from src.backend.retrieval.index_files import load_vector_index

    index_dir = PROJECT_ROOT / "data" / "index"
    if not (index_dir / "index.faiss").exists():
        return {}
    index, docs = load_vector_index(index_dir)
    vectors = index.reconstruct_n(0, index.ntotal)
    return {doc.page_content: vector.tolist() for doc, vector in zip(docs, vectors)}


class RedirectResolver:
    """Follows a page's redirects (at most 2 requests a second) to store its final URL.

    A redirect to the collection's index page means the slug wasn't found;
    then the original URL is kept rather than storing the index.
    """

    def __init__(self, client: httpx.Client, per_second: float = 2.0):
        self.client = client
        self.min_interval = 1.0 / per_second
        self.last = 0.0
        self.moved = 0

    def __call__(self, doc: SourceDocument) -> str:
        wait = self.last + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self.last = time.monotonic()
        try:
            response = self.client.get(doc.url)
        except httpx.HTTPError:
            return doc.url
        final = str(response.url)
        asked, landed = urlsplit(doc.url).path.rstrip("/"), urlsplit(final).path.rstrip("/")
        if response.status_code != 200 or landed == asked.rsplit("/", 1)[0]:
            return doc.url
        if landed != asked:
            self.moved += 1
            logger.info("Moved: %s/%s -> %s", doc.collection, doc.slug, final)
        return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="count documents, passages, tokens and cost only")
    parser.add_argument("--collection", action="append", help="only this collection (repeatable)")
    parser.add_argument("--max-cost", type=float, default=1.0, help="refuse a real run estimated above this (USD)")
    parser.add_argument("--approve-cost", action="store_true", help="allow a real run above --max-cost")
    parser.add_argument("--reuse-embeddings", action=argparse.BooleanOptionalAction, default=True,
                        help="take embeddings for identical passages from data/index/ (default on)")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="ingest at most this many new/changed documents per run (resume with the next run)")
    parser.add_argument("--store", choices=["auto", "rest", "sql"], default="auto",
                        help="auto = REST if the Supabase keys are set, else DATABASE_URL")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config()
    if args.collection:
        unknown = set(args.collection) - {c.name for c in config.collections}
        if unknown:
            sys.exit(f"unknown collection(s): {', '.join(sorted(unknown))}")
        config = replace(config, collections=[c for c in config.collections if c.name in args.collection])

    documents = [doc for collection in config.collections for doc in load_documents(collection)]
    store_name, store = make_store(args.store)
    print(f"store: {store_name}; collections: {', '.join(c.name for c in config.collections)}")

    known = load_known_embeddings() if args.reuse_embeddings else {}
    if known:
        print(f"reusing embeddings for {len(known):,} passages from data/index/")
    report = run_ingestion(config, documents, store, embed=None, dry_run=True,
                           already_embedded=known.__contains__, max_documents=args.max_docs)
    print(
        f"documents on disk: {report.documents_on_disk} | new {report.new}, changed {report.changed}, "
        f"unchanged {report.unchanged}, removed {report.removed}\n"
        f"passages to embed: {report.chunks_to_embed} | tokens: {report.tokens_to_embed:,} | "
        f"estimated cost: ${report.estimated_usd:.4f} ({config.embedding_model})"
    )
    if args.dry_run:
        return
    if report.estimated_usd > args.max_cost and not args.approve_cost:
        sys.exit(f"Estimated ${report.estimated_usd:.2f} is over --max-cost ${args.max_cost:.2f}; "
                 "rerun with --approve-cost once that's approved.")
    if not report.new and not report.changed and not report.removed:
        print("Nothing to do.")
        return

    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model=config.embedding_model, max_retries=2)

    def embed(texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in known]
        fresh = dict(zip(missing, embeddings.embed_documents(missing))) if missing else {}
        return [known[t] if t in known else fresh[t] for t in texts]

    with httpx.Client(follow_redirects=True, timeout=20, headers={"User-Agent": "gospel-topics-rag ingest"}) as client:
        resolver = RedirectResolver(client)
        started = time.monotonic()
        report = run_ingestion(config, documents, store, embed=embed, dry_run=False,
                               resolve_url=resolver, already_embedded=known.__contains__,
                               max_documents=args.max_docs)
    print({k: v for k, v in asdict(report).items()})
    print(f"done in {time.monotonic() - started:.0f}s; pages whose URL moved: {resolver.moved}")


if __name__ == "__main__":
    main()
