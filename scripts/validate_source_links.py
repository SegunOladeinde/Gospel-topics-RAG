"""Check that source citation links resolve to real churchofjesuschrist.org pages.

The site doesn't 404 on an unknown slug: it redirects to the collection's
index page (e.g. /study/scriptures/tg?lang=eng). So for every unique source id
in the BM25 index we build its URL with src/backend/source_links.py, follow
redirects, and sort the result into:

    ok       landed on the page itself
    index    redirected to the collection index (slug not found)
    moved    redirected to a different real page (the site merged or
             renamed the topic); fine for users, listed for review
    other    an HTTP error, a timeout, or an id the helper can't link at all

If data/raw/ is present, each URL is also compared with the "Source:" URL the
scraper stored in that file.

Usage:
    uv run python scripts/validate_source_links.py              # every id
    uv run python scripts/validate_source_links.py --sample 50  # 50 per collection
"""

import argparse
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.main import BM25_INDEX_PATH, describe_source  # noqa: E402
from src.backend.source_links import COLLECTIONS, source_link  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
USER_AGENT = "gospel-topics-rag source link validator"


def load_source_ids() -> list[str]:
    """Every unique source id the API can return, i.e. one per indexed file."""
    with open(BM25_INDEX_PATH, "rb") as bm25_file:
        retriever = pickle.load(bm25_file)
    return sorted({describe_source(document) for document in retriever.docs})


def stored_url(source_id: str) -> str | None:
    """The URL the scraper recorded on line 2 of the raw file, if we have it."""
    path = RAW_DIR / f"{source_id}.txt"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines()[:5]:
        if line.startswith("Source: "):
            return line.removeprefix("Source: ").strip()
    return None


def classify(client: httpx.Client, url: str) -> str:
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        return f"other ({type(exc).__name__})"
    if response.status_code != 200:
        return f"other (HTTP {response.status_code})"

    wanted = urlsplit(url).path.rstrip("/")
    landed = urlsplit(str(response.url)).path.rstrip("/")
    if landed == wanted:
        return "ok"
    if landed == wanted.rsplit("/", 1)[0]:
        return "index"
    return f"moved (to {landed})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, help="check at most N ids per collection")
    parser.add_argument("--rate", type=float, default=2.0, help="max requests per second (default 2)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for --sample")
    args = parser.parse_args()
    if not 0 < args.rate <= 2:
        parser.error("--rate must be between 0 and 2 requests per second")

    by_collection: dict[str, list[str]] = defaultdict(list)
    for source_id in load_source_ids():
        by_collection[source_id.partition("/")[0]].append(source_id)

    rng = random.Random(args.seed)
    to_check = {
        collection: rng.sample(ids, args.sample) if args.sample and len(ids) > args.sample else ids
        for collection, ids in sorted(by_collection.items())
    }

    min_interval = 1.0 / args.rate
    last_request = 0.0
    counts: dict[str, Counter] = defaultdict(Counter)
    problems: list[str] = []
    stored_mismatches: list[str] = []

    with httpx.Client(follow_redirects=True, timeout=20, headers={"User-Agent": USER_AGENT}) as client:
        for collection, ids in to_check.items():
            print(f"{collection}: checking {len(ids)} of {len(by_collection[collection])} ids", flush=True)
            for source_id in ids:
                link = source_link(source_id)
                stored = stored_url(source_id)
                if stored and stored != link.url:
                    stored_mismatches.append(f"{source_id}: built {link.url} but raw file has {stored}")
                if link.url is None:
                    result = "other (no link)"
                else:
                    wait = last_request + min_interval - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    last_request = time.monotonic()
                    result = classify(client, link.url)

                counts[collection][result.split(" ")[0]] += 1
                if result != "ok":
                    problems.append(f"  {result:<10} {source_id}  {link.url}")

    print()
    print(f"{'collection':<22}{'checked':>8}{'ok':>6}{'index':>7}{'moved':>7}{'other':>7}")
    for collection in to_check:
        c = counts[collection]
        print(f"{collection:<22}{sum(c.values()):>8}{c['ok']:>6}{c['index']:>7}{c['moved']:>7}{c['other']:>7}")
    unknown = sorted(set(by_collection) - set(COLLECTIONS))
    if unknown:
        print(f"\nCollections the helper doesn't know (rendered as plain text): {', '.join(unknown)}")
    if problems:
        print("\nNot ok:")
        print("\n".join(problems))
    if stored_mismatches:
        print("\nBuilt URL differs from the URL stored in data/raw:")
        print("\n".join(stored_mismatches))
    elif RAW_DIR.is_dir():
        print("\nAll built URLs match the Source: URLs stored in data/raw.")

    broken = any(c["index"] or c["other"] for c in counts.values())
    sys.exit(1 if broken or stored_mismatches else 0)


if __name__ == "__main__":
    main()
