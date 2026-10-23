"""Scrape the Institute manuals and Church History Topics into data/raw/.

    institute_bom           Book of Mormon Student Manual (2018 edition, Religion 121-122)
    institute_restoration   Foundations of the Restoration Class Preparation Material (2019),
                            the student readings for Religion 225 (there is no "student manual")
    church_history_topics   Church History Topics

Collected with the Church's written permission, for personal, non-commercial study.

Each page is saved as one .txt file, cut into sections (src/backend/ingestion/
sections.py): its real title, its URL after redirects, and per section the
heading, the heading's id (for deep links) and the scripture references the
section links to.

Polite crawling:
- An identifying User-Agent.
- At most one request a second (a 1.1 s pause after every request).
- Resume-safe: pages already saved are skipped, and files are written to a
  temporary name and renamed, so an interrupted run never leaves half a page.

Usage:
    uv run python scripts/fetch_manuals.py                       # all three
    uv run python scripts/fetch_manuals.py church_history_topics # just one
"""

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.ingestion.sections import extract_sections, render_raw  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.churchofjesuschrist.org"
USER_AGENT = "lds-rag-study-tool/1.0 (personal non-commercial study; collected with permission)"
REQUEST_DELAY_SECONDS = 1.1
MAX_RETRIES = 3
RAW_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class Source:
    collection: str
    index_path: str  # the contents page listing every page
    page_pattern: str  # regex for the page slugs to keep
    split_tags: tuple[str, ...]  # headings that start a new section


SOURCES = [
    # Chapters only: the appendix pages are maps and charts.
    Source("institute_bom", "/study/manual/book-of-mormon-student-manual-2018", r"chapter-\d+-[a-z0-9-]+", ("h2", "h3")),
    Source("institute_restoration", "/study/manual/foundations-of-the-restoration-class-preparation-material-2019",
           r"lesson-\d+-class-preparation-material-[a-z0-9-]+", ("h2",)),
    Source("church_history_topics", "/study/history/topics", r"[a-z0-9]+(?:-[a-z0-9]+)*", ("h2",)),
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def fetch(url: str) -> tuple[str, str]:
    """(html, final URL), retrying dropped connections and 5xx; then waits, so requests stay ≤ 1/s."""
    try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = SESSION.get(url, timeout=30)
                if response.status_code < 500:
                    response.raise_for_status()
                    response.encoding = "utf-8"  # the server doesn't declare it
                    return response.text, response.url
            except (requests.ConnectionError, requests.Timeout):
                if attempt == MAX_RETRIES:
                    raise
            logger.warning("  retry %d/%d: %s", attempt, MAX_RETRIES, url)
            time.sleep(5 * attempt)
        raise RuntimeError(f"{url}: server error after {MAX_RETRIES} attempts")
    finally:
        time.sleep(REQUEST_DELAY_SECONDS)


def canonical(url: str) -> str:
    """The page's URL without query or fragment, plus ?lang=eng (how the site links pages)."""
    return f"{BASE_URL}{urlsplit(url).path}?lang=eng"


def page_slugs(source: Source) -> list[str]:
    html, _ = fetch(f"{BASE_URL}{source.index_path}?lang=eng")
    pattern = re.compile(rf'href="{re.escape(source.index_path)}/({source.page_pattern})(?:\?[^"]*)?"')
    return sorted(set(pattern.findall(html)))


def save(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def scrape(source: Source) -> tuple[int, int, list[str]]:
    slugs = page_slugs(source)
    logger.info("%s: %d pages", source.collection, len(slugs))
    fetched = skipped = 0
    failures = []
    for number, slug in enumerate(slugs, start=1):
        path = RAW_DIR / source.collection / f"{slug}.txt"
        if path.exists() and path.stat().st_size > 0:
            skipped += 1
            continue
        url = f"{BASE_URL}{source.index_path}/{slug}?lang=eng"
        logger.info("  %d/%d %s", number, len(slugs), slug)
        try:
            html, final_url = fetch(url)
            title, sections = extract_sections(html, source.split_tags)
            if not sections:
                raise ValueError("no text found")
            save(path, render_raw(title, canonical(final_url), sections))
            fetched += 1
        except Exception as exc:  # noqa: BLE001 - log it and carry on; a re-run retries it
            logger.error("  FAILED %s: %s", url, exc)
            failures.append(url)
    return fetched, skipped, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("collections", nargs="*", help="collections to scrape (default: all)")
    args = parser.parse_args()
    chosen = [s for s in SOURCES if not args.collections or s.collection in args.collections]
    if not chosen:
        sys.exit(f"unknown collection; choose from {', '.join(s.collection for s in SOURCES)}")

    all_failures = []
    for source in chosen:
        fetched, skipped, failures = scrape(source)
        logger.info("%s: %d fetched, %d already saved, %d failed", source.collection, fetched, skipped, len(failures))
        all_failures += failures
    if all_failures:
        logger.warning("Failed (re-run to retry):\n  %s", "\n  ".join(all_failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
