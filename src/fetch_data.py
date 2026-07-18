"""Scrape every entry of the Bible Dictionary, Topical Guide, and Topics and
Questions from churchofjesuschrist.org into data/raw/.

Each index page (bd, tg, gospel-topics) is a single fully-rendered page
listing links to every entry, so no pagination handling is needed. The
per-entry cleaning logic (strip to article#main, drop script/style/iframe,
pull header + paragraph/list text) is unchanged from the initial sample.

Respectful-crawling behavior:
- A custom, identifying User-Agent.
- A 1-second delay after every single HTTP request (index pages included).
- Already-saved entries are skipped on re-run, so an interrupted crawl can
  resume without re-fetching thousands of pages or hammering the server.
"""

import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.churchofjesuschrist.org"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
REQUEST_DELAY_SECONDS = 1
RETRY_DELAY_SECONDS = 5
MAX_RETRIES = 3
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

# (category, index page URL, entry path prefix)
INDEXES = [
    ("bible_dictionary", f"{BASE_URL}/study/scriptures/bd?lang=eng", "/study/scriptures/bd"),
    ("topical_guide", f"{BASE_URL}/study/scriptures/tg?lang=eng", "/study/scriptures/tg"),
    ("topics_and_questions", f"{BASE_URL}/study/manual/gospel-topics?lang=eng", "/study/manual/gospel-topics"),
]


def fetch_html(url: str) -> str:
    """Fetch a URL, retrying on connection errors/timeouts (not on HTTP error statuses)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(url, timeout=30)
            response.raise_for_status()
            # The server doesn't declare a charset, so requests defaults to
            # ISO-8859-1 and mangles the page's actual UTF-8 text (e.g. curly quotes).
            response.encoding = "utf-8"
            return response.text
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                "  connection issue, retry %d/%d in %ds",
                attempt,
                MAX_RETRIES,
                RETRY_DELAY_SECONDS,
                exc_info=exc,
            )
            time.sleep(RETRY_DELAY_SECONDS)


def get_entry_urls(index_url: str, path_prefix: str) -> list[str]:
    """Extract every entry link from a fully-rendered index page."""
    html = fetch_html(index_url)
    time.sleep(REQUEST_DELAY_SECONDS)

    soup = BeautifulSoup(html, "html.parser")
    pattern = re.compile(rf"^{re.escape(path_prefix)}/([^/?]+)\?lang=eng$")

    slugs = set()
    for link in soup.find_all("a", href=pattern):
        slugs.add(pattern.match(link["href"]).group(1))

    return [f"{BASE_URL}{path_prefix}/{slug}?lang=eng" for slug in sorted(slugs) if not slug.startswith("_")]


def extract_doctrinal_text(html: str) -> tuple[str, str]:
    """Return (title, body_text) with nav/footer/chrome stripped, keeping headers and paragraphs."""
    soup = BeautifulSoup(html, "html.parser")

    article = soup.select_one("article#main")
    if article is None:
        raise ValueError("Could not find core article content on page")

    # Global site nav/header/footer already live outside article#main, so this
    # only needs to drop embedded non-text chrome (the Topical Guide's own
    # reference lists are legitimately wrapped in a <nav> tag and must stay).
    for tag in article.select("script, style, iframe"):
        tag.decompose()

    title_tag = article.select_one("h1")
    title = title_tag.get_text(" ", strip=True) if title_tag else "untitled"

    body = article.select_one("div.body") or article
    text_tags = ("h1", "h2", "h3", "h4", "p", "li")
    lines = []
    for tag in body.find_all(text_tags):
        # Skip containers whose text will already be captured by a nested
        # match (e.g. Topical Guide entries wrap each <p> inside an <li>).
        if tag.find(text_tags):
            continue
        text = tag.get_text(" ", strip=True)
        if text:
            lines.append(text)

    body_text = "\n\n".join(lines)
    body_text = re.sub(r"[ \t]+", " ", body_text).strip()

    return title, body_text


def slugify(url: str) -> str:
    path = url.split("?", 1)[0].rstrip("/")
    return path.rsplit("/", 1)[-1]


def out_path_for(category: str, url: str) -> Path:
    return OUTPUT_DIR / category / f"{slugify(url)}.txt"


def already_saved(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def save_text(category: str, url: str, title: str, body_text: str) -> Path:
    out_path = out_path_for(category, url)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"{title}\nSource: {url}\n\n{body_text}\n"

    # Write to a temp file and rename so a mid-write kill can't leave a
    # truncated .txt that already_saved() would mistake for a finished entry.
    tmp_path = out_path.with_suffix(".txt.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path


def main() -> None:
    logger.info("Building full entry list from index pages...")
    entries: list[tuple[str, str]] = []  # (category, url)
    for category, index_url, prefix in INDEXES:
        urls = get_entry_urls(index_url, prefix)
        logger.info("  %s: %d entries", category, len(urls))
        entries.extend((category, url) for url in urls)

    total = len(entries)
    logger.info("Total entries to fetch: %d", total)

    failures: list[str] = []
    skipped = 0
    for i, (category, url) in enumerate(entries, start=1):
        slug = slugify(url)

        if already_saved(out_path_for(category, url)):
            skipped += 1
            continue

        logger.info("Fetching %d/%d [%s]: %s", i, total, category, slug)
        try:
            html = fetch_html(url)
            title, body_text = extract_doctrinal_text(html)
            save_text(category, url, title, body_text)
        except Exception as exc:
            logger.error("  FAILED: %s", url, exc_info=exc)
            failures.append(url)
        time.sleep(REQUEST_DELAY_SECONDS)

    logger.info(
        "Done. %d fetched, %d already present, %d failed.",
        total - skipped - len(failures),
        skipped,
        len(failures),
    )
    if failures:
        logger.warning("Failed URLs:")
        for url in failures:
            logger.warning("  %s", url)


if __name__ == "__main__":
    main()
