"""Turn source ids like "topical_guide/faith" into links to churchofjesuschrist.org.

Source ids come from describe_source() in main.py: the data/raw/ folder name
plus the scraped file's stem. Every scraped file records the page it came from
on its second line ("Source: https://..."), and those URLs follow the patterns
below exactly. data/raw/ isn't deployed (it's gitignored), so we rebuild the
URL from the id instead of looking it up; scripts/validate_source_links.py
checks the two still agree.

This module is the single place that mapping lives. The API returns its output
as ``source_links`` and the frontend just renders it.
"""

import re

from pydantic import BaseModel

BASE_URL = "https://www.churchofjesuschrist.org/study"

# folder name -> (readable collection name, URL template for a slug)
COLLECTIONS: dict[str, tuple[str, str]] = {
    "bible_dictionary": ("Bible Dictionary", f"{BASE_URL}/scriptures/bd/{{slug}}?lang=eng"),
    "topical_guide": ("Topical Guide", f"{BASE_URL}/scriptures/tg/{{slug}}?lang=eng"),
    "topics_and_questions": ("Gospel Topics", f"{BASE_URL}/manual/gospel-topics/{{slug}}?lang=eng"),
    "institute_bom": (
        "Institute Manual (Book of Mormon)",
        f"{BASE_URL}/manual/book-of-mormon-student-manual-2018/{{slug}}?lang=eng",
    ),
    "institute_restoration": (
        "Institute Manual (Restoration)",
        f"{BASE_URL}/manual/foundations-of-the-restoration-class-preparation-material-2019/{{slug}}?lang=eng",
    ),
    "church_history_topics": ("Church History Topics", f"{BASE_URL}/history/topics/{{slug}}?lang=eng"),
}
# Sectioned collections' ids are "<page>--<section>" (ingestion/documents.py);
# without a stored URL, the fallback link goes to the page.
SECTIONED = {"institute_bom", "institute_restoration", "church_history_topics"}
_SECTION_SEPARATOR = "--"

# Slugs on the site are lowercase words joined by single hyphens. Anything
# else (e.g. a "_" left behind by describe_source's sanitising) wouldn't
# resolve, so we show it as plain text rather than link to a broken page.
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STUDY_GUIDE_SUFFIX = "-study-guide"
_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}


class SourceLink(BaseModel):
    """One citation chip: the raw id, a readable label, and a URL (or None)."""

    id: str
    label: str
    url: str | None


def _title_from_slug(slug: str) -> str:
    """'aaron-descendants-of' -> 'Aaron Descendants of'."""
    words = [word for word in re.split(r"[-_\s]+", slug) if word]
    return " ".join(
        word if index and word in _SMALL_WORDS else word.capitalize()
        for index, word in enumerate(words)
    )


def source_link(source_id: str) -> SourceLink:
    """Map a source id to a label and an absolute URL.

    Unknown collections and malformed slugs get ``url=None`` so they render
    as plain text instead of a link that would 404.
    """
    collection, _, raw_slug = source_id.strip().partition("/")
    slug = raw_slug.strip().lower()

    if collection not in COLLECTIONS:
        return SourceLink(id=source_id, label=source_id, url=None)

    collection_name, url_template = COLLECTIONS[collection]
    if collection in SECTIONED:
        slug = slug.split(_SECTION_SEPARATOR, 1)[0]
    if slug.endswith(_STUDY_GUIDE_SUFFIX) and len(slug) > len(_STUDY_GUIDE_SUFFIX):
        title = f"{_title_from_slug(slug.removesuffix(_STUDY_GUIDE_SUFFIX))} (Study Guide)"
    else:
        title = _title_from_slug(slug)

    if not _SLUG.fullmatch(slug):
        label = f"{collection_name}: {title}" if title else source_id
        return SourceLink(id=source_id, label=label, url=None)

    return SourceLink(
        id=source_id,
        label=f"{collection_name}: {title}",
        url=url_template.format(slug=slug),
    )


_CHURCH_SITE = re.compile(r"https://www\.churchofjesuschrist\.org/[^\s\"'<>]*")


def source_link_for(source_id: str, title: str | None = None, url: str | None = None) -> SourceLink:
    """Like source_link(), but using the page's real title and URL when known.

    The pgvector store keeps each page's title ("Aaron, Descendants of") and
    canonical URL (after redirects), so links read and resolve correctly;
    the FAISS index only has ids, which fall back to source_link().
    """
    fallback = source_link(source_id)
    collection, _, slug = source_id.partition("/")
    if not title or collection not in COLLECTIONS:
        return fallback
    label_title = title.strip()
    if slug.endswith(_STUDY_GUIDE_SUFFIX) and "study guide" not in label_title.lower():
        label_title = f"{label_title} (Study Guide)"
    # Only ever link to the Church's site.
    link = url if url and _CHURCH_SITE.fullmatch(url) else fallback.url
    return SourceLink(id=source_id, label=f"{COLLECTIONS[collection][0]}: {label_title}", url=link)
