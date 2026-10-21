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
}

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
