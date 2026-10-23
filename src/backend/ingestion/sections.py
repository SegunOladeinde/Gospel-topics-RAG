"""Pages split into sections: Institute manuals and Church History Topics.

scripts/fetch_manuals.py turns each page's HTML into sections with
extract_sections() and saves them with render_raw(); ingestion reads them
back with parse_raw() (documents.py), so each section becomes its own
document with its own deep link and title.

A sectioned raw file looks like:

    Chapter 30: Alma 32–35
    Source: https://www.churchofjesuschrist.org/study/manual/.../chapter-30-alma-32-35?lang=eng

    ## {#title1}
    Scriptures: Alma 32–35
    The chapter's opening text...

    ## Alma 32:21. Faith and Hope {#title8}
    Scriptures: Alma 32:21; Hebrews 11:1
    The section's text...

Each "## heading {#anchor}" line starts a section; the anchor is the
heading's id on the page (links go to <url>&id=<anchor>#<anchor>). The
first section has no heading: it's the text before the first one.
"""

import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlsplit

# Standard works URL slugs (/study/scriptures/<volume>/<book>/<chapter>) -> book names.
BOOKS: dict[str, str] = {
    # Old Testament
    "gen": "Genesis", "ex": "Exodus", "lev": "Leviticus", "num": "Numbers", "deut": "Deuteronomy",
    "josh": "Joshua", "judg": "Judges", "ruth": "Ruth", "1-sam": "1 Samuel", "2-sam": "2 Samuel",
    "1-kgs": "1 Kings", "2-kgs": "2 Kings", "1-chr": "1 Chronicles", "2-chr": "2 Chronicles",
    "ezra": "Ezra", "neh": "Nehemiah", "esth": "Esther", "job": "Job", "ps": "Psalms", "prov": "Proverbs",
    "eccl": "Ecclesiastes", "song": "Song of Solomon", "isa": "Isaiah", "jer": "Jeremiah",
    "lam": "Lamentations", "ezek": "Ezekiel", "dan": "Daniel", "hosea": "Hosea", "joel": "Joel",
    "amos": "Amos", "obad": "Obadiah", "jonah": "Jonah", "micah": "Micah", "nahum": "Nahum",
    "hab": "Habakkuk", "zeph": "Zephaniah", "hag": "Haggai", "zech": "Zechariah", "mal": "Malachi",
    # New Testament
    "matt": "Matthew", "mark": "Mark", "luke": "Luke", "john": "John", "acts": "Acts", "rom": "Romans",
    "1-cor": "1 Corinthians", "2-cor": "2 Corinthians", "gal": "Galatians", "eph": "Ephesians",
    "philip": "Philippians", "col": "Colossians", "1-thes": "1 Thessalonians", "2-thes": "2 Thessalonians",
    "1-tim": "1 Timothy", "2-tim": "2 Timothy", "titus": "Titus", "philem": "Philemon", "heb": "Hebrews",
    "james": "James", "1-pet": "1 Peter", "2-pet": "2 Peter", "1-jn": "1 John", "2-jn": "2 John",
    "3-jn": "3 John", "jude": "Jude", "rev": "Revelation",
    # Book of Mormon
    "1-ne": "1 Nephi", "2-ne": "2 Nephi", "jacob": "Jacob", "enos": "Enos", "jarom": "Jarom", "omni": "Omni",
    "w-of-m": "Words of Mormon", "mosiah": "Mosiah", "alma": "Alma", "hel": "Helaman", "3-ne": "3 Nephi",
    "4-ne": "4 Nephi", "morm": "Mormon", "ether": "Ether", "moro": "Moroni",
    "three": "Testimony of Three Witnesses", "eight": "Testimony of Eight Witnesses",
    # Doctrine and Covenants, Pearl of Great Price
    "dc": "Doctrine and Covenants", "od": "Official Declaration",
    "moses": "Moses", "abr": "Abraham", "js-m": "Joseph Smith—Matthew", "js-h": "Joseph Smith—History",
    "a-of-f": "Articles of Faith",
}

_SCRIPTURE_PATH = re.compile(r"^/study/scriptures/[a-z-]+/([a-z0-9-]+)(?:/(\d+))?/?$")
_VERSE_RANGE = re.compile(r"^p?(\d+)(?:-p?(\d+))?$")
_HEADING = re.compile(r"^## (.*?) ?\{#([\w-]+)\}$")
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?)])")


@dataclass
class Section:
    heading: str  # "" for the text before the first heading
    anchor: str  # the heading's id on the page, e.g. "title8"
    scriptures: list[str] = field(default_factory=list)  # e.g. ["Alma 32:21", "Hebrews 11:1"]
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


def _verses(id_param: str) -> str:
    """"p8-p16" -> "8–16", "p14,16" -> "14, 16"; "" if it isn't a verse list."""
    parts = []
    for token in id_param.split(","):
        match = _VERSE_RANGE.match(token.strip())
        if not match:
            return ""
        first, last = match.groups()
        parts.append(f"{first}–{last}" if last and last != first else first)
    return ", ".join(parts)


def scripture_ref(href: str) -> str | None:
    """A scripture link's reference, e.g. "/study/scriptures/bofm/alma/32?lang=eng&id=p21#p21"
    -> "Alma 32:21". None for links that aren't to a known book."""
    parts = urlsplit(href)
    match = _SCRIPTURE_PATH.match(parts.path)
    if not match or match.group(1) not in BOOKS:
        return None
    book, chapter = BOOKS[match.group(1)], match.group(2)
    if not chapter:
        return book
    name = "D&C" if book == "Doctrine and Covenants" else book
    verses = _verses(parse_qs(parts.query).get("id", [""])[0])
    return f"{name} {chapter}:{verses}" if verses else f"{name} {chapter}"


def _clean(text: str) -> str:
    # get_text(" ") puts a space after every link ("Alma 32 . Having Faith").
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", _SPACES.sub(" ", text)).strip()


def extract_sections(html: str, split_tags: tuple[str, ...] = ("h2",)) -> tuple[str, list[Section]]:
    """(page title, sections) from a churchofjesuschrist.org study page.

    The page is cut at every heading in split_tags; other headings stay in
    the text as their own lines. Footnotes, footnote markers, image credits,
    "Church Resources" reading lists and the page's citation lines are left
    out. A section's scriptures are the scripture links in its heading and
    text, in order, without repeats.
    """
    from bs4 import BeautifulSoup  # only needed when scraping

    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article#main")
    if article is None:
        raise ValueError("no article#main on the page")
    for tag in article.select(
        "script, style, iframe, footer.notes, sup.marker, .credit, .citation-info, "
        "p.reference, p.short-reference, nav.toc"
    ):
        tag.decompose()
    for aside in article.select("aside"):
        title = aside.select_one(".title")
        if title and _clean(title.get_text()).lower() in ("church resources", "sources"):
            aside.decompose()

    h1 = article.select_one("h1")
    if h1 is None:
        raise ValueError("no h1 on the page")
    heading_number = h1.find_previous_sibling("p", class_="title-number")
    page_title = _clean(h1.get_text(" "))
    if heading_number:
        page_title = f"{_clean(heading_number.get_text(' '))}: {page_title}"

    text_tags = ("h2", "h3", "h4", "h5", "h6", "p", "li")
    sections = [Section(heading="", anchor=h1.get("id") or "title1")]
    for tag in article.find_all(text_tags):
        if tag.find(text_tags):  # a container whose text a nested tag already gives
            continue
        if "title-number" in (tag.get("class") or []):
            continue  # folded into the heading or title it numbers
        refs = [ref for a in tag.select("a[href]") if (ref := scripture_ref(a["href"]))]
        text = _clean(tag.get_text(" "))
        if tag.name in split_tags and tag.get("id"):
            number = tag.find_previous_sibling("p", class_="title-number")
            heading = f"{_clean(number.get_text(' '))}: {text}" if number else text
            sections.append(Section(heading=heading, anchor=tag["id"]))
        elif not text:
            continue
        else:
            sections[-1].paragraphs.append(text)
        for ref in refs:
            if ref not in sections[-1].scriptures:
                sections[-1].scriptures.append(ref)
    return page_title, [s for s in sections if s.paragraphs]


def render_raw(title: str, url: str, sections: list[Section]) -> str:
    """The raw file for a sectioned page (see the module docstring)."""
    blocks = [f"{title}\nSource: {url}"]
    for section in sections:
        lines = [f"## {section.heading} {{#{section.anchor}}}" if section.heading else f"## {{#{section.anchor}}}"]
        if section.scriptures:
            lines.append("Scriptures: " + "; ".join(section.scriptures))
        lines.append(section.text)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def parse_raw(content: str) -> tuple[str, str, list[Section]]:
    """(title, url, sections) from a raw file written by render_raw."""
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    title = lines[0].strip() if lines else ""
    url = lines[1].removeprefix("Source:").strip() if len(lines) > 1 and lines[1].startswith("Source:") else ""
    sections: list[Section] = []
    for line in lines[2:]:
        match = _HEADING.match(line)
        if match:
            sections.append(Section(heading=match.group(1).strip(), anchor=match.group(2)))
        elif not sections:
            continue
        elif line.startswith("Scriptures: ") and not sections[-1].paragraphs and not sections[-1].scriptures:
            sections[-1].scriptures = [ref.strip() for ref in line.removeprefix("Scriptures: ").split(";") if ref.strip()]
        elif line.strip():
            sections[-1].paragraphs.append(line.strip())
    return title, url, [s for s in sections if s.paragraphs]


def section_url(url: str, anchor: str, first: bool) -> str:
    """The page URL, deep-linked to the section's heading (the page itself for the first section)."""
    if first or not url:
        return url
    return f"{url}{'&' if '?' in url else '?'}id={anchor}#{anchor}"
