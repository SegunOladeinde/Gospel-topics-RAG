"""Raw .txt files -> SourceDocuments with their real title and URL."""

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from src.backend.ingestion.config import CollectionConfig
from src.backend.ingestion.sections import BOOKS, parse_raw, section_url

# Bump to force every document to re-ingest when chunking/content rules change.
# 2: passages are cut from the raw file text unchanged (title and Source line
# included), exactly like the FAISS index, so both backends search the same
# passages and the FAISS embeddings can be reused.
CONTENT_VERSION = 2


@dataclass(frozen=True)
class SourceDocument:
    collection: str
    slug: str
    title: str
    url: str
    text: str  # the raw file text, what gets chunked
    sha256: str  # of the raw file plus the chunking settings
    # Sectioned collections only: a line put before every passage (where the
    # passage comes from), and what's stored in each passage's metadata.
    chunk_prefix: str = ""
    metadata: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def key(self) -> tuple[str, str]:
        return (self.collection, self.slug)

    def with_url(self, url: str) -> "SourceDocument":
        return replace(self, url=url)

    def load(self) -> "SourceDocument":
        return self


@dataclass(frozen=True)
class DocumentRef:
    """A page on disk known only by its sha256 until load() reads it again,
    so planning a run over thousands of pages keeps only hashes in memory."""

    config: CollectionConfig
    slug: str
    path: Path
    sha256: str

    @property
    def collection(self) -> str:
        return self.config.name

    @property
    def key(self) -> tuple[str, str]:
        return (self.config.name, self.slug)

    def load(self) -> SourceDocument:
        return parse_document(self.config, self.slug, self.path.read_bytes())


def _title_from_slug(slug: str) -> str:
    return " ".join(word.capitalize() for word in slug.split("-") if word)


def document_sha256(config: CollectionConfig, raw: bytes) -> str:
    """Of the raw file plus the chunking settings, so changing either re-ingests the page."""
    settings = json.dumps(
        {"size": config.chunk_size, "overlap": config.chunk_overlap, "version": CONTENT_VERSION}, sort_keys=True
    )
    return hashlib.sha256(raw + settings.encode()).hexdigest()


def parse_document(config: CollectionConfig, slug: str, raw: bytes) -> SourceDocument:
    # Universal newlines, as Path.read_text does (the FAISS index was built
    # from read_text, and some files have Windows line endings).
    content = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    lines = content.splitlines()
    title = lines[0].strip() if config.title_source == "first_line" and lines and lines[0].strip() else ""
    title = title or _title_from_slug(slug)

    source_line = 1 if config.title_source == "first_line" else 0
    url = config.url_pattern.format(slug=slug)
    if config.url_source == "source_line" and len(lines) > source_line and lines[source_line].startswith("Source:"):
        url = lines[source_line].removeprefix("Source:").strip() or url

    return SourceDocument(
        collection=config.name,
        slug=slug,
        title=title,
        url=url,
        # Unchanged raw text: the same passages as the FAISS index (see
        # CONTENT_VERSION). The title is its first line anyway.
        text=content,
        sha256=document_sha256(config, raw),
    )


_NOT_SLUG = re.compile(r"[^a-z0-9]+")
# A heading that is itself a scripture reference: "Alma 32:21. Faith and Hope", "1 Nephi 3–4. ..."
_STARTS_WITH_REFERENCE = re.compile(
    r"^(?:%s|D&C) \d" % "|".join(re.escape(book) for book in sorted(set(BOOKS.values()), key=len, reverse=True))
)


def _slugify(text: str, limit: int = 60) -> str:
    return _NOT_SLUG.sub("-", text.lower()).strip("-")[:limit].strip("-")


def section_title(page_title: str, heading: str) -> str:
    """The section's title in source links.

    "Chapter 30: Alma 32–35" + "Alma 32:21. Faith and Hope" -> "Alma 32:21. Faith and Hope"
    "Chapter 30: Alma 32–35" + "Introduction"                -> "Chapter 30: Alma 32–35 — Introduction"
    "Lesson 5 Class Preparation Material: The Coming Forth..." + "Section 1: How did..."
                                                             -> "Lesson 5, Section 1: How did..."
    "Liberty Jail" + ""                                      -> "Liberty Jail"
    """
    # The Restoration lessons' titles all say "Lesson N Class Preparation Material: ...".
    short = page_title.replace(" Class Preparation Material", "")
    if not heading:
        return short
    if _STARTS_WITH_REFERENCE.match(heading):
        return heading
    if heading.startswith("Section ") and ":" in short:
        return f"{short.split(':', 1)[0]}, {heading}"
    return f"{short} — {heading}"


def section_documents(config: CollectionConfig, slug: str, raw: bytes) -> list[SourceDocument]:
    """One document per section of a sectioned page (see sections.py), each
    with its own title, deep link and scripture references. The first
    section keeps the page's slug, the others are "<page>--<heading>"."""
    title, url, sections = parse_raw(raw.decode("utf-8"))
    title = title or _title_from_slug(slug)
    url = url or config.url_pattern.format(slug=slug)
    documents, used = [], set()
    for index, section in enumerate(sections):
        section_slug = slug if index == 0 else f"{slug}--{_slugify(section.heading) or section.anchor}"
        if section_slug in used:
            section_slug = f"{section_slug}-{section.anchor}"
        used.add(section_slug)
        text = section.text
        if section.scriptures:
            text = f"Scriptures: {'; '.join(section.scriptures)}\n\n{text}"
        prefix = f"{config.label}, {title}" + (f" — {section.heading}" if section.heading else "")
        metadata = {"section": section.heading, "anchor": section.anchor, "scriptures": section.scriptures}
        hashed = json.dumps({"prefix": prefix, "text": text, "metadata": metadata}, sort_keys=True).encode()
        documents.append(
            SourceDocument(
                collection=config.name,
                slug=section_slug,
                title=section_title(title, section.heading),
                url=section_url(url, section.anchor, first=index == 0),
                text=text,
                sha256=document_sha256(config, hashed),
                chunk_prefix=prefix,
                metadata=metadata,
            )
        )
    return documents


def load_documents(config: CollectionConfig) -> list[SourceDocument]:
    if not config.raw_path.is_dir():
        raise FileNotFoundError(f"{config.name}: raw_path {config.raw_path} doesn't exist")
    paths = sorted(config.raw_path.glob("*.txt"))
    if config.sections:
        return [doc for path in paths for doc in section_documents(config, path.stem, path.read_bytes())]
    return [parse_document(config, path.stem, path.read_bytes()) for path in paths]


def scan_documents(config: CollectionConfig) -> list[DocumentRef]:
    """Every page in the collection as a DocumentRef: each file is read once
    for its hash and not kept (load() reads it again when it's ingested).
    Sectioned collections are small (hundreds of pages), so their section
    documents are simply built up front."""
    if config.sections:
        return load_documents(config)
    if not config.raw_path.is_dir():
        raise FileNotFoundError(f"{config.name}: raw_path {config.raw_path} doesn't exist")
    return [
        DocumentRef(config, path.stem, path, document_sha256(config, path.read_bytes()))
        for path in sorted(config.raw_path.glob("*.txt"))
    ]
