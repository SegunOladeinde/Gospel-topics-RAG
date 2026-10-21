"""Raw .txt files -> SourceDocuments with their real title and URL."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from src.backend.ingestion.config import CollectionConfig

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


def load_documents(config: CollectionConfig) -> list[SourceDocument]:
    if not config.raw_path.is_dir():
        raise FileNotFoundError(f"{config.name}: raw_path {config.raw_path} doesn't exist")
    return [
        parse_document(config, path.stem, path.read_bytes())
        for path in sorted(config.raw_path.glob("*.txt"))
    ]


def scan_documents(config: CollectionConfig) -> list[DocumentRef]:
    """Every page in the collection as a DocumentRef: each file is read once
    for its hash and not kept (load() reads it again when it's ingested)."""
    if not config.raw_path.is_dir():
        raise FileNotFoundError(f"{config.name}: raw_path {config.raw_path} doesn't exist")
    return [
        DocumentRef(config, path.stem, path, document_sha256(config, path.read_bytes()))
        for path in sorted(config.raw_path.glob("*.txt"))
    ]
