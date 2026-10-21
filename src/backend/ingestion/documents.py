"""Raw .txt files -> SourceDocuments with their real title and URL."""

import hashlib
import json
from dataclasses import dataclass, replace

from src.backend.ingestion.config import CollectionConfig

# Bump to force every document to re-ingest when chunking/content rules change.
CONTENT_VERSION = 1


@dataclass(frozen=True)
class SourceDocument:
    collection: str
    slug: str
    title: str
    url: str
    text: str  # title + page text, what gets chunked
    sha256: str  # of the raw file plus the chunking settings

    @property
    def key(self) -> tuple[str, str]:
        return (self.collection, self.slug)

    def with_url(self, url: str) -> "SourceDocument":
        return replace(self, url=url)


def _title_from_slug(slug: str) -> str:
    return " ".join(word.capitalize() for word in slug.split("-") if word)


def parse_document(config: CollectionConfig, slug: str, raw: bytes) -> SourceDocument:
    lines = raw.decode("utf-8").splitlines()
    title = lines[0].strip() if config.title_source == "first_line" and lines and lines[0].strip() else ""
    title = title or _title_from_slug(slug)

    body_start = 1 if config.title_source == "first_line" else 0
    url = config.url_pattern.format(slug=slug)
    if len(lines) > body_start and lines[body_start].startswith("Source:"):
        if config.url_source == "source_line":
            url = lines[body_start].removeprefix("Source:").strip() or url
        body_start += 1
    body = "\n".join(lines[body_start:]).strip()

    settings = json.dumps(
        {"size": config.chunk_size, "overlap": config.chunk_overlap, "version": CONTENT_VERSION}, sort_keys=True
    )
    return SourceDocument(
        collection=config.name,
        slug=slug,
        title=title,
        url=url,
        # The title leads the text so the first passage says what it's about
        # (the "Source:" line is dropped: a URL isn't useful to search).
        text=f"{title}\n\n{body}" if body else title,
        sha256=hashlib.sha256(raw + settings.encode()).hexdigest(),
    )


def load_documents(config: CollectionConfig) -> list[SourceDocument]:
    if not config.raw_path.is_dir():
        raise FileNotFoundError(f"{config.name}: raw_path {config.raw_path} doesn't exist")
    return [
        parse_document(config, path.stem, path.read_bytes())
        for path in sorted(config.raw_path.glob("*.txt"))
    ]
