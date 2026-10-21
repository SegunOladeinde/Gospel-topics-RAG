"""data/collections.yaml: which collections to ingest and how."""

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "data" / "collections.yaml"


@dataclass(frozen=True)
class CollectionConfig:
    name: str
    label: str
    raw_path: Path
    url_pattern: str
    title_source: str = "first_line"  # or "slug"
    url_source: str = "source_line"  # or "pattern"
    resolve_redirects: bool = False
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass(frozen=True)
class IngestConfig:
    collections: list[CollectionConfig]
    embedding_model: str
    tokenizer: str
    usd_per_million_tokens: float


def load_config(path: Path = DEFAULT_CONFIG, root: Path = PROJECT_ROOT) -> IngestConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    collections = []
    for entry in data["collections"]:
        merged = {k: v for k, v in defaults.items() if k in ("chunk_size", "chunk_overlap")} | entry
        if merged.get("title_source", "first_line") not in ("first_line", "slug"):
            raise ValueError(f"{entry['name']}: title_source must be first_line or slug")
        if merged.get("url_source", "source_line") not in ("source_line", "pattern"):
            raise ValueError(f"{entry['name']}: url_source must be source_line or pattern")
        if "{slug}" not in merged["url_pattern"]:
            raise ValueError(f"{entry['name']}: url_pattern needs a {{slug}} placeholder")
        merged["raw_path"] = root / merged["raw_path"]
        collections.append(CollectionConfig(**merged))
    names = [c.name for c in collections]
    if len(names) != len(set(names)):
        raise ValueError("collection names must be unique")
    return IngestConfig(
        collections=collections,
        embedding_model=defaults.get("embedding_model", "text-embedding-3-small"),
        tokenizer=defaults.get("tokenizer", "cl100k_base"),
        usd_per_million_tokens=float(defaults.get("usd_per_million_tokens", 0.02)),
    )
