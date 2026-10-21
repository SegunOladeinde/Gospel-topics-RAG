"""Incremental ingestion: plan, chunk, embed, write.

A run compares the documents on disk with what the store already holds
(by sha256), then only re-chunks and re-embeds new or changed documents and
deletes the documents (and their passages) that disappeared from disk.

Writes are ordered so an interrupted run can't leave a document looking
up to date: a changed document's sha256 is cleared first, its passages are
replaced, and only then is the new sha256 recorded. The next run redoes
anything left half-written.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.backend.ingestion.config import CollectionConfig, IngestConfig
from src.backend.ingestion.documents import SourceDocument

logger = logging.getLogger(__name__)

DocKey = tuple[str, str]


@dataclass
class StoredDocument:
    id: int
    sha256: str


@dataclass
class Chunk:
    document: SourceDocument
    index: int
    content: str


class RagStore(Protocol):
    """Where documents and passages live (see stores.py)."""

    def existing_documents(self, collections: Sequence[str]) -> dict[DocKey, StoredDocument]: ...
    def upsert_documents(self, documents: Sequence[SourceDocument], sha256: dict[DocKey, str],
                         chunk_counts: dict[DocKey, int]) -> dict[DocKey, int]: ...
    def replace_chunks(self, document_ids: Sequence[int], rows: Sequence[dict]) -> None: ...
    def delete_documents(self, document_ids: Sequence[int]) -> None: ...


@dataclass
class Plan:
    new: list[SourceDocument] = field(default_factory=list)
    changed: list[SourceDocument] = field(default_factory=list)
    unchanged: list[SourceDocument] = field(default_factory=list)
    removed: dict[DocKey, StoredDocument] = field(default_factory=dict)

    @property
    def to_ingest(self) -> list[SourceDocument]:
        return self.new + self.changed


def plan_run(on_disk: Sequence[SourceDocument], stored: dict[DocKey, StoredDocument]) -> Plan:
    plan = Plan()
    seen = set()
    for doc in on_disk:
        seen.add(doc.key)
        existing = stored.get(doc.key)
        if existing is None:
            plan.new.append(doc)
        elif existing.sha256 != doc.sha256:  # includes '' left by an interrupted run
            plan.changed.append(doc)
        else:
            plan.unchanged.append(doc)
    plan.removed = {key: doc for key, doc in stored.items() if key not in seen}
    return plan


def chunk_documents(documents: Sequence[SourceDocument], configs: dict[str, CollectionConfig]) -> list[Chunk]:
    chunks = []
    splitters: dict[str, RecursiveCharacterTextSplitter] = {}
    for doc in documents:
        config = configs[doc.collection]
        splitter = splitters.setdefault(
            doc.collection,
            RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap),
        )
        for index, text in enumerate(splitter.split_text(doc.text)):
            chunks.append(Chunk(document=doc, index=index, content=text))
    return chunks


def count_tokens(texts: Sequence[str], tokenizer: str) -> int:
    import tiktoken

    encoding = tiktoken.get_encoding(tokenizer)
    return sum(len(tokens) for tokens in encoding.encode_batch(list(texts)))


def embed_with_retries(
    texts: Sequence[str],
    embed: Callable[[list[str]], list[list[float]]],
    batch_size: int = 256,
    max_attempts: int = 6,
    sleep: Callable[[float], None] = time.sleep,
) -> list[list[float]]:
    """Embed in batches; retry each failed batch with exponential backoff."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        for attempt in range(1, max_attempts + 1):
            try:
                vectors.extend(embed(batch))
                break
            except Exception as exc:  # noqa: BLE001 - rate limits, timeouts, 5xx
                if attempt == max_attempts:
                    raise
                delay = min(2 ** attempt, 60)
                logger.warning("Embedding batch at %d failed (%s); retrying in %ss", start, exc, delay)
                sleep(delay)
    return vectors


@dataclass
class RunReport:
    documents_on_disk: int
    new: int
    changed: int
    unchanged: int
    removed: int
    chunks_to_embed: int
    tokens_to_embed: int
    estimated_usd: float
    written: bool = False


def run_ingestion(
    config: IngestConfig,
    documents: Sequence[SourceDocument],
    store: RagStore,
    embed: Callable[[list[str]], list[list[float]]] | None,
    dry_run: bool,
    resolve_url: Callable[[SourceDocument], str] | None = None,
    write_batch_docs: int = 100,
    already_embedded: Callable[[str], bool] | None = None,
    max_documents: int | None = None,
) -> RunReport:
    configs = {c.name: c for c in config.collections}
    stored = store.existing_documents([c.name for c in config.collections])
    plan = plan_run(documents, stored)
    # Optionally cap one run (removals are always done in full); the next run
    # carries on with the rest, since finished documents count as unchanged.
    if max_documents is not None:
        limited = plan.to_ingest[:max_documents]
        plan.new = [d for d in plan.new if d in limited]
        plan.changed = [d for d in plan.changed if d in limited]
    chunks = chunk_documents(plan.to_ingest, configs)
    # Passages whose embedding is already known (reused) cost nothing.
    to_pay = [c.content for c in chunks if not (already_embedded and already_embedded(c.content))]
    tokens = count_tokens(to_pay, config.tokenizer) if to_pay else 0
    report = RunReport(
        documents_on_disk=len(documents),
        new=len(plan.new),
        changed=len(plan.changed),
        unchanged=len(plan.unchanged),
        removed=len(plan.removed),
        chunks_to_embed=len(chunks),
        tokens_to_embed=tokens,
        estimated_usd=tokens / 1_000_000 * config.usd_per_million_tokens,
    )
    if dry_run:
        return report
    if embed is None:
        raise ValueError("an embed function is needed for a real run")

    if plan.removed:
        store.delete_documents([doc.id for doc in plan.removed.values()])

    by_doc: dict[DocKey, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.document.key, []).append(chunk)

    to_ingest = plan.to_ingest
    if resolve_url:
        to_ingest = [
            doc.with_url(resolve_url(doc)) if configs[doc.collection].resolve_redirects else doc
            for doc in to_ingest
        ]

    for start in range(0, len(to_ingest), write_batch_docs):
        batch = to_ingest[start : start + write_batch_docs]
        batch_chunks = [c for doc in batch for c in by_doc.get(doc.key, [])]
        vectors = embed_with_retries([c.content for c in batch_chunks], embed)
        counts = {doc.key: len(by_doc.get(doc.key, [])) for doc in batch}
        # 1. Mark the batch as in progress (sha256 '') so a crash re-ingests it.
        ids = store.upsert_documents(batch, {doc.key: "" for doc in batch}, counts)
        # 2. Replace their passages.
        rows = [
            {
                "document_id": ids[chunk.document.key],
                "collection": chunk.document.collection,
                "chunk_index": chunk.index,
                "content": chunk.content,
                "embedding": vector,
            }
            for chunk, vector in zip(batch_chunks, vectors, strict=True)
        ]
        store.replace_chunks(list(ids.values()), rows)
        # 3. Only now record the new sha256.
        store.upsert_documents(batch, {doc.key: doc.sha256 for doc in batch}, counts)
        logger.info("Ingested %d/%d documents", min(start + write_batch_docs, len(to_ingest)), len(to_ingest))

    report.written = True
    return report
