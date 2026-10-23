"""Incremental ingestion: plan, chunk, embed, write.

A run compares the documents on disk with what the store already holds
(by sha256), then only re-chunks and re-embeds new or changed documents and
deletes the documents (and their passages) that disappeared from disk.

Pages are read, chunked, embedded and written one batch at a time, so a
run's memory doesn't grow with the number of pages.

An interrupted run can't leave a document looking up to date. Over a direct
Postgres connection each batch is one transaction (documents, passages and
sha256 together), so a crash rolls the batch back. Over REST a changed
document's sha256 is cleared first, its passages are replaced, and only then
is the new sha256 recorded. Either way the next run redoes what's missing.
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
    # Optional: write_batch(documents, sha256, chunk_counts, rows_for) upserts the
    # documents, then replaces their passages with rows_for(ids), all in one
    # transaction (SqlRagStore).


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
    """Cut each document into passages of at most chunk_size characters.

    A document with a chunk_prefix (a section: "Institute Manual (Book of
    Mormon), Chapter 30: Alma 32–35 — Alma 32:21. Faith and Hope") gets it
    on every passage, so each passage says where it's from; the text is cut
    smaller to leave room for it.
    """
    chunks = []
    splitters: dict[tuple[str, int], RecursiveCharacterTextSplitter] = {}
    for doc in documents:
        config = configs[doc.collection]
        prefix = f"{doc.chunk_prefix}\n\n" if doc.chunk_prefix else ""
        size = max(config.chunk_size - len(prefix), config.chunk_size // 2)
        splitter = splitters.get((doc.collection, size))
        if splitter is None:
            splitter = splitters[(doc.collection, size)] = RecursiveCharacterTextSplitter(
                chunk_size=size, chunk_overlap=min(config.chunk_overlap, size // 4)
            )
        for index, text in enumerate(splitter.split_text(doc.text)):
            chunks.append(Chunk(document=doc, index=index, content=prefix + text))
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
    seconds: float = 0.0

    @property
    def pages_per_minute(self) -> float:
        return (self.new + self.changed) / self.seconds * 60 if self.seconds else 0.0


def format_progress(done: int, total: int, passages: int, elapsed: float) -> str:
    """e.g. "300/5073 pages (612 passages) | 540 pages/min | ETA 8m50s"."""
    rate = done / elapsed * 60 if elapsed > 0 else 0.0
    remaining = (total - done) / rate * 60 if rate else 0.0
    minutes, seconds = divmod(int(remaining), 60)
    return f"{done}/{total} pages ({passages:,} passages) | {rate:.0f} pages/min | ETA {minutes}m{seconds:02d}s"


def _batches(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_ingestion(
    config: IngestConfig,
    documents: Sequence,
    store: RagStore,
    embed: Callable[[list[str]], list[list[float]]] | None,
    dry_run: bool,
    resolve_url: Callable[[SourceDocument], str] | None = None,
    write_batch_docs: int = 100,
    already_embedded: Callable[[str], bool] | None = None,
    max_documents: int | None = None,
    force: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> RunReport:
    """documents: SourceDocuments, or DocumentRefs (documents.scan_documents),
    which are only read in full one batch at a time, so memory stays flat
    however many pages there are. force: re-ingest unchanged pages too."""
    configs = {c.name: c for c in config.collections}
    stored = store.existing_documents([c.name for c in config.collections])
    plan = plan_run(documents, stored)
    if force:
        plan.changed, plan.unchanged = plan.changed + plan.unchanged, []
    # Optionally cap one run (removals are always done in full); the next run
    # carries on with the rest, since finished documents count as unchanged.
    if max_documents is not None:
        limited = {d.key for d in plan.to_ingest[:max_documents]}
        plan.new = [d for d in plan.new if d.key in limited]
        plan.changed = [d for d in plan.changed if d.key in limited]

    # Count passages and tokens one batch at a time (nothing is kept).
    # Passages whose embedding is already known (reused) cost nothing.
    chunk_total = tokens = 0
    for batch in _batches(plan.to_ingest, write_batch_docs):
        chunks = chunk_documents([doc.load() for doc in batch], configs)
        chunk_total += len(chunks)
        to_pay = [c.content for c in chunks if not (already_embedded and already_embedded(c.content))]
        tokens += count_tokens(to_pay, config.tokenizer) if to_pay else 0
    report = RunReport(
        documents_on_disk=len(documents),
        new=len(plan.new),
        changed=len(plan.changed),
        unchanged=len(plan.unchanged),
        removed=len(plan.removed),
        chunks_to_embed=chunk_total,
        tokens_to_embed=tokens,
        estimated_usd=tokens / 1_000_000 * config.usd_per_million_tokens,
    )
    if dry_run:
        return report
    if embed is None:
        raise ValueError("an embed function is needed for a real run")

    started = clock()
    if plan.removed:
        store.delete_documents([doc.id for doc in plan.removed.values()])

    # A store that can write a whole batch in one transaction (SqlRagStore)
    # makes a crash roll the batch back; otherwise the sha256 ordering below
    # makes the next run redo it.
    write_batch = getattr(store, "write_batch", None)
    to_ingest = plan.to_ingest
    done = passages = 0
    for refs in _batches(to_ingest, write_batch_docs):
        batch = [ref.load() for ref in refs]
        if resolve_url:
            batch = [
                doc.with_url(resolve_url(doc)) if configs[doc.collection].resolve_redirects else doc
                for doc in batch
            ]
        batch_chunks = chunk_documents(batch, configs)
        vectors = embed_with_retries([c.content for c in batch_chunks], embed)
        counts = {doc.key: 0 for doc in batch}
        for chunk in batch_chunks:
            counts[chunk.document.key] += 1

        def rows_for(ids: dict[DocKey, int]) -> list[dict]:
            return [
                {
                    "document_id": ids[chunk.document.key],
                    "collection": chunk.document.collection,
                    "chunk_index": chunk.index,
                    "content": chunk.content,
                    "embedding": vector,
                    "metadata": chunk.document.metadata,
                }
                for chunk, vector in zip(batch_chunks, vectors, strict=True)
            ]

        if write_batch:
            write_batch(batch, {doc.key: doc.sha256 for doc in batch}, counts, rows_for)
        else:
            # 1. Mark the batch as in progress (sha256 '') so a crash re-ingests it.
            ids = store.upsert_documents(batch, {doc.key: "" for doc in batch}, counts)
            # 2. Replace their passages.
            store.replace_chunks(list(ids.values()), rows_for(ids))
            # 3. Only now record the new sha256.
            store.upsert_documents(batch, {doc.key: doc.sha256 for doc in batch}, counts)
        done += len(batch)
        passages += len(batch_chunks)
        logger.info("Ingested %s", format_progress(done, len(to_ingest), passages, clock() - started))

    report.written = True
    report.seconds = clock() - started
    return report
