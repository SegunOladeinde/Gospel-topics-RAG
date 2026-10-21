"""Keeps pgvector_bm25's in-memory BM25 index in step with the store, without a restart.

Two triggers, both rebuilding in the background (see
PgvectorBm25Retriever.rebuild_keyword_index): queries keep using the old
index until the new one is swapped in.
- POST /admin/reindex (scripts/ingest.py calls it after a run)
- an automatic check every BM25_REFRESH_SECONDS (default 600): one small
  query comparing the passage count, highest passage id and latest document
  update with the ones the index was built from

Only one rebuild runs at a time; a trigger during a rebuild is a no-op.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Protocol

logger = logging.getLogger(__name__)


class Refreshable(Protocol):
    def is_stale(self) -> bool: ...
    def rebuild_keyword_index(self) -> int: ...


class KeywordIndexRefresher:
    def __init__(self, retriever: Refreshable, interval_seconds: float):
        self.retriever = retriever
        self.interval_seconds = interval_seconds
        self.refreshed_at: datetime | None = None
        self._lock = asyncio.Lock()
        self._watcher: asyncio.Task | None = None
        self._rebuild: asyncio.Task | None = None

    @property
    def rebuilding(self) -> bool:
        return self._lock.locked()

    async def rebuild(self) -> int | None:
        """Rebuild now; returns the passage count, or None if one is already running."""
        if self._lock.locked():
            return None
        async with self._lock:
            started = asyncio.get_running_loop().time()
            chunks = await asyncio.to_thread(self.retriever.rebuild_keyword_index)
            self.refreshed_at = datetime.now(UTC)
            logger.info(
                "Keyword index rebuilt: %d passages in %.1fs",
                chunks, asyncio.get_running_loop().time() - started,
            )
            return chunks

    def start_rebuild(self) -> bool:
        """Start a rebuild in the background; False if one is already running."""
        if self._lock.locked() or (self._rebuild and not self._rebuild.done()):
            return False
        self._rebuild = asyncio.create_task(self._rebuild_logged())
        return True

    async def _rebuild_logged(self) -> None:
        try:
            await self.rebuild()
        except Exception:  # noqa: BLE001 - the old index keeps serving
            logger.exception("Keyword index rebuild failed; still using the previous index.")

    async def check_once(self) -> bool:
        """Rebuild if the store changed; True if a rebuild ran."""
        if self._lock.locked() or not await asyncio.to_thread(self.retriever.is_stale):
            return False
        logger.info("Passage store changed: rebuilding the keyword index.")
        return await self.rebuild() is not None

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(self.interval_seconds)
            try:
                await self.check_once()
            except Exception:  # noqa: BLE001 - try again next interval
                logger.warning("Keyword index check failed; will retry.", exc_info=True)

    def start(self) -> None:
        if self.interval_seconds > 0:
            self._watcher = asyncio.create_task(self._watch())

    async def stop(self) -> None:
        for task in (self._watcher, self._rebuild):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
