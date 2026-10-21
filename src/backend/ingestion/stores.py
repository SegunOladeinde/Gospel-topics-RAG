"""Where ingested documents and passages are written.

SqlRagStore talks to Postgres directly (DATABASE_URL), writing each batch
in one transaction with passages streamed in by binary COPY; RestRagStore goes
through Supabase's REST API with the service-role key, for machines that
only have SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY. Both write the same
rag_documents / rag_chunks rows (sql/004_rag_store.sql).
"""

import struct
import time
from collections.abc import Callable, Sequence

import httpx
import numpy as np
from sqlalchemy import text

from src.backend.ingestion.documents import SourceDocument
from src.backend.ingestion.pipeline import DocKey, StoredDocument


def vector_literal(vector: Sequence[float]) -> str:
    """pgvector's text format for a halfvec column, e.g. "[0.1235,-0.0423]".

    Each value is written as the shortest decimal that reads back as the same
    float16, which is all halfvec keeps: about half the bytes of 8 significant
    digits, and uploads are the slow part of an ingestion run.
    """
    halves = np.asarray(vector, dtype=np.float16)
    return "[" + ",".join(np.format_float_positional(x, unique=True, trim="-") for x in halves) + "]"


def halfvec_binary(vector: Sequence[float]) -> bytes:
    """pgvector's binary wire format for halfvec (what halfvec_recv reads):
    dimensions and an unused field as big-endian int16, then each value as a
    big-endian float16. A third of the bytes of vector_literal's text."""
    halves = np.asarray(vector, dtype=">f2")
    return struct.pack(">HH", halves.size, 0) + halves.tobytes()


def _document_row(doc: SourceDocument, sha256: str, chunk_count: int) -> dict:
    return {
        "collection": doc.collection,
        "slug": doc.slug,
        "title": doc.title,
        "url": doc.url,
        "content_sha256": sha256,
        "chunk_count": chunk_count,
    }


class SqlRagStore:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def existing_documents(self, collections: Sequence[str]) -> dict[DocKey, StoredDocument]:
        with self.session_factory() as db:
            rows = db.execute(
                text("SELECT id, collection, slug, content_sha256 FROM rag_documents WHERE collection = ANY(:c)"),
                {"c": list(collections)},
            ).all()
        return {(r.collection, r.slug): StoredDocument(r.id, r.content_sha256) for r in rows}

    _UPSERT = text(
        "INSERT INTO rag_documents (collection, slug, title, url, content_sha256, chunk_count) "
        "SELECT * FROM unnest(CAST(:collection AS TEXT[]), CAST(:slug AS TEXT[]), CAST(:title AS TEXT[]), "
        "CAST(:url AS TEXT[]), CAST(:content_sha256 AS TEXT[]), CAST(:chunk_count AS INTEGER[])) "
        "ON CONFLICT (collection, slug) DO UPDATE SET title = EXCLUDED.title, url = EXCLUDED.url, "
        "content_sha256 = EXCLUDED.content_sha256, chunk_count = EXCLUDED.chunk_count, "
        "updated_at = now() RETURNING id, collection, slug"
    )

    @staticmethod
    def _upsert(db, documents, sha256, chunk_counts) -> dict[DocKey, int]:
        # One statement per batch (arrays unnested), not one per document:
        # far fewer round trips on a slow link.
        rows = [_document_row(doc, sha256[doc.key], chunk_counts.get(doc.key, 0)) for doc in documents]
        if not rows:
            return {}
        columns = {name: [row[name] for row in rows] for name in rows[0]}
        returned = db.execute(SqlRagStore._UPSERT, columns).all()
        return {(r.collection, r.slug): r.id for r in returned}

    def upsert_documents(self, documents, sha256, chunk_counts) -> dict[DocKey, int]:
        with self.session_factory() as db:
            ids = self._upsert(db, documents, sha256, chunk_counts)
            db.commit()
        return ids

    _COPY_CHUNKS = (
        "COPY rag_chunks (document_id, collection, chunk_index, content, embedding) FROM STDIN (FORMAT BINARY)"
    )

    def write_batch(
        self,
        documents: Sequence[SourceDocument],
        sha256: dict[DocKey, str],
        chunk_counts: dict[DocKey, int],
        rows_for: Callable[[dict[DocKey, int]], Sequence[dict]],
    ) -> dict[DocKey, int]:
        """Upsert the documents (with their final sha256) and replace their
        passages in one transaction: if anything fails, none of it is kept and
        the next run redoes the batch. Passages go in with binary COPY, one
        stream instead of a statement per row, at a third of the bytes."""
        with self.session_factory() as db:
            ids = self._upsert(db, documents, sha256, chunk_counts)
            db.execute(text("DELETE FROM rag_chunks WHERE document_id = ANY(:ids)"), {"ids": list(ids.values())})
            # The psycopg connection under this session, in the same transaction.
            raw = db.connection().connection.driver_connection
            with raw.cursor() as cursor, cursor.copy(self._COPY_CHUNKS) as copy:
                # Binary COPY hands each value to the column's binary input
                # function; "bytea" just passes the halfvec bytes through.
                copy.set_types(["int8", "text", "int4", "text", "bytea"])
                for row in rows_for(ids):
                    copy.write_row((row["document_id"], row["collection"], row["chunk_index"], row["content"],
                                    halfvec_binary(row["embedding"])))
            db.commit()
        return ids

    def replace_chunks(self, document_ids, rows) -> None:
        with self.session_factory() as db:
            db.execute(text("DELETE FROM rag_chunks WHERE document_id = ANY(:ids)"), {"ids": list(document_ids)})
            if rows:
                db.execute(
                    text(
                        "INSERT INTO rag_chunks (document_id, collection, chunk_index, content, embedding) "
                        "VALUES (:document_id, :collection, :chunk_index, :content, CAST(:embedding AS extensions.halfvec))"
                    ),
                    [{**r, "embedding": vector_literal(r["embedding"])} for r in rows],
                )
            db.commit()

    def delete_documents(self, document_ids) -> None:
        with self.session_factory() as db:
            db.execute(text("DELETE FROM rag_documents WHERE id = ANY(:ids)"), {"ids": list(document_ids)})
            db.commit()


class RestRagStore:
    """Supabase REST (PostgREST) with the service-role key, which bypasses RLS."""

    def __init__(self, supabase_url: str, service_key: str, client: httpx.Client | None = None):
        self.base = f"{supabase_url.rstrip('/')}/rest/v1"
        self.client = client or httpx.Client(timeout=120)
        self.headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}"}

    def _request(self, method: str, path: str, attempts: int = 5, **kwargs) -> httpx.Response:
        """One REST call, retried with backoff on dropped connections, timeouts,
        429 and 5xx (all seen from Supabase during long runs). Every write here
        is idempotent (upserts, delete-then-insert per batch), so retrying is safe."""
        headers = self.headers | kwargs.pop("headers", {})
        for attempt in range(1, attempts + 1):
            try:
                response = self.client.request(method, f"{self.base}/{path}", headers=headers, **kwargs)
                if response.status_code != 429 and response.status_code < 500:
                    response.raise_for_status()
                    return response
                error: Exception = httpx.HTTPStatusError(
                    f"{response.status_code} from Supabase", request=response.request, response=response
                )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                error = exc
            if attempt == attempts:
                raise error
            time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")

    def existing_documents(self, collections: Sequence[str]) -> dict[DocKey, StoredDocument]:
        found, start, page = {}, 0, 1000
        names = ",".join(f'"{c}"' for c in collections)
        while True:
            rows = self._request(
                "GET", "rag_documents",
                params={"select": "id,collection,slug,content_sha256", "collection": f"in.({names})", "order": "id"},
                headers={"Range": f"{start}-{start + page - 1}"},
            ).json()
            found.update({(r["collection"], r["slug"]): StoredDocument(r["id"], r["content_sha256"]) for r in rows})
            if len(rows) < page:
                return found
            start += page

    def upsert_documents(self, documents, sha256, chunk_counts) -> dict[DocKey, int]:
        rows = [_document_row(doc, sha256[doc.key], chunk_counts.get(doc.key, 0)) for doc in documents]
        returned = self._request(
            "POST", "rag_documents",
            params={"on_conflict": "collection,slug", "select": "id,collection,slug"},
            json=rows,
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        ).json()
        return {(r["collection"], r["slug"]): r["id"] for r in returned}

    def replace_chunks(self, document_ids, rows, batch_rows: int = 100) -> None:
        for start in range(0, len(document_ids), 100):
            ids = ",".join(str(i) for i in document_ids[start : start + 100])
            self._request("DELETE", "rag_chunks", params={"document_id": f"in.({ids})"})
        for start in range(0, len(rows), batch_rows):
            batch = [{**r, "embedding": vector_literal(r["embedding"])} for r in rows[start : start + batch_rows]]
            self._request("POST", "rag_chunks", json=batch, headers={"Prefer": "return=minimal"})

    def delete_documents(self, document_ids) -> None:
        for start in range(0, len(document_ids), 100):
            ids = ",".join(str(i) for i in document_ids[start : start + 100])
            self._request("DELETE", "rag_documents", params={"id": f"in.({ids})"})
