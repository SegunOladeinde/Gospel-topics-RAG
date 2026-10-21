"""Where ingested documents and passages are written.

SqlRagStore talks to Postgres directly (DATABASE_URL); RestRagStore goes
through Supabase's REST API with the service-role key, for machines that
only have SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY. Both write the same
rag_documents / rag_chunks rows (sql/004_rag_store.sql).
"""

from collections.abc import Sequence

import httpx
from sqlalchemy import text

from src.backend.ingestion.documents import SourceDocument
from src.backend.ingestion.pipeline import DocKey, StoredDocument


def vector_literal(vector: Sequence[float]) -> str:
    """pgvector's text format, e.g. "[0.1,0.2]" (works for vector and halfvec)."""
    return "[" + ",".join(f"{x:.8g}" for x in vector) + "]"


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

    def upsert_documents(self, documents, sha256, chunk_counts) -> dict[DocKey, int]:
        ids = {}
        with self.session_factory() as db:
            for doc in documents:
                row = _document_row(doc, sha256[doc.key], chunk_counts.get(doc.key, 0))
                ids[doc.key] = db.execute(
                    text(
                        "INSERT INTO rag_documents (collection, slug, title, url, content_sha256, chunk_count) "
                        "VALUES (:collection, :slug, :title, :url, :content_sha256, :chunk_count) "
                        "ON CONFLICT (collection, slug) DO UPDATE SET title = EXCLUDED.title, url = EXCLUDED.url, "
                        "content_sha256 = EXCLUDED.content_sha256, chunk_count = EXCLUDED.chunk_count, "
                        "updated_at = now() RETURNING id"
                    ),
                    row,
                ).scalar_one()
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

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        response = self.client.request(method, f"{self.base}/{path}", headers=self.headers | kwargs.pop("headers", {}), **kwargs)
        response.raise_for_status()
        return response

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
