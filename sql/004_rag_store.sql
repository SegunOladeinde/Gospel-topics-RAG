-- 004: passage store for retrieval (pgvector + Postgres full-text search).
--
-- Replaces the FAISS/BM25 files in data/index/ once RETRIEVER_BACKEND=pgvector.
-- Filled by scripts/ingest.py; queried through hybrid_search() by
-- src/backend/retrieval/pgvector.py. init_db() runs this file at startup on
-- Postgres, so every statement must stay safe to re-run.
--
-- Embeddings are halfvec(1536): text-embedding-3-small vectors survive the
-- float16 round trip exactly (measured: identical top-20 neighbours for every
-- eval question), at half the storage of vector(1536).

CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- One row per source page (e.g. topical_guide/faith).
CREATE TABLE IF NOT EXISTS public.rag_documents (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    collection     TEXT        NOT NULL,          -- e.g. topical_guide (data/collections.yaml)
    slug           TEXT        NOT NULL,          -- file stem, e.g. aaron-descendants-of
    title          TEXT        NOT NULL,          -- the page's real title, e.g. "Aaron, Descendants of"
    url            TEXT,                          -- canonical page URL (redirects resolved)
    content_sha256 TEXT        NOT NULL,          -- '' while a re-ingest is in progress
    chunk_count    INTEGER     NOT NULL DEFAULT 0,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (collection, slug)
);

-- The searchable passages. collection is repeated here so collection filters
-- don't need a join (which would stop Postgres using the HNSW index).
CREATE TABLE IF NOT EXISTS public.rag_chunks (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT  NOT NULL REFERENCES public.rag_documents (id) ON DELETE CASCADE,
    collection  TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT    NOT NULL,
    embedding   extensions.halfvec(1536) NOT NULL,
    tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    metadata    JSONB   NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw
    ON public.rag_chunks USING hnsw (embedding extensions.halfvec_cosine_ops);
CREATE INDEX IF NOT EXISTS rag_chunks_tsv_gin ON public.rag_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS rag_chunks_document ON public.rag_chunks (document_id);

-- Same as the other app tables: RLS on, no policies, so Supabase's public API
-- keys can't read them. The backend connects as the table owner.
ALTER TABLE public.rag_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rag_chunks ENABLE ROW LEVEL SECURITY;

-- Semantic + full-text search fused by weighted reciprocal rank, exactly like
-- the FAISS/BM25 HybridRetriever: each passage scores
-- sum(weight / (rrf_k + rank)) over the two top-match_count lists.
--
-- query_text is a ready-made OR query of content words ("baptism | covenant"),
-- already stripped of stopwords, instruction words, collection names and
-- generic verbs by the caller (src/backend/retrieval/keyword.py). Empty means
-- semantic search only.
CREATE OR REPLACE FUNCTION public.hybrid_search(
    query_embedding  extensions.halfvec(1536),
    query_text       TEXT,
    match_count      INTEGER DEFAULT 4,
    collections      TEXT[]  DEFAULT NULL,
    rrf_k            INTEGER DEFAULT 60,
    semantic_weight  FLOAT   DEFAULT 0.5,
    full_text_weight FLOAT   DEFAULT 0.5
)
RETURNS TABLE (
    chunk_id       BIGINT,
    content        TEXT,
    collection     TEXT,
    slug           TEXT,
    title          TEXT,
    url            TEXT,
    score          FLOAT,
    semantic_rank  INTEGER,
    full_text_rank INTEGER
)
LANGUAGE sql STABLE
SET search_path = public, extensions
AS $$
    WITH semantic AS (
        SELECT c.id, row_number() OVER (ORDER BY c.embedding <=> query_embedding)::int AS rank
        FROM rag_chunks c
        WHERE collections IS NULL OR c.collection = ANY (collections)
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_count
    ),
    parsed AS (
        SELECT CASE WHEN coalesce(btrim(query_text), '') = '' THEN NULL
                    ELSE to_tsquery('english', query_text) END AS q
    ),
    full_text AS (
        SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, p.q) DESC, c.id)::int AS rank
        FROM rag_chunks c, parsed p
        WHERE p.q IS NOT NULL
          AND c.tsv @@ p.q
          AND (collections IS NULL OR c.collection = ANY (collections))
        ORDER BY ts_rank_cd(c.tsv, p.q) DESC, c.id
        LIMIT match_count
    )
    SELECT c.id, c.content, d.collection, d.slug, d.title, d.url,
           coalesce(semantic_weight / (rrf_k + s.rank), 0)
             + coalesce(full_text_weight / (rrf_k + f.rank), 0) AS score,
           s.rank, f.rank
    FROM semantic s
    FULL OUTER JOIN full_text f ON f.id = s.id
    JOIN rag_chunks c ON c.id = coalesce(s.id, f.id)
    JOIN rag_documents d ON d.id = c.document_id
    -- Ties: semantic hits first (in their order), then full-text ones, like
    -- the Python fusion's first-seen order.
    ORDER BY score DESC, coalesce(s.rank, 2147483647), coalesce(f.rank, 2147483647)
$$;

-- Functions in public are callable by everyone by default, which on Supabase
-- would expose this through the public REST API. Only the backend (postgres)
-- and the ingestion/eval scripts (service_role) may call it.
REVOKE ALL ON FUNCTION public.hybrid_search(extensions.halfvec, TEXT, INTEGER, TEXT[], INTEGER, FLOAT, FLOAT) FROM PUBLIC;
DO $$
DECLARE r TEXT;
BEGIN
    FOREACH r IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
            EXECUTE format('REVOKE ALL ON FUNCTION public.hybrid_search(extensions.halfvec, TEXT, INTEGER, TEXT[], INTEGER, FLOAT, FLOAT) FROM %I', r);
        END IF;
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT EXECUTE ON FUNCTION public.hybrid_search(extensions.halfvec, TEXT, INTEGER, TEXT[], INTEGER, FLOAT, FLOAT) TO service_role;
    END IF;
END
$$;
