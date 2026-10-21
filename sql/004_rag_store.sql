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

-- Keep updated_at current on every update, whoever writes the row (REST
-- upserts didn't set it). The app's BM25 refresh check compares it, with the
-- passage count and highest passage id, to spot changes to the store.
CREATE OR REPLACE FUNCTION public.rag_documents_touch()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

CREATE OR REPLACE TRIGGER rag_documents_touch
    BEFORE UPDATE ON public.rag_documents
    FOR EACH ROW EXECUTE FUNCTION public.rag_documents_touch();

-- Semantic + full-text search fused by weighted reciprocal rank, exactly like
-- the FAISS/BM25 HybridRetriever: each passage scores
-- sum(weight / (rrf_k + rank)) over the two top-match_count lists.
--
-- query_text is a ready-made OR query of content words ("baptism | covenant"),
-- already stripped of stopwords, instruction words, collection names and
-- generic verbs by the caller (src/backend/retrieval/keyword.py). Empty means
-- semantic search only.
--
-- Full-text ranking weights each word by its IDF (how rare it is among the
-- passages, counted live through the GIN index) times ts_rank. Plain
-- ts_rank/ts_rank_cd treat every word alike, so for "why is tithing ten
-- percent" short pages titled "Ten" outranked the tithing pages; BM25 (the
-- FAISS backend's keyword search) avoided that with IDF too.
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
    terms AS (
        -- The query's words as stemmed lexemes ("tithing" -> "tith").
        SELECT DISTINCT l.lexeme
        FROM unnest(tsvector_to_array(to_tsvector('english', replace(coalesce(query_text, ''), '|', ' ')))) AS l (lexeme)
    ),
    weights AS (
        SELECT t.lexeme, ln((n.total - d.df + 0.5) / (d.df + 0.5) + 1) AS idf
        FROM terms t
        CROSS JOIN (
            SELECT count(*)::float AS total FROM rag_chunks
            WHERE collections IS NULL OR collection = ANY (collections)
        ) n
        CROSS JOIN LATERAL (
            SELECT count(*)::float AS df FROM rag_chunks c
            WHERE c.tsv @@ to_tsquery('simple', t.lexeme)
              AND (collections IS NULL OR c.collection = ANY (collections))
        ) d
    ),
    full_text AS (
        SELECT s.id, row_number() OVER (ORDER BY s.score DESC, s.id)::int AS rank
        FROM (
            SELECT c.id, sum(w.idf * ts_rank(c.tsv, to_tsquery('simple', w.lexeme))) AS score
            FROM weights w
            JOIN rag_chunks c ON c.tsv @@ to_tsquery('simple', w.lexeme)
            WHERE collections IS NULL OR c.collection = ANY (collections)
            GROUP BY c.id
        ) s
        ORDER BY s.score DESC, s.id
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

-- Semantic top-k plus the similarity of extra candidate passages, in one call.
-- Used by the pgvector_bm25 backend (src/backend/retrieval/pgvector.py): BM25
-- runs in the app's memory, then this returns the vector hits and how similar
-- each BM25 hit is to the question, so keyword-only hits that aren't about
-- the question at all can be dropped. similarity = 1 - cosine distance.
CREATE OR REPLACE FUNCTION public.match_chunks(
    query_embedding extensions.halfvec(1536),
    match_count     INTEGER  DEFAULT 4,
    candidate_ids   BIGINT[] DEFAULT '{}',
    collections     TEXT[]   DEFAULT NULL
)
RETURNS TABLE (
    chunk_id      BIGINT,
    content       TEXT,
    collection    TEXT,
    slug          TEXT,
    title         TEXT,
    url           TEXT,
    similarity    FLOAT,
    semantic_rank INTEGER
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
    wanted AS (
        SELECT id, rank FROM semantic
        UNION
        SELECT unnest(coalesce(candidate_ids, '{}')), NULL::int
    )
    SELECT c.id, c.content, d.collection, d.slug, d.title, d.url,
           1 - (c.embedding <=> query_embedding) AS similarity,
           min(w.rank) AS semantic_rank
    FROM wanted w
    JOIN rag_chunks c ON c.id = w.id
    JOIN rag_documents d ON d.id = c.document_id
    GROUP BY c.id, c.content, d.collection, d.slug, d.title, d.url, c.embedding
    ORDER BY min(w.rank) NULLS LAST, similarity DESC
$$;

REVOKE ALL ON FUNCTION public.match_chunks(extensions.halfvec, INTEGER, BIGINT[], TEXT[]) FROM PUBLIC;
DO $$
DECLARE r TEXT;
BEGIN
    FOREACH r IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
            EXECUTE format('REVOKE ALL ON FUNCTION public.match_chunks(extensions.halfvec, INTEGER, BIGINT[], TEXT[]) FROM %I', r);
        END IF;
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT EXECUTE ON FUNCTION public.match_chunks(extensions.halfvec, INTEGER, BIGINT[], TEXT[]) TO service_role;
    END IF;
END
$$;
