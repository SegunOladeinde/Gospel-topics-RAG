-- 001: rate-limit log table used by src/backend/main.py.
--
-- Equivalent to what init_db() (src/backend/database.py) creates at startup;
-- run it once in the Supabase SQL editor if you'd rather create the table
-- yourself. Safe to re-run. Unlike init_db(), it also adds the indexes to a
-- table that already exists, because create_all() skips existing tables.

CREATE TABLE IF NOT EXISTS public.user_query_logs (
    -- The backend supplies its own 32-char hex id; the default only matters
    -- for rows inserted by hand.
    id              VARCHAR     PRIMARY KEY DEFAULT replace(gen_random_uuid()::text, '-', ''),
    user_identifier VARCHAR     NOT NULL,   -- Clerk user id, or client IP for guests
    is_guest        BOOLEAN     NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_user_query_logs_user_identifier
    ON public.user_query_logs (user_identifier);

-- Covers both rate-limit queries: guests filter on (user_identifier,
-- is_guest); signed-in users also filter on created_at >= now() - 24h.
CREATE INDEX IF NOT EXISTS ix_user_query_logs_rate_limit
    ON public.user_query_logs (user_identifier, is_guest, created_at);

-- Supabase exposes the public schema through its REST API. With RLS on and
-- no policies, the anon/authenticated keys can't read or write this table,
-- while the backend (connecting as the table owner, e.g. `postgres`) is
-- unaffected.
ALTER TABLE public.user_query_logs ENABLE ROW LEVEL SECURITY;
