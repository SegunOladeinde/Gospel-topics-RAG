-- 002: flag admin queries in user_query_logs.
--
-- Admins (ADMIN_EMAILS on the frontend) skip the rate limit. Their queries
-- are still logged, with is_admin = true, and the rate-limit count ignores
-- those rows. init_db() (src/backend/database.py) adds this column at
-- startup too; run this in the Supabase SQL editor if you'd rather do it
-- yourself. Safe to re-run.

ALTER TABLE public.user_query_logs
    ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT false;
