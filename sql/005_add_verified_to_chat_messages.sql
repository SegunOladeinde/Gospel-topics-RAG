-- 005: mark answers whose claims were all checked against their sources.
--
-- The backend checks every claim in an answer against the passages it used
-- (src/backend/claims.py); answers that pass are saved with verified = true
-- and shown with "Verified against sources" when the chat is reopened.
-- init_db() (src/backend/database.py) adds this column at startup too; run
-- this in the Supabase SQL editor if you'd rather do it yourself. Safe to
-- re-run.

ALTER TABLE public.chat_messages
    ADD COLUMN IF NOT EXISTS verified BOOLEAN NOT NULL DEFAULT false;
