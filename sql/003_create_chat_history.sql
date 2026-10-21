-- 003: saved chats for signed-in users, used by src/backend/history.py.
--
-- Equivalent to what init_db() (src/backend/database.py) creates at startup;
-- run it once in the Supabase SQL editor if you'd rather create the tables
-- yourself. Safe to re-run. Guests' chats are never stored.

CREATE TABLE IF NOT EXISTS public.chat_conversations (
    -- The backend supplies its own 32-char hex ids; the defaults only matter
    -- for rows inserted by hand.
    id         VARCHAR     PRIMARY KEY DEFAULT replace(gen_random_uuid()::text, '-', ''),
    user_id    VARCHAR     NOT NULL,   -- Clerk user id
    title      VARCHAR     NOT NULL,   -- first question, cut to 80 characters
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- The sidebar lists a user's chats newest first.
CREATE INDEX IF NOT EXISTS ix_chat_conversations_user_updated
    ON public.chat_conversations (user_id, updated_at);

CREATE TABLE IF NOT EXISTS public.chat_messages (
    id              VARCHAR     PRIMARY KEY DEFAULT replace(gen_random_uuid()::text, '-', ''),
    conversation_id VARCHAR     NOT NULL REFERENCES public.chat_conversations (id) ON DELETE CASCADE,
    role            VARCHAR     NOT NULL,   -- 'user' or 'assistant'
    content         TEXT        NOT NULL,
    sources         JSON        NOT NULL DEFAULT '[]',   -- source ids, e.g. "topical_guide/faith"
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_chat_messages_conversation_created
    ON public.chat_messages (conversation_id, created_at);

-- Same reasoning as 001: with RLS on and no policies, Supabase's public API
-- keys can't touch these tables; the backend (the table owner) is unaffected.
-- init_db() also turns this on at startup.
ALTER TABLE public.chat_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;
