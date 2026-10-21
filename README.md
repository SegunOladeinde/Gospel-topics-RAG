# LDS Gospel RAG System

## Project Overview
The LDS Gospel RAG (Retrieval-Augmented Generation) System is an intelligent, context-aware conversational AI that answers doctrinal questions by retrieving and synthesizing knowledge directly from approved LDS Gospel source texts (like the standard works, General Conference, and Gospel Topics essays). Designed to strictly ground its responses in primary source material, the system acts as a knowledgeable, faithful study companion that cites its sources and mitigates LLM hallucinations.

## Architecture
The platform is built on a high-performance **dual-stack architecture**:
- **Frontend**: A modern, responsive **Next.js** web application utilizing modular React components and Tailwind CSS. The interface features a premium "Smoked Glass" (glassmorphism) UI aesthetic built over dynamic mesh and radial backgrounds.
- **Backend**: A robust, async-first **Python FastAPI** service. It utilizes LangChain for LLM orchestration, providing a type-safe, enterprise-grade generative engine.

## Key Features
-  **AI Intent Router**: Dynamically classifies incoming user queries (`CHITCHAT`, `RAG_FOLLOWUP`, or `RAG_NEW_TOPIC`) via a 3-way branching classifier. This enables intelligent history-rewriting and avoids unnecessary database retrievals. 
-  **FAISS/BM25 Ensemble Retriever**: Combines semantic meaning (FAISS embeddings) with exact keyword matching (BM25) to provide highly accurate, dual-pronged context retrieval against the vast text corpus.
-  **Dual-Theme Glassmorphism UI**: Seamless Light and Dark mode integration utilizing `next-themes`, ensuring frosted glass interfaces and floating elements remain visually stunning across all device preferences.

## Security Measures
The backend has been deeply refactored and audited to guarantee production-grade security:
- **HMAC API Key Validation**: Uses constant-time string comparisons to prevent timing-oracle side-channel attacks during authentication.
- **Payload Size Limits**: Custom Starlette middleware drops requests exceeding 1MB before parsing, neutralizing RAM-exhaustion (DDoS) vectors.
- **SHA-256 Index Validation**: Verifies the integrity of the binary `.pkl` BM25 index on startup, neutralizing potential Remote Code Execution (RCE) via malicious `pickle` injection.
- **Prompt Injection Guards**: Secures LLM structures via strict `typing.Literal` Pydantic bounds and regex-sanitized path citations.

## Environment Setup
Create a `.env` file in the root of the project to configure the environment. Do not commit this file to version control.

```env
# Required: Your OpenAI API key for embeddings and generation
OPENAI_API_KEY="sk-proj-..."

# Required: Shared secret to authenticate the frontend requests to the backend
APP_API_KEY="your-secure-random-string"

# Optional/Prod: SHA-256 checksum of your data/index/bm25.pkl file
# to guarantee database integrity prior to server boot.
BM25_INDEX_SHA256="your-sha256-hash"

# Optional: CORS allowed origins (defaults to http://localhost:3000)
ALLOWED_ORIGINS="http://localhost:3000,https://app.yourdomain.com"

# Optional: Postgres (e.g. Supabase) connection string for per-user rate
# limiting. postgres:// and postgresql:// URLs are accepted. If unset, the
# app still starts and answers questions, with rate limiting disabled.
# With the Supabase transaction pooler (port 6543), prepared statements are
# turned off automatically.
DATABASE_URL="postgresql://postgres:password@db.yourproject.supabase.co:5432/postgres"

# Optional: what to do if the rate-limit database is unreachable.
# false (default) = answer anyway (fail-open); true = return 503 (fail-closed).
RATE_LIMIT_FAIL_CLOSED="false"
```

### Database setup (rate limiting)
On startup the backend creates the `user_query_logs` table if it doesn't exist. To create it yourself instead, or to add the indexes to a table that already exists, run [`sql/001_create_user_query_logs.sql`](sql/001_create_user_query_logs.sql) once in the Supabase SQL editor. The script is safe to re-run, and it also turns on row-level security so Supabase's public API keys can't read the table.

The backend also adds the `is_admin` column at startup if it's missing; [`sql/002_add_is_admin_to_user_query_logs.sql`](sql/002_add_is_admin_to_user_query_logs.sql) does the same by hand.

Signed-in users' chats are saved in `chat_conversations` and `chat_messages`, created at startup like the other table (or by hand with [`sql/003_create_chat_history.sql`](sql/003_create_chat_history.sql)). On Postgres the backend turns on row-level security for all three tables at startup. Guests' chats are never stored.

### Limits and history at a glance
| Who | Questions | Resets | Chat history |
|---|---|---|---|
| Guest (by IP) | 1 in total | never; signing in gives 7 a day | this browser tab only |
| Signed in | 7 in any rolling 24 hours | a question frees up 24 hours after it was asked | saved, listed in the sidebar |
| Admin (`ADMIN_EMAILS`) | unlimited | n/a | saved |

`GET /api/v1/usage` returns the caller's allowance (`limit`, `used`, `remaining`, `reset_at`), and every answer includes the updated `usage`. A limit-reached 429 keeps its `detail` code (`GUEST_LIMIT_REACHED` / `USER_LIMIT_REACHED`) and adds `reset_at` and `usage`. `GET /api/v1/conversations` and `GET /api/v1/conversations/{id}` return the signed-in user's saved chats, and `DELETE /api/v1/conversations/{id}` deletes one (deleting doesn't give questions back); a chat that belongs to someone else is a 404. All of them need the API key, so the browser reaches them through the Next.js routes `/api/usage` and `/api/conversations`.

To run the Postgres integration test against a real database:

```bash
TEST_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/postgres" uv run pytest -m postgres -v --no-cov
```
*`--no-cov` is needed because the 70% coverage gate in `pyproject.toml` fails when only one test is selected.*

### Checking retrieval quality
`uv run python scripts/diagnose_retrieval.py` prints, for ten sample questions (or the ones you pass), the keyword (BM25) hits with the words they matched, the FAISS hits with their relevance, and the final sources. Keyword search ignores stopwords and only returns chunks that actually match a content word (see `src/backend/retrieval/keyword.py`).

### Retrieval code
Everything that finds passages lives in `src/backend/retrieval/`, behind one small `Retriever` interface (`ainvoke(query)` and `chunk_counts()`, in `base.py`). Today `HybridRetriever` fuses FAISS semantic search (`vector.py`) and BM25 keyword search (`keyword.py`) with reciprocal rank fusion (`hybrid.py`), loaded from `data/index/` by `local_index.py`. Index files are read with a restricted unpickler (`index_files.py`) that only builds the few classes they contain, so LangChain's community package isn't needed and a tampered file can't run code. A Supabase pgvector + full-text implementation only needs to provide the same two methods and be loaded in `main.load_retriever()`.

### Evaluating changes
`uv run python scripts/eval_baseline.py eval/<name>.json` runs the 15 questions in `tests/eval/questions.json` through the real pipeline (real indexes and OpenAI calls, a few cents) and records each one's intent, sources, answer length and step timings (routing, retrieval, time to first token). Compare a run against `eval/after_1x.json` after changing models, prompts or retrieval.

## Local Development

### 1. Start the FastAPI Backend
Ensure your Python environment is set up and locked via `uv`. From the root repository directory, start the server:

```bash
uv run uvicorn src.backend.main:app --reload --port 8000
```
*The endpoint will be available at `http://localhost:8000`. Interactive API documentation can be viewed at `http://localhost:8000/docs`.*

### 2. Start the Next.js Frontend
Navigate into the Next.js application directory and run the Node development server:

```bash
cd src/frontend
npm run dev
```
*The user interface will be available at `http://localhost:3000`.*

## Deployment

The backend runs on **Render** (`https://gospel-topics-rag.onrender.com`) and the frontend on **Vercel** (`https://gospel-topics-rag.vercel.app`). The browser only talks to Vercel. The Next.js route `src/frontend/src/app/api/query/route.ts` forwards each question to Render server-side and attaches the API key there, so the key never reaches the browser.

### Backend on Render
The service is described in [`render.yaml`](render.yaml) (a Render Blueprint):

| Setting | Value |
|---|---|
| Build command | `uv sync --frozen --no-dev && uv cache prune --ci` |
| Start command | `uv run --no-dev uvicorn src.backend.main:app --host 0.0.0.0 --port $PORT` |
| Python | taken from `.python-version` (3.12) |
| Health check path | `/health` |
| Auto-deploy | on every commit to `main` |

`--no-dev` is needed in **both** commands: `uv run` re-syncs the default dependency groups before it runs, so without it the start command would reinstall pytest and coverage on every boot.

Environment variables (Render dashboard → Environment; never commit values):

| Variable | Required | Value |
|---|---|---|
| `OPENAI_API_KEY` | yes | OpenAI key used for embeddings and chat |
| `APP_API_KEY` | yes | Shared secret; must equal `BACKEND_API_KEY` on Vercel |
| `DATABASE_URL` | for rate limiting | Supabase **session pooler** URL (port 5432, user `postgres.<project-ref>`). `postgres://` is fine. If unset, rate limiting is off |
| `ALLOWED_ORIGINS` | recommended | `https://gospel-topics-rag.vercel.app` (comma-separated for more; spaces and trailing slashes are ignored) |
| `BM25_INDEX_SHA256` | recommended | SHA-256 of `data/index/bm25.pkl`; startup refuses to load a file that doesn't match |
| `RATE_LIMIT_FAIL_CLOSED` | no | `true` returns 503 when the rate-limit database is down; default `false` answers anyway |

To get the value for `BM25_INDEX_SHA256` (update it whenever you rebuild the index):

```bash
uv run python -c "import hashlib; print(hashlib.sha256(open('data/index/bm25.pkl','rb').read()).hexdigest())"
```

Use the session pooler rather than the transaction pooler (6543): Render runs one long-lived process, which is what session mode is for. The direct `db.<ref>.supabase.co` host needs IPv6 or Supabase's IPv4 add-on.

Memory: with the FAISS + BM25 indexes loaded (10,160 chunks each), the process uses about 290 MB after startup and about 335 MB after answering queries, so a 512 MB instance is enough with `WEB_CONCURRENCY=1`. Each extra uvicorn worker loads its own copy of the indexes, so don't raise `WEB_CONCURRENCY` on a 512 MB instance.

### Frontend on Vercel
- Root directory: `src/frontend` (framework preset: Next.js)
- Environment variables (Project → Settings → Environment Variables → Production):

| Variable | Value |
|---|---|
| `BACKEND_URL` | `https://gospel-topics-rag.onrender.com/api/v1/query`, the **full endpoint path**, not just the host |
| `BACKEND_API_KEY` | same value as `APP_API_KEY` on Render |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk publishable key (`pk_live_…` for production) |
| `CLERK_SECRET_KEY` | Clerk secret key (`sk_live_…` for production) |
| `ADMIN_EMAILS` | Optional. Comma-separated emails (case-insensitive, spaces ignored) whose users get unlimited questions, e.g. `you@example.com,other@example.com`. Only **verified** Clerk email addresses count. Unset or empty means nobody is an admin |

Admins: `/api/query` checks the signed-in user's verified emails against `ADMIN_EMAILS` and, for a match, sends `X-User-Role: admin` to the backend, which then skips the daily question limit (the 10/minute burst limit still applies). Any `X-User-Role` sent by the browser is dropped. Admin questions are still logged in `user_query_logs` with `is_admin = true`, and those rows never count toward a limit. Admins see an "Admin · unlimited" badge next to their avatar.

The proxy waits up to 55 seconds for the backend. A sleeping Render instance can take about that long to boot and load the indexes. If the backend times out, is unreachable, or returns Render's 502/503/504 page, the user sees "Waking up the server, please try again in a moment."

### Checking a deployment
```bash
curl https://gospel-topics-rag.onrender.com/health
# {"status":"ok","commit":"<short sha>","indexes_loaded":true,"db":"ok"}
```
- `commit` should match the latest commit on `main` (`git log -1 --format=%h`). If it doesn't, the new build hasn't gone live yet; check Render → Events.
- `db` is `ok`, `disabled` (no `DATABASE_URL`), or `error` (database unreachable). `error` makes `status` read `degraded` but still returns HTTP 200, because questions are still answered and Render shouldn't restart the service.
- `/health` returns 503 only while the indexes are still loading.
- The Render startup log lists the commit, which env vars are set (names only), the CORS origins and the index chunk counts.
