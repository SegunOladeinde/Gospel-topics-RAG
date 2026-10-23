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

## Data platform (Supabase)

Passage search can run from Supabase Postgres instead of the FAISS/BM25 files, so nothing large is loaded into Render's memory and new collections become searchable without a redeploy.

| Piece | Where |
|---|---|
| Raw pages backup | private Storage bucket `raw-data`, written by `scripts/backup_raw.py` with a `manifest.json` (path, sha256, size per file). Re-runs upload only changed files |
| Passage store | `rag_documents` (one row per page: collection, slug, real title, canonical URL, sha256) and `rag_chunks` (passage text, `halfvec(1536)` embedding with an HNSW cosine index, generated `tsvector` with a GIN index). Schema: [`sql/004_rag_store.sql`](sql/004_rag_store.sql), also applied by `init_db()` on startup. RLS on, no public policies |
| Search | `hybrid_search(query_embedding, query_text, k, collections)`: semantic and IDF-weighted full-text top-k fused by reciprocal rank. `match_chunks(query_embedding, k, candidate_ids, collections)`: semantic top-k plus the similarity of given passages (used with in-memory BM25), optionally within some collections. Only `postgres` and `service_role` may call them |
| Ingestion | `scripts/ingest.py`, configured by [`data/collections.yaml`](data/collections.yaml) |

`halfvec` was chosen over `vector`: text-embedding-3-small vectors survive the float16 round trip exactly (identical top-20 neighbours for every eval question) at half the storage.

### Ingesting
```bash
uv run python scripts/backup_raw.py          # back up data/raw/ first (only changes upload)
uv run python scripts/ingest.py --dry-run    # documents, passages, tokens, estimated cost
uv run python scripts/ingest.py              # refuses to spend over $1 without --approve-cost
```
Runs are incremental: unchanged pages (same sha256) are skipped, edited pages are re-chunked and re-embedded, and pages deleted from disk are removed with their passages. Passages are cut from the raw file text exactly as the FAISS index was, and `--reuse-embeddings` (on by default) takes their vectors from `data/index/`, so the current corpus costs nothing to load. `--max-docs N` caps a run; the next run carries on. `--force` re-writes unchanged pages too. Embeddings are batched with retries and backoff. Each page's real title ("Aaron, Descendants of") and canonical URL (redirects followed for collections with `resolve_redirects: true`) are stored, and source links use them.

How it writes:
- **Directly to Postgres when `DATABASE_URL` is in `.env`** (the default; `--store sql`). Each batch of `--batch-docs` pages (100) is one transaction: the documents with their final sha256, then their passages streamed in by binary COPY. A crash rolls the batch back, and the next run redoes it.
- **Through Supabase's REST API** (`--store rest`, or when there's no `DATABASE_URL`). Several times slower.
- **Pages are read and embedded one batch at a time,** so memory stays flat however many there are. Reused embeddings are held in one float32 matrix (about 60 MB).
- **Every batch logs its progress,** pages/min and an ETA.

Measured from this repo's development machine (slow link to Supabase), re-writing the same pages with reused embeddings:

| Run | REST | Postgres + COPY |
|---|---|---|
| 200 Topical Guide pages | 185 pages/min | 708 pages/min |
| all 1,275 Bible Dictionary pages | | 878 pages/min (1,333 passages/min) |

Gospel Topics pages are capped at 2 a second while their redirects are resolved.

When a run has written anything, it asks the backend to rebuild its in-memory keyword index: `POST /admin/reindex` on `BACKEND_BASE_URL` (default `https://gospel-topics-rag.onrender.com`), with `APP_API_KEY` and the admin role. `--no-reindex` skips it; the backend's own 10-minute check (below) picks the change up anyway.

**Adding a collection** (e.g. church history): put one `.txt` per page in a new folder under `data/raw/` (title on line 1, `Source: <url>` on line 2), add an entry to `data/collections.yaml`, add its label and URL pattern to `COLLECTIONS` in `src/backend/source_links.py`, then back up and ingest. With `RETRIEVER_BACKEND=pgvector` it's searchable immediately.

### Switching backends
`RETRIEVER_BACKEND` on Render, `/health` shows the active one and its passage count under `retriever`:

| Value | Semantic search | Keyword search | RAM for retrieval |
|---|---|---|---|
| `faiss` (default) | FAISS file in memory | BM25 file in memory | all indexes |
| `pgvector` | Supabase (pgvector) | Postgres full text | none |
| `pgvector_bm25` | Supabase (pgvector) | BM25 built in memory from the stored passages at startup (chunk ids only, `sparse_bm25.py`); keyword-only hits below 0.30 similarity dropped | about 6 MB |

On the 30-question retrieval eval (`uv run python scripts/eval_retrieval.py`, relevance lists in `tests/eval/questions.json`, results in `eval/retrieval_comparison.json`), `pgvector_bm25` was no worse than FAISS on any question and better on one; `pgvector` was worse on 9, because Postgres full-text ranking picks weaker keyword matches than BM25. `scripts/eval_baseline.py --backend ...` compares answers end to end.

`pgvector_bm25`'s BM25 query leaves out collection names (built in, plus every name and label in `data/collections.yaml`) and generic verbs ("increase", "introduce", "help"...): they say where to look or how, not what about. On the eval that took it from 97 relevant / 29 off-topic sources to 99 / 26, worse on no question than either FAISS or the unfiltered version ("What does the Bible Dictionary say about grace?" no longer cites "Bible Dictionary: Introduction"). The FAISS backend's BM25 query is unchanged.

**Institute manuals and Church History Topics (study collections):** `scripts/fetch_manuals.py` collects the Book of Mormon Student Manual (2018), the Foundations of the Restoration Class Preparation Material (2019; the course's student readings, as there is no student manual) and Church History Topics, with the Church's permission (≤ 1 request/s, resume-safe). Each page is cut at its headings (`src/backend/ingestion/sections.py`) and every section is its own document: a deep link to the heading (`...?lang=eng&id=title8#title8`), a title like "Alma 32:21. Faith and Hope", and the scripture references it discusses (in each passage's `metadata` and on a "Scriptures:" line). Every passage starts with its section line.

These collections are `group: study` in `data/collections.yaml`. With one shared top-k, their 4,971 passages displaced older sources on 16 of 27 existing eval questions, so `pgvector_bm25` searches the groups separately (`GroupedRetriever`): the core collections keep their own semantic top-4 (filtered in SQL with `hnsw.iterative_scan` and `ef_search = 200`, which matched an exact search) and their own BM25, then up to 2 study passages are added when they are at least 0.60 similar to the question or at least as similar as the best core passage. `scripts/eval_collections.py` compares runs before and after (`--tune` scores caps 2-4 x thresholds 0.55/0.60, `eval/phase_d_tuning.json`): on the 45-question eval this setting kept every older source on 26 of 27 existing questions (the 27th lost a page the before-run had found through an inexact search), made 1 existing question worse and 13 of 15 new ones better, at about 1,600 prompt tokens of passages per question.

**Keeping BM25 current without a restart:** newly ingested passages are found by semantic search at once. They reach the in-memory BM25 index when it's rebuilt:
- after an ingest run (`POST /admin/reindex`, admin only), or
- automatically, when a check every `BM25_REFRESH_SECONDS` (default 600; `0` turns it off) sees the store changed.

The check is one small query: passage count, highest passage id, and latest `rag_documents.updated_at`. Ingestion replaces passages by delete and insert, and ids only grow, so any added, replaced or removed passage changes it. A trigger keeps `updated_at` current for title and URL edits. The new index is built beside the live one and swapped in with a single assignment, so queries never wait. `/health` shows `retriever.keyword_index.refreshed_at` and whether a rebuild is running.

### Rollback and planned cleanup
FAISS stays as the instant fallback: set `RETRIEVER_BACKEND=faiss` and redeploy. **FAISS rollback is only valid until Phase D data is ingested; remove FAISS after Phase D is stable.** The FAISS files hold only today's three collections, so after Phase D a rollback would silently drop the new material from answers.

Removing FAISS means deleting:
- `data/index/*` and `scripts/build_local_index.py`
- the FAISS-only modules in `src/backend/retrieval/`: `local_index.py`, `vector.py`, `index_files.py` and `hybrid.py`'s `HybridRetriever` (`keyword.py`'s BM25 scorer stays: `pgvector_bm25` uses it)
- the `faiss-cpu` dependency and `BM25_INDEX_SHA256`
- `--reuse-embeddings` in `scripts/ingest.py`

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
| `RETRIEVER_BACKEND` | no | `faiss` (default, index files in memory), `pgvector` or `pgvector_bm25` (Supabase; need `DATABASE_URL` and an ingested store). Production runs `pgvector_bm25` |
| `BM25_REFRESH_SECONDS` | no | `pgvector_bm25` only: how often to check the store for changes and rebuild the keyword index (default `600`; `0` = only on `POST /admin/reindex`) |

To get the value for `BM25_INDEX_SHA256` (update it whenever you rebuild the index):

```bash
uv run python -c "import hashlib; print(hashlib.sha256(open('data/index/bm25.pkl','rb').read()).hexdigest())"
```

Use the session pooler rather than the transaction pooler (6543): Render runs one long-lived process, which is what session mode is for. The direct `db.<ref>.supabase.co` host needs IPv6 or Supabase's IPv4 add-on.

Memory: `/health` reports the process's resident memory as `memory_mb`. Measured live with `pgvector_bm25` and 10,160 passages:

| | Before (rank_bm25, passages in memory) | Now (`SparseBm25`) |
|---|---|---|
| Startup | 301 MB | 156 MB |
| After a rebuild | 355 MB | 164 MB (peak 172) |
| Warm (after questions) | 369 MB | about 200 MB |
| Rebuild on a warm instance | | peak 207 MB |

What changed:
- **`SparseBm25` replaces rank_bm25.** It keeps word ids, frequencies and chunk ids in flat arrays (6.3 MB) and holds no text: keyword hits' text comes back from `match_chunks`. Scores are bit-identical to rank_bm25.
- **Streamed builds:** the index is built from passages streamed 500 at a time, without sorting or copying.
- **Memory handed back:** `gc.collect()` and `malloc_trim` run after each build, and glibc is capped at 2 malloc arenas.
- **No faiss on the pgvector paths:** faiss and rank_bm25 are only imported by the `faiss` backend.

The FAISS backend used about 290 MB after startup and about 375 MB after queries.

Projection, assuming the data-dependent part grows linearly with the passage count. That's conservative, since the vocabulary grows more slowly than the passages:
- **Fixed part:** 156 MB of startup is imports and clients. The first questions add about 41 MB of one-time warm-up, whatever the data size.
- **Data-dependent part:** at most 19 MB per current corpus (index arrays plus allocator residue). A rebuild needs up to 15 MB more on top.
- **Formula:** warm-instance rebuild peak = 200 + 19 × (x − 1) + 15 × x MB.

| Data | Startup | Warm | Rebuild peak |
|---|---|---|---|
| 1× (10,160 passages) | 156 | 200 | 215 (measured 207) |
| 2.5× | 185 | 229 | 266 |
| 4× | 213 | 257 | 317 |

All of these fit a 512 MB instance with `WEB_CONCURRENCY=1`. Each extra uvicorn worker loads its own copy of everything, so don't raise `WEB_CONCURRENCY` on a 512 MB instance.

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

The query routes (`/api/query` and `/api/query/stream`) set `maxDuration = 300`, the most the Hobby plan allows with Fluid compute. They wait up to 150 seconds for the backend's first byte, enough to sit out a Render cold start. Once awake, the backend opens the stream with an SSE keep-alive comment straight away, and repeats it every 5 seconds while it routes and searches, so the stream is never silent. The other routes wait up to 55 seconds. If the backend times out, is unreachable, or returns Render's 502/503/504 page, the user sees "Waking up the server, please try again in a moment."

### Checking a deployment
```bash
curl https://gospel-topics-rag.onrender.com/health
# {"status":"ok","commit":"<short sha>","indexes_loaded":true,"db":"ok",
#  "retriever":{"backend":"pgvector_bm25","chunks":15131,"keyword_index":{...}},"memory_mb":...}
```
- `commit` should match the latest commit on `main` (`git log -1 --format=%h`). If it doesn't, the new build hasn't gone live yet; check Render → Events.
- `db` is `ok`, `disabled` (no `DATABASE_URL`), or `error` (database unreachable). `error` makes `status` read `degraded` but still returns HTTP 200, because questions are still answered and Render shouldn't restart the service.
- `/health` returns 503 only while the indexes are still loading.
- The Render startup log lists the commit, which env vars are set (names only), the CORS origins and the index chunk counts.
