"""FastAPI backend for the LDS Doctrinal RAG System.

WHAT THIS FILE DOES, IN PLAIN ENGLISH
--------------------------------------
1.  When the server starts, it loads two pre-built search indexes from disk
    into memory: a FAISS index (good at "meaning" / semantic search) and a
    BM25 index (good at exact keyword matching). Both were built ahead of
    time by ``scripts/ingest.py`` from the scraped .txt files in ``data/raw/``.
2.  It exposes one endpoint, ``POST /api/v1/query``, that the Next.js
    frontend calls with a user's question.
3.  For every question, it searches both indexes, merges the results, and
    stuffs the most relevant text chunks into a prompt for OpenAI's
    ``gpt-4o-mini`` model, instructing it to answer *only* using that
    retrieved text (this is what keeps answers grounded in the real LDS
    source material instead of the model making things up).
4.  It returns the generated answer together with the list of source
    documents it was grounded in, so the frontend can show citations.

WHY INDEXES ARE LOADED ONCE AT STARTUP (NOT PER REQUEST)
----------------------------------------------------------
Loading a FAISS index and unpickling a BM25 index from disk is relatively
slow. If we did that inside the ``/query`` endpoint, every single request
would pay that cost, which would make the API feel sluggish. Instead we use
FastAPI's "lifespan" feature (see ``lifespan()`` below) to load everything
exactly once, right when the server boots up, and keep it in memory for as
long as the server is running.

HOW TO RUN THIS SERVER
-----------------------
From the project root, with the virtual environment managed by ``uv``:

    uv run uvicorn src.backend.main:app --reload --port 8000

The ``--reload`` flag is handy during development (the server restarts
automatically when you edit code), but you would drop it in production.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import logging.config
import os
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_ipaddr
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from src.backend.database import UserQueryLog, init_db
from src.backend.history import (
    ConversationDetail,
    ConversationSummary,
    delete_conversation,
    get_conversation,
    list_conversations,
    save_turn,
)
from src.backend.retrieval import (
    PgvectorBm25Retriever,
    PgvectorRetriever,
    RestTransport,
    Retriever,
    SqlTransport,
)
from src.backend.retrieval.keyword import GENERIC_VERBS, collection_phrases
from src.backend.retrieval.refresh import KeywordIndexRefresher
from src.backend.source_links import SourceLink, source_link, source_link_for
from src.backend.usage import Usage, get_usage
from dataclasses import dataclass, field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Server-side logging
# ---------------------------------------------------------------------------
# Configured once, at import time, before anything else might log. INFO so
# our own startup/shutdown messages are visible; ERROR-level entries (see the
# exception handlers below) always include the full stack trace via
# exc_info, regardless of this base level.
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"default": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "default"}},
    "root": {"level": "INFO", "handlers": ["console"]},
})
logger = logging.getLogger(__name__)

# Load .env as early as possible, before any module-level `os.getenv()` call
# below (e.g. ALLOWED_ORIGINS) -- those run once at import time, which is
# *before* lifespan() executes, so if .env weren't loaded until lifespan()
# they'd always see the default/empty value instead of what's configured.
load_dotenv()

# Fail fast: OPENAI_API_KEY is required for both retrieval (OpenAIEmbeddings)
# and answer generation (ChatOpenAI). Checking this at import time -- before
# the indexes are loaded or the server starts accepting connections -- means
# a misconfigured deployment crashes immediately with a clear reason instead
# of booting "successfully" and then failing confusingly on the first request.
if not os.getenv("OPENAI_API_KEY"):
    logger.critical(
        "OPENAI_API_KEY is not set. The application cannot start without it "
        "-- add it to your .env file or the environment."
    )
    raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
# ``Path(__file__)`` is this file's own location: src/backend/main.py.
# We walk up three levels (backend -> src -> project root) so this code
# works no matter what directory you happen to run ``uvicorn`` from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "index"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"

# Must match the embedding model used to build the FAISS index in
# scripts/ingest.py. If these ever drift apart, similarity search silently
# produces bad results (the query vector and the stored vectors would be in
# different "spaces").
EMBEDDING_MODEL = "text-embedding-3-small"

# The chat model that turns retrieved context + the user's question into a
# final answer. gpt-4o-mini is fast and inexpensive, per CLAUDE.md.
CHAT_MODEL = "gpt-4o-mini"

# Model used for the cheap standalone-question rewrite step.
# Deliberately kept separate from CHAT_MODEL so we can swap it independently.
REWRITE_MODEL = "gpt-4o-mini"

# Longest question we'll answer. The UI caps its input at the same length;
# this is the server-side guarantee. (Message.content allows more because
# replayed assistant answers can be longer.)
MAX_QUESTION_CHARS = 2000
QUESTION_TOO_LONG_DETAIL = (
    f"That question is a bit long. Please keep it under {MAX_QUESTION_CHARS:,} characters."
)

# How many chunks each retriever should fetch before we merge and hand them
# to the LLM. More chunks = more context, but also a bigger/slower prompt.
CHUNKS_PER_RETRIEVER = 4

# Comma-separated list of origins allowed to call this API (see CORS setup
# below), e.g. "https://app.example.com,https://staging.example.com" in
# production. Defaults to the local Next.js dev server so `uv run uvicorn
# ...` still works out of the box without any .env changes.
DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000"


def parse_allowed_origins(raw: str | None) -> list[str]:
    """Split ALLOWED_ORIGINS into exact origins CORSMiddleware can match.

    Browsers send ``Origin`` as scheme://host[:port] with no path and no
    trailing slash, and the middleware compares strings exactly. So spaces
    and trailing slashes from a hand-typed value would silently block the
    frontend. Strip both, drop empty entries and duplicates.
    """
    origins = (part.strip().rstrip("/") for part in (raw or DEFAULT_ALLOWED_ORIGINS).split(","))
    return list(dict.fromkeys(origin for origin in origins if origin))


ALLOWED_ORIGINS = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS"))

SYSTEM_PROMPT = (
    "You are a warm, knowledgeable gospel-study companion. "
    "You speak in a friendly, encouraging tone — like a trusted friend "
    "who has studied LDS scriptures and Gospel Topics deeply. "
    "When the user asks follow-up questions, acknowledge what was said "
    "before and build naturally on the conversation. "
    "Base every answer strictly on the reference passages that come with "
    "each question, and don't add doctrine they don't support. If they "
    "don't cover the question, say honestly that you don't have enough "
    "to answer it well, rather than speculating. "
    "Write as if you simply know the material: never mention the passages "
    "themselves, the \"context\", \"documents\", \"sources provided\", or "
    "\"the text above\". Instead, cite naturally by name when it helps, "
    "e.g. \"The Bible Dictionary explains...\", \"In the Topical Guide "
    "under Faith...\", or \"Gospel Topics notes...\". "
    "Occasionally (not every message) remind the user that personal "
    "prayer and the Holy Spirit are the ultimate witnesses of truth, "
    "and note that you are not an official resource of "
    "The Church of Jesus Christ of Latter-day Saints."
)

# System prompt for route_question()'s one model call (only made when there
# is earlier conversation): decide the intent and produce the search query
# together, instead of a classify call followed by a rewrite call.
ROUTER_PROMPT = """You route the latest message in a gospel-study chat and prepare it for search.

Return two fields:

intent:
  CHITCHAT      - the message is ONLY a greeting, thanks, compliment or small
                  talk, with no question or request at all.
  RAG_FOLLOWUP  - a question that needs the earlier conversation to make sense
                  (e.g. "Why should I keep it?", "Can you share scriptures about that?").
  RAG_NEW_TOPIC - a new gospel question that makes sense on its own.
  Small talk combined with ANY question or request (e.g. "Thanks! And what
  about tithing?") is never CHITCHAT.

standalone_question:
  The latest message rewritten as one self-contained question for searching a
  document database: replace pronouns and references ("it", "that topic") with
  what they refer to in the conversation, and drop any greeting or thanks."""


# Lightweight system prompt for the CHITCHAT branch — no RAG context needed.
CHITCHAT_SYSTEM_PROMPT = (
    "You are a warm, friendly gospel-study companion. "
    "The user has sent a conversational message (a greeting, thank-you, or "
    "compliment). Respond naturally and warmly in one or two sentences. "
    "Do not cite scriptures or source documents — this is just friendly chat."
)


# ---------------------------------------------------------------------------
# Body-size guard middleware (Issue #6)
# ---------------------------------------------------------------------------
class ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds 1 MB before the body
    is parsed into memory, preventing RAM-exhaustion from oversized payloads."""

    _MAX_BYTES: int = 1_048_576  # 1 MB

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        content_length = request.headers.get("content-length")
        if content_length is not None:
            # A malformed header ("abc", "-1", "1e9") is the client's mistake:
            # answer 400 rather than letting int() raise into a 500.
            try:
                size = int(content_length)
            except ValueError:
                size = -1
            if size < 0:
                return StarletteResponse("Invalid Content-Length header.", status_code=400)
            if size > self._MAX_BYTES:
                return StarletteResponse(
                    "Request body too large (max 1 MB).", status_code=413
                )
        return await call_next(request)


# ---------------------------------------------------------------------------
# BM25 index integrity check (Issue #1)
# ---------------------------------------------------------------------------
def _verify_bm25_hash(path: Path) -> None:
    """SHA-256 integrity check before unpickling the BM25 index.

    If the ``BM25_INDEX_SHA256`` environment variable is set (in .env locally,
    or in the Render dashboard), compute the file's digest and
    compare with ``hmac.compare_digest`` (constant-time, no timing oracle).
    A mismatch aborts startup immediately.  If the variable is unset, a
    warning is logged and loading proceeds.

    To get the hash after building the index::

        python -c \
          "import hashlib; \
           print(hashlib.sha256(open('data/index/bm25.pkl','rb').read()).hexdigest())"
    """
    expected = os.getenv("BM25_INDEX_SHA256", "").strip().lower()
    if not expected:
        logger.warning(
            "BM25_INDEX_SHA256 environment variable is not set -- skipping the "
            "integrity check. Set it to the SHA-256 hex digest of "
            "data/index/bm25.pkl for tamper detection in production."
        )
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65_536), b""):
            h.update(block)
    actual = h.hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise RuntimeError(
            "BM25 index integrity check FAILED: data/index/bm25.pkl does not "
            "match BM25_INDEX_SHA256.\n"
            f"  Expected (BM25_INDEX_SHA256): {expected[:12]}...\n"
            f"  Actual (file on disk):        {actual[:12]}...\n"
            "If you rebuilt the index on purpose, set the BM25_INDEX_SHA256 "
            "environment variable (Render dashboard -> Environment, or .env "
            "locally) to the new full digest, printed by:\n"
            "  uv run python -c \"import hashlib; print(hashlib.sha256("
            "open('data/index/bm25.pkl','rb').read()).hexdigest())\"\n"
            "If you didn't rebuild it, the file may have been tampered with: "
            "don't start the app until it's restored from git."
        )
    logger.info("BM25 index integrity check passed.")


RETRIEVER_BACKENDS = ("faiss", "pgvector", "pgvector_bm25")

# pgvector_bm25: keyword-only hits less similar than this to the question are
# dropped. Chosen on the 30-question eval (scripts/eval_retrieval.py): 0.30
# removed an off-topic source without losing a relevant one; 0.35 and up
# started dropping relevant pages.
PGVECTOR_BM25_MIN_SIMILARITY = 0.30

# pgvector_bm25: how often to check whether the store changed and the
# in-memory BM25 index needs rebuilding (src/backend/retrieval/refresh.py).
# BM25_REFRESH_SECONDS=0 turns the check off (POST /admin/reindex still works).
DEFAULT_BM25_REFRESH_SECONDS = 600


def bm25_refresh_seconds() -> float:
    raw = os.getenv("BM25_REFRESH_SECONDS", "").strip()
    try:
        return max(float(raw), 0.0) if raw else DEFAULT_BM25_REFRESH_SECONDS
    except ValueError:
        logger.warning("BM25_REFRESH_SECONDS=%r isn't a number; using %ds.", raw, DEFAULT_BM25_REFRESH_SECONDS)
        return DEFAULT_BM25_REFRESH_SECONDS


def keyword_drop_phrases() -> tuple[str, ...]:
    """Collection names left out of pgvector_bm25's BM25 query: the built-in
    ones plus every collection's name and label in data/collections.yaml, so
    a collection added later is covered without a code change."""
    from src.backend.ingestion.config import load_config

    try:
        collections = load_config().collections
    except Exception:  # noqa: BLE001 - the built-in names still apply
        logger.warning("Couldn't read data/collections.yaml; using the built-in collection names.", exc_info=True)
        collections = []
    return collection_phrases([value for c in collections for value in (c.name, c.label)])


def retriever_backend() -> str:
    backend = os.getenv("RETRIEVER_BACKEND", "faiss").strip().lower() or "faiss"
    if backend not in RETRIEVER_BACKENDS:
        raise RuntimeError(f"RETRIEVER_BACKEND must be one of {', '.join(RETRIEVER_BACKENDS)}, not {backend!r}")
    return backend


def load_retriever(session_factory=None) -> Retriever:
    """Set up passage search for RETRIEVER_BACKEND (see src/backend/retrieval).

    faiss (default): semantic search over the FAISS file plus BM25 keyword
    search, both loaded into memory from data/index/.
    pgvector: the same two searches inside Supabase Postgres (hybrid_search()),
    over the backend's database connection, or Supabase's REST API with
    SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY when there's no DATABASE_URL.
    Nothing is loaded into memory, and newly ingested collections are
    searchable without a redeploy.
    pgvector_bm25: semantic search in Supabase plus BM25 built in memory from
    the stored passages at startup (about 30 MB). Matched FAISS on every
    eval question; Postgres full-text ranking didn't.

    The embeddings client is created once here and reused for every question
    (it keeps its HTTP connection pool).
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, timeout=10.0)
    backend = retriever_backend()
    if backend in ("pgvector", "pgvector_bm25"):
        if session_factory is not None:
            transport = SqlTransport(session_factory)
        elif os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
            transport = RestTransport(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
        else:
            raise RuntimeError(
                f"RETRIEVER_BACKEND={backend} needs DATABASE_URL (or SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)."
            )
        if backend == "pgvector_bm25":
            # Its BM25 query also drops collection names ("what does the Bible
            # Dictionary say about grace") and generic verbs ("increase"):
            # they say where to look or how, not what about. Checked on the
            # 30-question eval (scripts/eval_retrieval.py): no question worse.
            retriever = PgvectorBm25Retriever(
                transport,
                embeddings,
                k=CHUNKS_PER_RETRIEVER,
                min_similarity=PGVECTOR_BM25_MIN_SIMILARITY,
                drop_phrases=keyword_drop_phrases(),
                drop_words=GENERIC_VERBS,
            )
        else:
            retriever = PgvectorRetriever(transport, embeddings, k=CHUNKS_PER_RETRIEVER)
        if not retriever.chunk_counts()["vector"]:
            logger.error("The pgvector store is empty: run scripts/ingest.py. Every question will find nothing.")
        return retriever

    if not INDEX_DIR.exists():
        raise RuntimeError(
            f"Index directory not found at {INDEX_DIR}. "
            "Run `uv run scripts/build_local_index.py` first to build the indexes."
        )
    _verify_bm25_hash(BM25_INDEX_PATH)
    # Imported here so the pgvector backends never load faiss.
    from src.backend.retrieval.local_index import load_local_retriever

    return load_local_retriever(INDEX_DIR, embeddings, k=CHUNKS_PER_RETRIEVER)


REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "APP_API_KEY")
OPTIONAL_ENV_VARS = (
    "DATABASE_URL",
    "ALLOWED_ORIGINS",
    "BM25_INDEX_SHA256",
    "RATE_LIMIT_FAIL_CLOSED",
    "RETRIEVER_BACKEND",
    "BM25_REFRESH_SECONDS",
)


def log_startup_config() -> None:
    """Log the build commit and which env vars are set -- names only, never values."""
    def is_set(name: str) -> bool:
        return bool(os.getenv(name, "").strip())

    present = [name for name in REQUIRED_ENV_VARS + OPTIONAL_ENV_VARS if is_set(name)]
    missing_optional = [name for name in OPTIONAL_ENV_VARS if not is_set(name)]
    logger.info("Starting commit %s", current_commit())
    logger.info("CORS allowed origins: %s", ", ".join(ALLOWED_ORIGINS))
    logger.info("Env vars set: %s", ", ".join(present) or "(none)")
    if missing_optional:
        logger.info("Optional env vars not set: %s", ", ".join(missing_optional))
    for name in REQUIRED_ENV_VARS:
        if not is_set(name):
            # OPENAI_API_KEY is already enforced at import; a missing
            # APP_API_KEY means every /api/v1/query request gets a 401.
            logger.warning("Required env var %s is not set.", name)


def index_chunk_counts(retriever: Retriever) -> dict[str, int] | None:
    """Passages per index ({"vector": ..., "keyword": ...}), for the startup log."""
    if not isinstance(retriever, Retriever):
        return None  # e.g. a test stub
    return retriever.chunk_counts()


def log_index_counts(retriever: Retriever) -> None:
    """Log the loaded chunk counts. Diagnostics only: any error is logged as
    a warning and startup carries on, since the indexes themselves loaded."""
    try:
        counts = index_chunk_counts(retriever)
    except Exception:  # noqa: BLE001 - never fail startup over a log line
        logger.warning("Could not count the loaded index chunks.", exc_info=True)
        return
    if counts:
        logger.info("Indexes loaded: %d vector chunks, %d keyword chunks.", counts["vector"], counts["keyword"])
        if counts["vector"] != counts["keyword"]:
            logger.warning("Vector and keyword chunk counts differ -- the indexes may be out of sync.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup and shutdown code for the FastAPI app.

    Code before ``yield`` runs once, when the server starts. Code after
    ``yield`` runs once, when the server shuts down. This is the officially
    recommended place to load expensive resources (like our indexes) so
    they are ready in memory before the first request ever arrives.
    """
    # .env is already loaded at module import time (see top of file), which
    # covers OPENAI_API_KEY, APP_API_KEY, and ALLOWED_ORIGINS alike.

    log_startup_config()
    # Creates the app's tables if missing (and the pgvector store's schema).
    # None when DATABASE_URL is unset or invalid, which disables rate
    # limiting. Set up first: the pgvector backend searches through it.
    app.state.db_session_factory = await asyncio.to_thread(init_db)
    backend = retriever_backend()
    logger.info("Starting up: retriever backend %s...", backend)
    # Attached directly to FastAPI's own per-app state object -- no separate
    # global dict to keep in sync with the app's lifecycle. Populated once,
    # here, and only ever read from inside request handlers afterward.
    app.state.retriever = await asyncio.to_thread(  # Issue #4: never block the event loop
        load_retriever, app.state.db_session_factory
    )
    app.state.retriever_backend = backend
    log_index_counts(app.state.retriever)
    # pgvector_bm25 keeps a BM25 index in memory: rebuild it when the store
    # changes (automatic check, or POST /admin/reindex) instead of restarting.
    app.state.keyword_refresher = None
    if isinstance(app.state.retriever, PgvectorBm25Retriever):
        app.state.keyword_refresher = KeywordIndexRefresher(app.state.retriever, bm25_refresh_seconds())
        app.state.keyword_refresher.start()
    app.state.chat_model = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    logger.info("Startup complete: retrieval is ready for queries.")

    yield  # <-- the app runs here, handling requests, until it shuts down

    logger.info("Shutting down: releasing in-memory indexes.")
    if app.state.keyword_refresher is not None:
        await app.state.keyword_refresher.stop()


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_chat_model(request: Request) -> ChatOpenAI:
    return request.app.state.chat_model

def get_db(request: Request) -> Iterator[Session | None]:
    """Yield a DB session for one request (None if no database is configured).

    The ``with`` block closes the session even if the request raises.
    """
    session_factory = getattr(request.app.state, "db_session_factory", None)
    if session_factory is None:
        yield None
        return
    with session_factory() as db:
        yield db


RATE_LIMIT_UNAVAILABLE_DETAIL = (
    "We can't check your usage limit right now, so we can't answer just yet. "
    "Please try again in a few minutes."
)


def rate_limit_fail_closed() -> bool:
    """True if RATE_LIMIT_FAIL_CLOSED asks us to refuse requests (503) when
    the rate-limit database is unavailable. Read per request; default false."""
    return os.getenv("RATE_LIMIT_FAIL_CLOSED", "false").strip().lower() in {"1", "true", "yes"}


def rollback_quietly(db: Session) -> None:
    """Roll back after a failed DB call. If the connection is already gone,
    the rollback can fail too; that error is logged, not raised."""
    try:
        db.rollback()
    except SQLAlchemyError:
        logger.warning("Session rollback failed.", exc_info=True)



# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LDS Doctrinal RAG API",
    description="Retrieval-augmented question answering over LDS source texts.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (Cross-Origin Resource Sharing) middleware is what allows a web page
# served from one of ALLOWED_ORIGINS (our Next.js frontend, wherever it's
# deployed) to make fetch() requests to this API, which runs on a different
# origin. Without this, the browser would block those requests for security
# reasons.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Issue #6: Reject oversized bodies before they are parsed into memory.
app.add_middleware(ContentSizeLimitMiddleware)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# Issue #7: get_ipaddr reads X-Forwarded-For first, so the limit applies
# to real client IPs rather than the shared reverse-proxy IP.
limiter = Limiter(key_func=get_ipaddr)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------
def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency that gates a route behind a shared API key.

    The expected key lives in APP_API_KEY (set in .env, loaded via
    load_dotenv() in lifespan()). If it isn't configured at all, we fail
    closed (reject every request) rather than silently allowing anyone in.
    """
    expected_key = os.getenv("APP_API_KEY")
    # Issue #2: constant-time comparison prevents timing side-channel attacks.
    if not expected_key or not hmac.compare_digest(x_api_key or "", expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
# Catches anything that isn't already handled by a more specific handler
# (FastAPI's own HTTPException handler, slowapi's RateLimitExceeded handler,
# or a local try/except). The real exception -- message and full stack
# trace -- goes to the server log only. The client always gets a generic,
# detail-free message, regardless of what actually went wrong internally.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception on %s %s", request.method, request.url.path, exc_info=exc
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred."},
    )


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
# Pydantic models describe the exact shape of the JSON we expect to receive
# and send. FastAPI uses them to validate incoming requests automatically
# (e.g. rejecting a request with an empty question with a clear 422 error)
# and to generate interactive API docs at /docs.
class Message(BaseModel):
    """A single turn in the conversation, as kept by the frontend's chat state."""

    role: Literal["user", "assistant"] = Field(
        ...,
        description="Who sent this message: 'user' or 'assistant'.",
    )
    # max_length caps a single message at 8000 characters -- comfortably
    # larger than a typical AI response (2 000-5 000 chars) so that replaying
    # prior assistant turns does not trigger a 422 string_too_long error,
    # while still blocking unreasonably large blobs that would inflate costs.
    # Note: MAX_HISTORY_CHARS (4 000) is the separate LLM-replay budget;
    # this request-acceptance cap must always be >= that value.
    content: str = Field(
        ..., min_length=1, max_length=8000, description="The message text."
    )


class QueryRequest(BaseModel):
    """The JSON body the frontend sends to POST /api/v1/query.

    ``messages`` is the full conversation so far (oldest first), so the
    model can remember earlier turns. The last message must be the new
    question from the user -- it's the one we run retrieval against.
    """

    # max_length caps how many turns of history a single request can carry.
    # Combined with each message's own max_length, this bounds the total
    # size (and therefore cost) of any one request to the LLM.
    messages: list[Message] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Full conversation history, oldest first. Last entry is the new question.",
    )
    conversation_id: str | None = Field(
        default=None,
        max_length=64,
        description="Saved conversation to append this turn to (signed-in users). Omit to start a new one.",
    )


class QueryResponse(BaseModel):
    """The JSON body we send back to the frontend."""

    answer: str = Field(..., description="The generated, grounded answer.")
    sources: list[str] = Field(
        default_factory=list,
        description="Names of the source documents the answer was grounded in.",
    )

    conversation_id: str | None = Field(
        default=None,
        description="The saved conversation this turn was added to; None for guests or without a database.",
    )
    usage: Usage | None = Field(
        default=None,
        description="The caller's question allowance after this question; None without a database.",
    )

    source_links: list[SourceLink] = Field(
        default_factory=list,
        description="Each source with a readable label and its churchofjesuschrist.org URL.",
    )

    @model_validator(mode="after")
    def _links_from_ids(self) -> "QueryResponse":
        # Callers that only know the ids get links built from them.
        if self.sources and not self.source_links:
            self.source_links = [source_link(source) for source in self.sources]
        return self


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
# Issue #5: in-process cache for route_question, keyed on the question plus
# a digest of the last INTENT_CACHE_TURNS messages: the same words can be a
# follow-up in one conversation and a new topic in another. FIFO eviction at
# _INTENT_CACHE_MAX entries.
_INTENT_CACHE: dict[tuple[str, str], "RouteDecision"] = {}
_INTENT_CACHE_MAX = 256
INTENT_CACHE_TURNS = 4

# Phrases that make up pure small talk. A message counts as CHITCHAT only if
# every clause in it is one of these (see is_pure_small_talk), so anything
# with a real question or request in it always goes to retrieval.
_SMALL_TALK_PHRASES = {
    "hi", "hello", "hey", "hi there", "hello there", "hey there", "greetings",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "thanks a lot", "thanks so much", "thank you so much",
    "thank you very much", "many thanks", "thx", "ty", "much appreciated",
    "i appreciate it", "appreciate it",
    "great", "great answer", "good answer", "nice", "awesome", "amazing",
    "perfect", "wonderful", "excellent", "cool", "that helps", "that helped",
    "that was helpful", "very helpful", "helpful", "got it", "i see",
    "ok", "okay", "alright", "sounds good", "makes sense",
    "bye", "goodbye", "see you", "see you later", "have a good day",
    "god bless", "god bless you",
}
_SMALL_TALK_CONNECTORS = {"and", "also", "so", "very", "really", "again", "oh", "wow"}
_SMALL_TALK_MAX_WORDS = max(len(phrase.split()) for phrase in _SMALL_TALK_PHRASES)
_WORDS = re.compile(r"[a-z']+")


def is_pure_small_talk(message: str) -> bool:
    """True only if the message is nothing but greetings / thanks / praise.

    The words must split cleanly into small-talk phrases (plus a few filler
    words like "and" or "so"): "Thanks!", "Hi, thank you so much!" and "ok
    thanks" are small talk; "Thanks! And what about tithing?" isn't, because
    "what about tithing" isn't. A question mark always means it isn't.
    """
    if "?" in message:
        return False
    words = _WORDS.findall(message.lower())
    if all(word in _SMALL_TALK_CONNECTORS for word in words):
        return False  # empty, or only filler like "and" / "so"
    # reachable[i]: the first i words split into phrases/connectors.
    reachable = [True] + [False] * len(words)
    for start in range(len(words)):
        if not reachable[start]:
            continue
        if words[start] in _SMALL_TALK_CONNECTORS:
            reachable[start + 1] = True
        for end in range(start + 1, min(start + _SMALL_TALK_MAX_WORDS, len(words)) + 1):
            if " ".join(words[start:end]) in _SMALL_TALK_PHRASES:
                reachable[end] = True
    return reachable[-1]


def _intent_cache_key(question: str, history: list[Message]) -> tuple[str, str]:
    recent = history[-INTENT_CACHE_TURNS:]
    digest = hashlib.sha256(
        "\x1e".join(f"{m.role}\x1f{m.content}" for m in recent).encode("utf-8")
    ).hexdigest()
    return question, digest


@dataclass(frozen=True)
class RouteDecision:
    """How to answer: the intent, and what to search for (unused for CHITCHAT)."""

    intent: str
    query: str


class _RouterOutput(BaseModel):
    intent: Literal["CHITCHAT", "RAG_FOLLOWUP", "RAG_NEW_TOPIC"]
    standalone_question: str = Field(description="The message as a self-contained search question.")


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def strip_leading_small_talk(question: str) -> str:
    """Drop opening pleasantries before searching.

    "Thanks! And what about tithing?" is searched as "And what about
    tithing?", so "Thanks" doesn't pull in keyword matches of its own.
    """
    sentences = _SENTENCE_END.split(question.strip())
    while len(sentences) > 1 and is_pure_small_talk(sentences[0]):
        sentences.pop(0)
    return " ".join(sentences)


async def route_question(question: str, history: list[Message], chat_model: ChatOpenAI) -> RouteDecision:
    """Decide how to answer the latest message, with as few model calls as possible.

    - Pure small talk (see is_pure_small_talk) is CHITCHAT, decided in code.
    - With no earlier conversation there's nothing to follow up on, and
      anything that isn't pure small talk is a real question: RAG_NEW_TOPIC,
      no model call. (Asking the model here only cost a second, and it
      sometimes labelled "Thanks! And what about tithing?" a follow-up to a
      conversation that didn't exist.)
    - Otherwise one structured-output call returns the intent and the
      standalone search question together. A CHITCHAT label is overridden,
      since the message isn't pure small talk.

    New topics are searched as asked (minus opening pleasantries), so the
    speculative search prepare_answer starts in parallel can be reused.
    Any model error falls back to RAG_NEW_TOPIC.
    """
    if is_pure_small_talk(question):
        return RouteDecision("CHITCHAT", question)
    as_asked = strip_leading_small_talk(question)
    if not history:
        return RouteDecision("RAG_NEW_TOPIC", as_asked)

    cache_key = _intent_cache_key(question, history)
    if cache_key in _INTENT_CACHE:
        logger.info("Intent cache hit for question=%r", question)
        return _INTENT_CACHE[cache_key]

    history_text = "\n".join(f"{m.role.capitalize()}: {m.content}" for m in history)
    try:
        router = chat_model.with_structured_output(_RouterOutput)
        output = await router.ainvoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=f"Conversation history:\n{history_text}\n\nLatest message: {question}"),
        ])
        intent = output.intent
        if intent == "CHITCHAT":
            logger.info("Overriding CHITCHAT for a message with a question in it: %r", question)
            intent = "RAG_FOLLOWUP"
        if intent == "RAG_FOLLOWUP":
            decision = RouteDecision(intent, output.standalone_question.strip() or as_asked)
        else:
            decision = RouteDecision("RAG_NEW_TOPIC", as_asked)
    except Exception:  # noqa: BLE001 - never block the pipeline
        logger.warning("route_question failed; defaulting to RAG_NEW_TOPIC.", exc_info=True)
        decision = RouteDecision("RAG_NEW_TOPIC", as_asked)

    logger.info("Routed %r as %s, searching %r", question, decision.intent, decision.query)
    if len(_INTENT_CACHE) >= _INTENT_CACHE_MAX:
        _INTENT_CACHE.pop(next(iter(_INTENT_CACHE)))
    _INTENT_CACHE[cache_key] = decision
    return decision


def format_context(documents: list[Document]) -> str:
    """Turn retrieved chunks into one text blob to paste into the prompt.

    Each chunk is labeled with its source so the model can (and is asked
    to) ground its answer in specific documents rather than blending
    everything together anonymously.
    """
    blocks = []
    for document in documents:
        # The readable label ("Topical Guide: Faith"), so a natural citation
        # names the source the way the UI's chips do, not "topical_guide/faith".
        source_label = source_link(describe_source(document)).label
        blocks.append(f"[{source_label}]\n{document.page_content}")
    return "\n\n---\n\n".join(blocks)


_SAFE = re.compile(r"[^\w\-]")

def describe_source(document: Document) -> str:
    """Build a human-readable source label from a chunk's metadata.

    scripts/ingest.py tags every chunk with metadata like
    ``{"source": "topical_guide", "topic": "Faith"}`` (the folder name it
    came from, and the original filename without its extension). We combine
    those into a single readable string, e.g. "topical_guide/Faith".
    """
    metadata = document.metadata
    source = _SAFE.sub("_", str(metadata.get("source", "unknown")))
    topic = _SAFE.sub("_", str(metadata.get("topic", "unknown")))
    return f"{source}/{topic}"


def collect_unique_sources(documents: list[Document]) -> list[str]:
    """Return source labels for the given chunks, de-duplicated, in order."""
    labels = (describe_source(document) for document in documents)
    # dict.fromkeys() is a common Python trick to remove duplicates while
    # preserving the original order (a plain set() would not preserve order).
    return list(dict.fromkeys(labels))


def collect_unique_links(documents: list[Document]) -> list[SourceLink]:
    """One link per source, in order, using the page's real title and URL when
    the retriever provides them (pgvector) and the id otherwise (FAISS)."""
    links: dict[str, SourceLink] = {}
    for document in documents:
        source_id = describe_source(document)
        if source_id not in links:
            links[source_id] = source_link_for(
                source_id, document.metadata.get("title"), document.metadata.get("url")
            )
    return list(links.values())


def to_langchain_message(message: Message) -> HumanMessage | AIMessage:
    """Convert one frontend chat message into the LangChain message type
    the chat model expects, so prior turns are replayed as real
    conversation history rather than flattened into a single string.
    """
    if message.role == "assistant":
        return AIMessage(content=message.content)
    return HumanMessage(content=message.content)


# Issue #3 already caps each message at 1500 chars and the whole request at
# 20 messages, but replaying that much history on every turn -- on top of
# the retrieved RAG context -- still risks blowing the token budget and,
# in a long-running conversation, the model's context window. These two
# limits bound what actually gets replayed to the LLM, independent of what
# the client is allowed to send.
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 4000


def trim_history(history: list[Message]) -> list[Message]:
    """Bound the prior-turn history replayed to the LLM.

    Two layers, both oldest-first: a sliding window on message *count*
    (last MAX_HISTORY_MESSAGES messages), then a safety budget on total
    *character* length, in case a handful of near-max-length messages still
    add up to too much context after windowing.
    """
    original_count = len(history)
    windowed = deque(history[-MAX_HISTORY_MESSAGES:], maxlen=MAX_HISTORY_MESSAGES)

    total_chars = sum(len(message.content) for message in windowed)
    while total_chars > MAX_HISTORY_CHARS and windowed:
        dropped = windowed.popleft()
        total_chars -= len(dropped.content)

    logger.info(
        "History trimmed: kept %d of %d message(s), %d char(s) total.",
        len(windowed),
        original_count,
        total_chars,
    )
    return list(windowed)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root() -> dict[str, str]:
    """Tiny landing response so a bare visit to the API isn't a 404."""
    return {"service": "gospel-topics-rag", "docs": "/docs", "health": "/health"}


# Upper bound on the database ping inside /health, so a slow or unreachable
# database can't make the health check itself hang.
HEALTH_DB_TIMEOUT_SECONDS = 2.0


def retriever_summary(state) -> dict | None:
    """Which backend answers searches and how many passages it holds."""
    retriever = getattr(state, "retriever", None)
    if retriever is None:
        return None
    try:
        counts = index_chunk_counts(retriever) or {}
    except Exception:  # noqa: BLE001 - health must never fail on this
        counts = {}
    summary = {"backend": getattr(state, "retriever_backend", None), "chunks": counts.get("vector")}
    refresher = getattr(state, "keyword_refresher", None)
    if refresher is not None:
        summary["keyword_index"] = {
            "rebuilding": refresher.rebuilding,
            "refreshed_at": refresher.refreshed_at.isoformat(timespec="seconds") if refresher.refreshed_at else None,
        }
    return summary


def process_memory_mb() -> float | None:
    """This process's resident memory (RSS) in MB, from /proc on Linux (Render);
    None where /proc isn't available (e.g. Windows)."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)  # reported in kB
    except (OSError, ValueError, IndexError):
        pass
    return None


def current_commit() -> str:
    """Short git commit of the running build (Render sets RENDER_GIT_COMMIT)."""
    return (os.getenv("RENDER_GIT_COMMIT") or "unknown")[:7]


def ping_db(session_factory) -> str:
    """Run ``SELECT 1``; return "ok" or "error" (details go to the log only)."""
    try:
        with session_factory() as db:
            db.execute(text("SELECT 1"))
        return "ok"
    except SQLAlchemyError:
        logger.warning("Health check: database ping failed.", exc_info=True)
        return "error"


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    """Health check for Render and uptime monitors.

    Unauthenticated and not rate-limited. Never calls OpenAI and never
    returns secrets. The only external call is a time-boxed ``SELECT 1``
    when a database is configured.

    Answers 200 whenever the indexes are loaded, even if the database is
    down: rate limiting fails open, so the app can still answer questions
    and Render shouldn't restart it. A database problem shows up as
    ``status: "degraded"`` / ``db: "error"``.
    """
    state = request.app.state
    indexes_loaded = getattr(state, "retriever", None) is not None

    session_factory = getattr(state, "db_session_factory", None)
    if session_factory is None:
        db_status = "disabled"
    else:
        try:
            db_status = await asyncio.wait_for(
                asyncio.to_thread(ping_db, session_factory), HEALTH_DB_TIMEOUT_SECONDS
            )
        except TimeoutError:
            logger.warning("Health check: database ping timed out.")
            db_status = "error"

    if not indexes_loaded:
        status = "starting"
    elif db_status == "error":
        status = "degraded"
    else:
        status = "ok"
    return JSONResponse(
        status_code=200 if indexes_loaded else 503,
        content={
            "status": status,
            "commit": current_commit(),
            "indexes_loaded": indexes_loaded,
            "db": db_status,
            "retriever": retriever_summary(state),
            "memory_mb": process_memory_mb(),
        },
    )


@dataclass(frozen=True)
class Caller:
    """Who is asking, as vouched for by the Next.js server (see verify_api_key).

    identifier is the Clerk user id for signed-in users and the client IP for
    guests. is_admin comes from X-User-Role: admin, which the Next.js server
    only sets for verified ADMIN_EMAILS; guests can never be admins.
    """

    identifier: str
    is_guest: bool
    is_admin: bool


def resolve_caller(
    request: Request, x_user_id: str | None, x_forwarded_for: str | None, x_user_role: str | None
) -> Caller:
    client_ip = request.client.host if request.client else "unknown"
    if x_forwarded_for and x_forwarded_for != "unknown-ip":
        client_ip = x_forwarded_for.split(",")[0].strip()
    user_id = (x_user_id or "").strip()
    is_guest = not user_id
    return Caller(
        identifier=client_ip if is_guest else user_id,
        is_guest=is_guest,
        is_admin=not is_guest and (x_user_role or "").strip().lower() == "admin",
    )


def limit_reached_response(usage: Usage) -> JSONResponse:
    """429 with the same detail codes as before, plus when to try again."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": "GUEST_LIMIT_REACHED" if usage.is_guest else "USER_LIMIT_REACHED",
            "reset_at": usage.reset_at.isoformat() if usage.reset_at else None,
            "usage": usage.model_dump(mode="json"),
        },
    )


class UsageResponse(BaseModel):
    usage: Usage | None = Field(description="None when there's no database, so no limits are enforced.")


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]


@app.get("/api/v1/usage", response_model=UsageResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("60/minute")
async def usage(
    request: Request,
    x_user_id: str | None = Header(default="", alias="X-User-Id"),
    x_forwarded_for: str | None = Header(default="", alias="X-Forwarded-For"),
    x_user_role: str | None = Header(default="", alias="X-User-Role"),
    db=Depends(get_db),
) -> UsageResponse:
    """How many questions the caller has left (for the remaining-questions
    indicator). Same counting as the /api/v1/query limit check."""
    if db is None:
        return UsageResponse(usage=None)
    caller = resolve_caller(request, x_user_id, x_forwarded_for, x_user_role)
    try:
        return UsageResponse(
            usage=get_usage(db, caller.identifier, is_guest=caller.is_guest, is_admin=caller.is_admin)
        )
    except SQLAlchemyError:
        rollback_quietly(db)
        logger.exception("Failed to read usage")
        return UsageResponse(usage=None)


@app.get(
    "/api/v1/conversations",
    response_model=ConversationListResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("60/minute")
async def conversations(
    request: Request,
    x_user_id: str | None = Header(default="", alias="X-User-Id"),
    db=Depends(get_db),
) -> ConversationListResponse:
    """The signed-in user's saved chats, most recently active first. Guests
    (and deployments without a database) get an empty list."""
    user_id = (x_user_id or "").strip()
    if db is None or not user_id:
        return ConversationListResponse(conversations=[])
    try:
        return ConversationListResponse(conversations=list_conversations(db, user_id))
    except SQLAlchemyError as exc:
        rollback_quietly(db)
        logger.exception("Failed to list conversations")
        raise HTTPException(status_code=503, detail="Chat history is unavailable right now.") from exc


@app.get(
    "/api/v1/conversations/{conversation_id}",
    response_model=ConversationDetail,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("60/minute")
async def conversation(
    request: Request,
    conversation_id: str,
    x_user_id: str | None = Header(default="", alias="X-User-Id"),
    db=Depends(get_db),
) -> ConversationDetail:
    """One of the signed-in user's saved chats with all its messages. 404 if
    it doesn't exist or belongs to someone else (indistinguishable on purpose)."""
    user_id = (x_user_id or "").strip()
    if db is None or not user_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    try:
        detail = get_conversation(db, user_id, conversation_id)
    except SQLAlchemyError as exc:
        rollback_quietly(db)
        logger.exception("Failed to load conversation")
        raise HTTPException(status_code=503, detail="Chat history is unavailable right now.") from exc
    if detail is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return detail


@app.delete(
    "/api/v1/conversations/{conversation_id}",
    status_code=204,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("60/minute")
async def remove_conversation(
    request: Request,
    conversation_id: str,
    x_user_id: str | None = Header(default="", alias="X-User-Id"),
    db=Depends(get_db),
) -> Response:
    """Delete one of the signed-in user's saved chats. 404 if it doesn't exist
    or belongs to someone else. Question counts are unaffected: they come from
    user_query_logs, not from saved chats."""
    user_id = (x_user_id or "").strip()
    if db is None or not user_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    try:
        deleted = delete_conversation(db, user_id, conversation_id)
    except SQLAlchemyError as exc:
        rollback_quietly(db)
        logger.exception("Failed to delete conversation")
        raise HTTPException(status_code=503, detail="Chat history is unavailable right now.") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return Response(status_code=204)


ANSWER_FAILED_DETAIL = "Failed to generate an answer. Please try again shortly."
NO_MATERIAL_ANSWER = (
    "I couldn't find anything in the available LDS source material to answer that question."
)


@dataclass
class PreparedAnswer:
    """What to answer with: either a fixed reply, or messages to send the model."""

    intent: str
    sources: list[str]
    llm_messages: list | None = None
    fixed_answer: str | None = None
    links: list[SourceLink] = field(default_factory=list)
    # Milliseconds spent in each step before answering, for logs and evals.
    timings: dict[str, float] = field(default_factory=dict)


def admit_question(payload: QueryRequest, caller: Caller, db) -> JSONResponse | None:
    """Length and question-limit checks. Returns the 429 to send if the caller
    is out of questions; raises HTTPException for other refusals."""
    # Before the limit check, so an over-long question doesn't use one up.
    if len(payload.messages[-1].content) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail=QUESTION_TOO_LONG_DETAIL)
    if db is None:
        return None
    try:
        usage = get_usage(db, caller.identifier, is_guest=caller.is_guest, is_admin=caller.is_admin)
    except SQLAlchemyError:
        # A database outage shouldn't take the whole Q&A service down
        # with it, so by default the question is answered unmetered
        # (fail-open). RATE_LIMIT_FAIL_CLOSED=true refuses it instead.
        logger.exception("Rate-limit check failed for %s", "guest" if caller.is_guest else "user")
        rollback_quietly(db)
        if rate_limit_fail_closed():
            raise HTTPException(status_code=503, detail=RATE_LIMIT_UNAVAILABLE_DETAIL)
        return None
    return limit_reached_response(usage) if usage.remaining == 0 else None


async def prepare_answer(
    question: str,
    history: list[Message],
    retriever: Retriever,
    chat_model: ChatOpenAI,
) -> PreparedAnswer:
    """Everything up to generating the answer: route, retrieve, build the prompt.

    Routing (route_question) only calls the model when there's earlier
    conversation. While it does, the question is already being searched as
    asked; if it turns out to be a new topic that search is used, otherwise
    (a follow-up) the standalone question is searched instead. Either way
    retrieval no longer waits for the router when it doesn't have to.
    Raises HTTPException(502) if retrieval fails.
    """
    started = time.perf_counter()
    timings: dict[str, float] = {}
    trimmed_history = trim_history(history)

    speculative: asyncio.Task | None = None
    as_asked = strip_leading_small_talk(question)
    if trimmed_history and not is_pure_small_talk(question):
        speculative = asyncio.create_task(retriever.ainvoke(as_asked))
        # If it's cancelled or unused, don't let an exception go unobserved.
        speculative.add_done_callback(lambda task: task.cancelled() or task.exception())

    route = await route_question(question, trimmed_history, chat_model)
    timings["route_ms"] = (time.perf_counter() - started) * 1000

    if route.intent == "CHITCHAT":
        if speculative:
            speculative.cancel()
        # No retrieval and no citations: just reply in kind.
        return PreparedAnswer(
            intent=route.intent,
            sources=[],
            llm_messages=[
                SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
                *(to_langchain_message(m) for m in trimmed_history),
                HumanMessage(content=question),
            ],
            timings=timings,
        )

    waited_from = time.perf_counter()
    try:
        if speculative and route.query == as_asked:
            documents = await speculative
        else:
            if speculative:
                speculative.cancel()
            documents = await retriever.ainvoke(route.query)
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval failed for question=%r", question, exc_info=exc)
        raise HTTPException(
            status_code=502,
            detail="Failed to search the document indexes. Please try again shortly.",
        ) from exc
    timings["retrieval_wait_ms"] = (time.perf_counter() - waited_from) * 1000
    timings["prepare_ms"] = (time.perf_counter() - started) * 1000
    logger.info(
        "Prepared %s in %.0f ms (routing %.0f ms, then waited %.0f ms for retrieval)",
        route.intent, timings["prepare_ms"], timings["route_ms"], timings["retrieval_wait_ms"],
    )

    if not documents:
        # No relevant material found: don't let the model improvise.
        return PreparedAnswer(intent=route.intent, sources=[], fixed_answer=NO_MATERIAL_ANSWER, timings=timings)

    # The labels here are for the model only; the system prompt tells it
    # never to refer to "passages" or "context" in its answer.
    user_prompt = (
        f"Reference passages:\n{format_context(documents)}\n\n"
        f"Question: {question}\n\n"
        "Answer from these passages only, citing them naturally by name."
    )
    return PreparedAnswer(
        intent=route.intent,
        sources=collect_unique_sources(documents),
        links=collect_unique_links(documents),
        llm_messages=[
            SystemMessage(content=SYSTEM_PROMPT),
            *(to_langchain_message(m) for m in trimmed_history),
            HumanMessage(content=user_prompt),
        ],
        timings=timings,
    )


def log_query(db, caller: Caller) -> None:
    """Count one question against the caller's allowance. Never raises."""
    if db is None:
        return
    try:
        db.add(UserQueryLog(
            user_identifier=caller.identifier,
            is_guest=caller.is_guest,
            is_admin=caller.is_admin,
        ))
        db.commit()
    except Exception as log_exc:  # noqa: BLE001
        rollback_quietly(db)
        logger.error("Failed to insert UserQueryLog", exc_info=log_exc)


def record_answer(
    db, caller: Caller, payload: QueryRequest, answer: str, sources: list[str],
    links: list[SourceLink] | None = None,
) -> QueryResponse:
    """Log the query, save the turn to the user's history, and build the
    response with their updated allowance. DB failures are logged, never
    raised: the user already has their answer."""
    conversation_id = None
    usage = None
    log_query(db, caller)
    if db is not None:
        if not caller.is_guest:
            try:
                conversation_id = save_turn(
                    db, caller.identifier, payload.conversation_id,
                    payload.messages[-1].content, answer, links or sources,
                )
            except SQLAlchemyError:
                rollback_quietly(db)
                logger.exception("Failed to save the chat turn")
        try:
            usage = get_usage(db, caller.identifier, is_guest=caller.is_guest, is_admin=caller.is_admin)
        except SQLAlchemyError:
            rollback_quietly(db)
            logger.exception("Failed to read usage after answering")
    return QueryResponse(
        answer=answer, sources=sources, source_links=links or [], conversation_id=conversation_id, usage=usage
    )


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
async def query(
    request: Request,
    payload: QueryRequest,
    x_user_id: str | None = Header(default="", alias="X-User-Id"),
    x_forwarded_for: str | None = Header(default="", alias="X-Forwarded-For"),
    x_user_role: str | None = Header(default="", alias="X-User-Role"),
    retriever: Retriever = Depends(get_retriever),
    chat_model: ChatOpenAI = Depends(get_chat_model),
    db = Depends(get_db),
) -> QueryResponse:
    """Answer a doctrinal question, grounded in the scraped LDS source texts,
    while remembering the rest of the conversation. The whole answer comes
    back at once; /api/v1/query/stream sends it as it's written.

    Requires a valid X-API-Key header (see verify_api_key) and is capped at
    10 requests/minute per client IP (see the `limiter` set up above).

    X-User-Role: admin skips the daily question limit. It's only honoured
    because verify_api_key has already passed: the key lives on the Next.js
    server, which sets the header itself after checking the signed-in user's
    verified emails against ADMIN_EMAILS, and never forwards one from the
    browser. Guests can't be admins.
    """
    caller = resolve_caller(request, x_user_id, x_forwarded_for, x_user_role)
    refusal = admit_question(payload, caller, db)
    if refusal is not None:
        return refusal

    question = payload.messages[-1].content
    prepared = await prepare_answer(question, payload.messages[:-1], retriever, chat_model)
    if prepared.fixed_answer is not None:
        return record_answer(db, caller, payload, prepared.fixed_answer, prepared.sources, prepared.links)

    try:
        response = await chat_model.ainvoke(prepared.llm_messages)
    except Exception as exc:  # noqa: BLE001
        logger.error("Chat completion failed for question=%r", question, exc_info=exc)
        raise HTTPException(status_code=502, detail=ANSWER_FAILED_DETAIL) from exc
    return record_answer(db, caller, payload, str(response.content), prepared.sources, prepared.links)


@app.post("/admin/reindex", status_code=202, dependencies=[Depends(verify_api_key)])
@limiter.limit("6/minute")
async def admin_reindex(
    request: Request,
    x_user_id: str | None = Header(default="", alias="X-User-Id"),
    x_forwarded_for: str | None = Header(default="", alias="X-Forwarded-For"),
    x_user_role: str | None = Header(default="", alias="X-User-Role"),
) -> dict:
    """Rebuild pgvector_bm25's in-memory keyword index from the store, in the
    background (queries keep using the current index until it's swapped).
    Admins only: APP_API_KEY plus X-User-Role: admin with a user id.
    scripts/ingest.py calls this after a run. /health shows when it finished
    (retriever.keyword_index.refreshed_at)."""
    caller = resolve_caller(request, x_user_id, x_forwarded_for, x_user_role)
    if not caller.is_admin:
        raise HTTPException(status_code=403, detail="Admins only.")
    refresher: KeywordIndexRefresher | None = getattr(request.app.state, "keyword_refresher", None)
    if refresher is None:
        raise HTTPException(
            status_code=409,
            detail=f"RETRIEVER_BACKEND={getattr(request.app.state, 'retriever_backend', None)} "
            "has no in-memory keyword index to rebuild.",
        )
    started = refresher.start_rebuild()
    logger.info("Keyword index rebuild requested by an admin (%s).", "started" if started else "already running")
    return {
        "status": "started" if started else "already running",
        "chunks": (index_chunk_counts(request.app.state.retriever) or {}).get("vector"),
    }


# While the answer is being prepared (routing + retrieval) the stream sends an
# SSE comment this often, and once straight away, so the Vercel function and
# any proxy see bytes immediately and never wait on a silent connection.
STREAM_KEEPALIVE_SECONDS = 5.0
SSE_KEEPALIVE = ": keep-alive\n\n"


def sse_event(event: str, data: dict) -> str:
    """One server-sent event: `event: <name>` plus a JSON `data:` line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk_text(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    # Some models stream a list of content parts.
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


@app.post("/api/v1/query/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def query_stream(
    request: Request,
    payload: QueryRequest,
    x_user_id: str | None = Header(default="", alias="X-User-Id"),
    x_forwarded_for: str | None = Header(default="", alias="X-Forwarded-For"),
    x_user_role: str | None = Header(default="", alias="X-User-Role"),
    retriever: Retriever = Depends(get_retriever),
    chat_model: ChatOpenAI = Depends(get_chat_model),
    # scope="request" keeps the session open until the stream has finished,
    # so the turn can be saved after the last token.
    db = Depends(get_db, scope="request"),
):
    """Same as /api/v1/query, but the answer streams as server-sent events:

        event: token  data: {"text": "..."}       one per chunk of the answer
        event: done   data: <QueryResponse JSON>  sources, conversation_id, usage
        event: error  data: {"detail": "..."}     generation failed mid-answer

    Refusals (bad key, too long, out of questions) come back as ordinary
    JSON errors before any streaming starts. After that the response starts
    at once with an SSE comment (": keep-alive", repeated every few seconds
    while routing and retrieval run), so the first byte never waits for the
    answer; a retrieval failure then arrives as an error event.

    The turn is logged and saved only after the answer is complete. If the
    client stops the stream early, the question still counts (the model was
    used) but nothing is saved to history.
    """
    started = time.perf_counter()
    caller = resolve_caller(request, x_user_id, x_forwarded_for, x_user_role)
    refusal = admit_question(payload, caller, db)
    if refusal is not None:
        return refusal

    question = payload.messages[-1].content

    async def events():
        yield SSE_KEEPALIVE
        preparing = asyncio.create_task(prepare_answer(question, payload.messages[:-1], retriever, chat_model))
        try:
            while True:
                done, _ = await asyncio.wait({preparing}, timeout=STREAM_KEEPALIVE_SECONDS)
                if done:
                    break
                yield SSE_KEEPALIVE
            prepared = preparing.result()
        except HTTPException as exc:
            yield sse_event("error", {"detail": exc.detail})
            return
        except Exception:  # noqa: BLE001 - headers are sent; report it in the stream
            logger.exception("Preparing the answer failed for question=%r", question)
            yield sse_event("error", {"detail": ANSWER_FAILED_DETAIL})
            return
        finally:
            # The client went away while we were preparing.
            if not preparing.done():
                preparing.cancel()

        if prepared.fixed_answer is not None:
            yield sse_event("token", {"text": prepared.fixed_answer})
            response = record_answer(db, caller, payload, prepared.fixed_answer, prepared.sources, prepared.links)
            yield sse_event("done", response.model_dump(mode="json"))
            return

        parts: list[str] = []
        finished = failed = False
        try:
            async for chunk in chat_model.astream(prepared.llm_messages):
                text = _chunk_text(chunk)
                if text:
                    if not parts:
                        logger.info(
                            "First token after %.0f ms (%s)", (time.perf_counter() - started) * 1000, prepared.intent
                        )
                    parts.append(text)
                    yield sse_event("token", {"text": text})
            finished = True
        except Exception:  # noqa: BLE001
            failed = True
            logger.exception("Streaming chat completion failed for question=%r", question)
            yield sse_event("error", {"detail": ANSWER_FAILED_DETAIL})
            return
        finally:
            # Stopped by the client (disconnect cancels the stream): the model
            # was used, so count it, but don't save a half answer to history.
            # A failed answer isn't counted, same as the non-streaming endpoint.
            if not finished and not failed and parts:
                log_query(db, caller)

        response = record_answer(db, caller, payload, "".join(parts), prepared.sources, prepared.links)
        yield sse_event("done", response.model_dump(mode="json"))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Don't let proxies buffer the stream.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
