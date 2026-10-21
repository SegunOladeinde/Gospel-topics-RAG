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
from src.backend.retrieval import Retriever, load_local_retriever
from src.backend.source_links import SourceLink, source_link
from src.backend.usage import Usage, get_usage
from dataclasses import dataclass
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
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, computed_field

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

STANDALONE_QUESTION_PROMPT = PromptTemplate.from_template(
    "Given the conversation history below and the follow-up question, "
    "rewrite the follow-up into a single, self-contained question that "
    "captures all necessary context for searching a document database. "
    "Output ONLY the rewritten question — no preamble, no explanation.\n\n"
    "History:\n{history}\n\n"
    "Follow-up: {question}\n\n"
    "Standalone question:"
)

# System prompt used exclusively inside classify_intent().
# Kept deliberately narrow so the model outputs only one of the three tokens.
INTENT_CLASSIFIER_PROMPT = """You are an intent classification engine.
Analyse the user's latest question in the context of the conversation history
and return EXACTLY one of these three strings — nothing else:

  CHITCHAT      — The message is ONLY a greeting, thanks, compliment or
                  casual small-talk, with no question or request at all
                  (e.g. "Hello", "Thanks!", "Great answer").
                  If small-talk is combined with ANY question or request
                  (e.g. "Thanks! And what about tithing?"), it is NOT
                  CHITCHAT: use RAG_FOLLOWUP or RAG_NEW_TOPIC.

  RAG_FOLLOWUP  — The user is asking a question that requires context from
                  the conversation history to make sense (e.g. "Why should I
                  keep it?", "Can you share scriptures about that?").

  RAG_NEW_TOPIC — The user is asking a completely new gospel question that is
                  unrelated to the previous exchange.

Output the label only. No punctuation, no explanation."""

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


def load_retriever() -> Retriever:
    """Load the passage indexes from data/index/ (see src/backend/retrieval).

    Semantic search over FAISS finds passages that *mean* the same as the
    question (it can match "sacred garments" to "temple clothing" without
    shared words); keyword search over BM25 finds exact names and terms that
    embeddings can blur. Both run for every question and their rankings are
    fused.
    """
    if not INDEX_DIR.exists():
        raise RuntimeError(
            f"Index directory not found at {INDEX_DIR}. "
            "Run `uv run scripts/ingest.py` first to build the indexes."
        )
    _verify_bm25_hash(BM25_INDEX_PATH)
    # The embeddings client is created once here and reused for every
    # question (it keeps its HTTP connection pool).
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, timeout=10.0)
    return load_local_retriever(INDEX_DIR, embeddings, k=CHUNKS_PER_RETRIEVER)


REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "APP_API_KEY")
OPTIONAL_ENV_VARS = (
    "DATABASE_URL",
    "ALLOWED_ORIGINS",
    "BM25_INDEX_SHA256",
    "RATE_LIMIT_FAIL_CLOSED",
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
    logger.info("Starting up: loading FAISS + BM25 indexes into memory...")
    # Attached directly to FastAPI's own per-app state object -- no separate
    # global dict to keep in sync with the app's lifecycle. Populated once,
    # here, and only ever read from inside request handlers afterward.
    app.state.retriever = await asyncio.to_thread(load_retriever)  # Issue #4: never block the event loop
    log_index_counts(app.state.retriever)
    app.state.chat_model = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    # Creates user_query_logs if missing. None when DATABASE_URL is unset or
    # invalid, which disables rate limiting.
    app.state.db_session_factory = await asyncio.to_thread(init_db)
    logger.info("Startup complete: indexes are loaded and ready for queries.")

    yield  # <-- the app runs here, handling requests, until it shuts down

    logger.info("Shutting down: releasing in-memory indexes.")


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

    @computed_field(description="Each source with a readable label and its churchofjesuschrist.org URL.")
    @property
    def source_links(self) -> list[SourceLink]:
        return [source_link(source) for source in self.sources]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
async def rewrite_to_standalone(
    question: str,
    history: list[Message],
    chat_model: ChatOpenAI,
) -> str:
    """Rewrite a follow-up question into a self-contained search query.

    Called only when there is prior history.  The resulting string is fed
    into FAISS/BM25 instead of the raw user input, so retrieval receives
    meaningful, context-rich text rather than bare pronouns or references
    (e.g. "it" or "that topic").

    Falls back to the original question if the LLM call fails, so retrieval
    always proceeds regardless of any transient upstream error.
    """
    history_text = "\n".join(
        f"{m.role.capitalize()}: {m.content}" for m in history
    )
    prompt = STANDALONE_QUESTION_PROMPT.invoke(
        {"history": history_text, "question": question}
    )
    try:
        response = await chat_model.ainvoke(prompt)
        rewritten = str(response.content).strip()
        logger.info(
            "Standalone rewrite: %r → %r", question, rewritten
        )
        return rewritten
    except Exception:  # noqa: BLE001 - fall back gracefully, never block retrieval
        logger.warning(
            "Standalone rewrite failed; falling back to raw question.",
            exc_info=True,
        )
        return question


_VALID_INTENTS = {"CHITCHAT", "RAG_FOLLOWUP", "RAG_NEW_TOPIC"}

# Issue #5: in-process cache for classify_intent, keyed on the question plus
# a digest of the last INTENT_CACHE_TURNS messages: the same words can be a
# follow-up in one conversation and a new topic in another. FIFO eviction at
# _INTENT_CACHE_MAX entries.
_INTENT_CACHE: dict[tuple[str, str], str] = {}
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


async def classify_intent(
    question: str,
    history: list[Message],
    chat_model: ChatOpenAI,
) -> str:
    """Return the routing label for the user's latest question.

    Asks the LLM to inspect the question against the conversation history
    and return exactly one of: ``CHITCHAT``, ``RAG_FOLLOWUP``, or
    ``RAG_NEW_TOPIC``.  Any unexpected response is treated as
    ``RAG_NEW_TOPIC`` so we always attempt a retrieval rather than silently
    swallowing the question.

    ``CHITCHAT``      — greeting / thanks / casual compliment; skip retrieval.
    ``RAG_FOLLOWUP``  — context-dependent follow-up; rewrite then retrieve.
    ``RAG_NEW_TOPIC`` — fresh gospel question; retrieve against raw question.

    CHITCHAT is decided in code, not by the model: pure small talk (see
    is_pure_small_talk) skips the OpenAI call entirely, and a CHITCHAT label
    from the model for anything else is overridden, so a question tucked
    behind a "thanks" still gets a grounded answer.
    """
    if is_pure_small_talk(question):
        return "CHITCHAT"

    # Cache lookup — skip the OpenAI call for a question already classified
    # in the same recent context.
    _cache_key = _intent_cache_key(question, history)
    if _cache_key in _INTENT_CACHE:
        cached = _INTENT_CACHE[_cache_key]
        logger.info("Intent cache hit: %r -> %r", question, cached)
        return cached

    history_text = "\n".join(
        f"{m.role.capitalize()}: {m.content}" for m in history
    ) if history else "(no prior history)"

    user_block = (
        f"Conversation history:\n{history_text}\n\n"
        f"Latest question: {question}"
    )
    try:
        response = await chat_model.ainvoke([
            SystemMessage(content=INTENT_CLASSIFIER_PROMPT),
            HumanMessage(content=user_block),
        ])
        label = str(response.content).strip().upper()
        # Strip any accidental punctuation the model might have added
        label = label.rstrip(".!,;")
        if label not in _VALID_INTENTS:
            logger.warning(
                "classify_intent returned unexpected label %r; defaulting to RAG_NEW_TOPIC.",
                label,
            )
            label = "RAG_NEW_TOPIC"
        elif label == "CHITCHAT":
            # Not pure small talk (checked above), so there's a real question
            # in it. With history it may lean on earlier turns; the follow-up
            # branch rewrites it into a standalone query either way.
            logger.info("Overriding CHITCHAT for a message with a question in it: %r", question)
            label = "RAG_FOLLOWUP" if history else "RAG_NEW_TOPIC"
    except Exception:  # noqa: BLE001 - never block the pipeline
        logger.warning(
            "classify_intent failed; defaulting to RAG_NEW_TOPIC.",
            exc_info=True,
        )
        label = "RAG_NEW_TOPIC"

    logger.info("Intent classified as %r for question=%r", label, question)

    # Store in cache; evict oldest entry (FIFO) when at capacity.
    if len(_INTENT_CACHE) >= _INTENT_CACHE_MAX:
        _INTENT_CACHE.pop(next(iter(_INTENT_CACHE)))
    _INTENT_CACHE[_cache_key] = label
    return label


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

    sources: list[str]
    llm_messages: list | None = None
    fixed_answer: str | None = None


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

    1. Classify the intent before touching FAISS/BM25, so chit-chat skips
       retrieval and a topic switch doesn't drag in the old subject.
    2. For a follow-up, rewrite it into a standalone query; a new topic is
       searched as asked.
    3. Retrieve, and build the prompt from the retrieved passages with the
       earlier turns replayed as real conversation history.
    Raises HTTPException(502) if retrieval fails.
    """
    trimmed_history = trim_history(history)
    intent = await classify_intent(question, trimmed_history, chat_model)

    if intent == "CHITCHAT":
        # No retrieval and no citations: just reply in kind.
        return PreparedAnswer(
            sources=[],
            llm_messages=[
                SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
                *(to_langchain_message(m) for m in trimmed_history),
                HumanMessage(content=question),
            ],
        )

    if intent == "RAG_FOLLOWUP":
        retrieval_query = await rewrite_to_standalone(question, trimmed_history, chat_model)
    else:  # RAG_NEW_TOPIC, and any unexpected label
        retrieval_query = question

    try:
        documents = await retriever.ainvoke(retrieval_query)
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval failed for question=%r", question, exc_info=exc)
        raise HTTPException(
            status_code=502,
            detail="Failed to search the document indexes. Please try again shortly.",
        ) from exc

    if not documents:
        # No relevant material found: don't let the model improvise.
        return PreparedAnswer(sources=[], fixed_answer=NO_MATERIAL_ANSWER)

    # The labels here are for the model only; the system prompt tells it
    # never to refer to "passages" or "context" in its answer.
    user_prompt = (
        f"Reference passages:\n{format_context(documents)}\n\n"
        f"Question: {question}\n\n"
        "Answer from these passages only, citing them naturally by name."
    )
    return PreparedAnswer(
        sources=collect_unique_sources(documents),
        llm_messages=[
            SystemMessage(content=SYSTEM_PROMPT),
            *(to_langchain_message(m) for m in trimmed_history),
            HumanMessage(content=user_prompt),
        ],
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
    db, caller: Caller, payload: QueryRequest, answer: str, sources: list[str]
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
                    payload.messages[-1].content, answer, sources,
                )
            except SQLAlchemyError:
                rollback_quietly(db)
                logger.exception("Failed to save the chat turn")
        try:
            usage = get_usage(db, caller.identifier, is_guest=caller.is_guest, is_admin=caller.is_admin)
        except SQLAlchemyError:
            rollback_quietly(db)
            logger.exception("Failed to read usage after answering")
    return QueryResponse(answer=answer, sources=sources, conversation_id=conversation_id, usage=usage)


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
        return record_answer(db, caller, payload, prepared.fixed_answer, prepared.sources)

    try:
        response = await chat_model.ainvoke(prepared.llm_messages)
    except Exception as exc:  # noqa: BLE001
        logger.error("Chat completion failed for question=%r", question, exc_info=exc)
        raise HTTPException(status_code=502, detail=ANSWER_FAILED_DETAIL) from exc
    return record_answer(db, caller, payload, str(response.content), prepared.sources)


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

    Refusals (bad key, too long, out of questions, retrieval failure) come
    back as ordinary JSON errors before any streaming starts.

    The turn is logged and saved only after the answer is complete. If the
    client stops the stream early, the question still counts (the model was
    used) but nothing is saved to history.
    """
    caller = resolve_caller(request, x_user_id, x_forwarded_for, x_user_role)
    refusal = admit_question(payload, caller, db)
    if refusal is not None:
        return refusal

    question = payload.messages[-1].content
    prepared = await prepare_answer(question, payload.messages[:-1], retriever, chat_model)

    async def events():
        if prepared.fixed_answer is not None:
            yield sse_event("token", {"text": prepared.fixed_answer})
            response = record_answer(db, caller, payload, prepared.fixed_answer, prepared.sources)
            yield sse_event("done", response.model_dump(mode="json"))
            return

        parts: list[str] = []
        finished = failed = False
        try:
            async for chunk in chat_model.astream(prepared.llm_messages):
                text = _chunk_text(chunk)
                if text:
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

        response = record_answer(db, caller, payload, "".join(parts), prepared.sources)
        yield sse_event("done", response.model_dump(mode="json"))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Don't let proxies buffer the stream.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
