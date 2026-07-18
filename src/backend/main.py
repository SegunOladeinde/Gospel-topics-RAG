"""FastAPI backend for the LDS Doctrinal RAG System.

WHAT THIS FILE DOES, IN PLAIN ENGLISH
--------------------------------------
1.  When the server starts, it loads two pre-built search indexes from disk
    into memory: a FAISS index (good at "meaning" / semantic search) and a
    BM25 index (good at exact keyword matching). Both were built ahead of
    time by ``src/ingest.py`` from the scraped .txt files in ``data/raw/``.
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

import logging
import os
import pickle
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Server-side logging
# ---------------------------------------------------------------------------
# Configured once, at import time, before anything else might log. INFO so
# our own startup/shutdown messages are visible; ERROR-level entries (see the
# exception handlers below) always include the full stack trace via
# exc_info, regardless of this base level.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
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

# Note: in langchain >= 1.0, EnsembleRetriever was moved out of the main
# `langchain` package and into the separate `langchain_classic` package.
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

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
# src/ingest.py. If these ever drift apart, similarity search silently
# produces bad results (the query vector and the stored vectors would be in
# different "spaces").
EMBEDDING_MODEL = "text-embedding-3-small"

# The chat model that turns retrieved context + the user's question into a
# final answer. gpt-4o-mini is fast and inexpensive, per CLAUDE.md.
CHAT_MODEL = "gpt-4o-mini"

# How many chunks each retriever should fetch before we merge and hand them
# to the LLM. More chunks = more context, but also a bigger/slower prompt.
CHUNKS_PER_RETRIEVER = 4

# Comma-separated list of origins allowed to call this API (see CORS setup
# below), e.g. "https://app.example.com,https://staging.example.com" in
# production. Defaults to the local Next.js dev server so `uv run uvicorn
# ...` still works out of the box without any .env changes.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

SYSTEM_PROMPT = (
    "You are a helpful, detailed, and pastoral assistant. Provide deep "
    "doctrinal explanations with relevant scripture references. Your "
    "answers should be comprehensive but encouraging. At the end of every "
    "answer, gently remind the user that while this RAG system provides "
    "helpful context, they should pray and rely on the Holy Spirit to "
    "receive a personal witness of the truth. You are not affiliated with "
    "The Church of Jesus Christ of Latter-day Saints."
)


def load_retriever() -> BaseRetriever:
    """Load the FAISS and BM25 indexes from disk and combine them.

    FAISS finds chunks that are *semantically* similar to the question
    (e.g. it can match "sacred garments" to "temple clothing" even without
    shared keywords). BM25 finds chunks that share *exact words* with the
    question, which is great for names, titles, and specific terms that
    embeddings sometimes blur together.

    ``EnsembleRetriever`` runs both searches and merges/re-ranks their
    results, giving us the best of both approaches for every query.
    """
    if not INDEX_DIR.exists():
        raise RuntimeError(
            f"Index directory not found at {INDEX_DIR}. "
            "Run `uv run src/ingest.py` first to build the indexes."
        )

    # The FAISS index only stores vectors + text; it needs an embeddings
    # object at load time so it can embed *new* incoming questions using the
    # same model that was used to embed the stored chunks.
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        # This flag is required by newer versions of LangChain before they
        # will unpickle a FAISS index. It's safe here because this index
        # was built by our own trusted ingest.py, not downloaded from the
        # internet.
        allow_dangerous_deserialization=True,
    )
    faiss_retriever = vectorstore.as_retriever(
        search_kwargs={"k": CHUNKS_PER_RETRIEVER}
    )

    # The BM25 index was saved as a plain pickle file in src/ingest.py, so
    # we load it back the same way.
    with open(BM25_INDEX_PATH, "rb") as bm25_file:
        bm25_retriever: BM25Retriever = pickle.load(bm25_file)
    bm25_retriever.k = CHUNKS_PER_RETRIEVER

    # `weights` controls how much each retriever's ranking counts towards
    # the final merged ranking. 0.5/0.5 means "trust them equally".
    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )


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

    logger.info("Starting up: loading FAISS + BM25 indexes into memory...")
    # Attached directly to FastAPI's own per-app state object -- no separate
    # global dict to keep in sync with the app's lifecycle. Populated once,
    # here, and only ever read from inside request handlers afterward.
    app.state.retriever = load_retriever()
    app.state.chat_model = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    logger.info("Startup complete: indexes are loaded and ready for queries.")

    yield  # <-- the app runs here, handling requests, until it shuts down

    logger.info("Shutting down: releasing in-memory indexes.")


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

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
# Keyed by client IP, so each address gets its own request budget on the
# rate-limited routes below (see the /api/v1/query route).
limiter = Limiter(key_func=get_remote_address)
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
    if not expected_key or x_api_key != expected_key:
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

    role: str = Field(
        ...,
        description="Who sent this message: 'user' or 'assistant'.",
    )
    # max_length caps a single message at ~1500 characters -- plenty of room
    # for a pasted doctrinal quote or FamilySearch reference, but small
    # enough to block someone stuffing a huge blob into one turn to run up
    # the OpenAI bill.
    content: str = Field(
        ..., min_length=1, max_length=1500, description="The message text."
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


class QueryResponse(BaseModel):
    """The JSON body we send back to the frontend."""

    answer: str = Field(..., description="The generated, grounded answer.")
    sources: list[str] = Field(
        default_factory=list,
        description="Names of the source documents the answer was grounded in.",
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def format_context(documents: list[Document]) -> str:
    """Turn retrieved chunks into one text blob to paste into the prompt.

    Each chunk is labeled with its source so the model can (and is asked
    to) ground its answer in specific documents rather than blending
    everything together anonymously.
    """
    blocks = []
    for document in documents:
        source_label = describe_source(document)
        blocks.append(f"[Source: {source_label}]\n{document.page_content}")
    return "\n\n---\n\n".join(blocks)


def describe_source(document: Document) -> str:
    """Build a human-readable source label from a chunk's metadata.

    src/ingest.py tags every chunk with metadata like
    ``{"source": "topical_guide", "topic": "Faith"}`` (the folder name it
    came from, and the original filename without its extension). We combine
    those into a single readable string, e.g. "topical_guide/Faith".
    """
    metadata = document.metadata
    source = metadata.get("source", "unknown")
    topic = metadata.get("topic", "unknown")
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
    windowed = history[-MAX_HISTORY_MESSAGES:]

    total_chars = sum(len(message.content) for message in windowed)
    while total_chars > MAX_HISTORY_CHARS and windowed:
        dropped = windowed.pop(0)
        total_chars -= len(dropped.content)

    logger.info(
        "History trimmed: kept %d of %d message(s), %d char(s) total.",
        len(windowed),
        original_count,
        total_chars,
    )
    return windowed


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, str]:
    """Lightweight liveness check for load balancers/uptime monitors.

    Deliberately unauthenticated and unrate-limited, and does not touch the
    indexes or OpenAI -- it only proves the server process is up and
    responsive, not that every downstream dependency is healthy.
    """
    return {"status": "ok"}


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    """Answer a doctrinal question, grounded in the scraped LDS source texts,
    while remembering the rest of the conversation.

    Requires a valid X-API-Key header (see verify_api_key) and is capped at
    10 requests/minute per client IP (see the `limiter` set up above).

    Flow:
    1. Use the combined FAISS + BM25 retriever to fetch the most relevant
       text chunks for the *newest* question (the last message).
    2. Replay the earlier turns as real conversation history, and attach
       the retrieved context to the newest question only.
    3. Ask the chat model to answer using that context, in the pastoral
       persona, with the full history in view.
    4. Return the answer along with which source documents were used.
    """
    retriever = request.app.state.retriever
    chat_model = request.app.state.chat_model

    question = payload.messages[-1].content
    history = payload.messages[:-1]

    # Step 1: retrieve relevant chunks. Any failure here (e.g. a transient
    # OpenAI embeddings error) is treated as a server-side problem, not a
    # bad request from the client, so we respond with a 502 (Bad Gateway)
    # to indicate "we failed to talk to an upstream service".
    try:
        documents = await retriever.ainvoke(question)
    except Exception as exc:  # noqa: BLE001 - we deliberately want to catch anything here
        logger.error("Retrieval failed for question=%r", question, exc_info=exc)
        raise HTTPException(
            status_code=502,
            detail="Failed to search the document indexes. Please try again shortly.",
        ) from exc

    if not documents:
        # No relevant material found at all -- per CLAUDE.md we must not
        # let the model hallucinate an answer, so we say so directly
        # instead of calling the LLM with an empty context.
        return QueryResponse(
            answer=(
                "I couldn't find anything in the available LDS source "
                "material to answer that question."
            ),
            sources=[],
        )

    # Step 2: build the prompt from the retrieved context and the newest
    # question, then prepend the earlier turns so the model has the full
    # conversation in view (this is what lets it "remember" prior turns).
    # History is trimmed immediately before use, right here, so only the
    # tightly-budgeted window (not the full client-supplied history) ever
    # reaches the LLM.
    trimmed_history = trim_history(history)
    context_text = format_context(documents)
    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above."
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *(to_langchain_message(message) for message in trimmed_history),
        HumanMessage(content=user_prompt),
    ]

    # Step 3: ask the chat model to generate the final answer.
    try:
        response = await chat_model.ainvoke(messages)
    except Exception as exc:  # noqa: BLE001 - same reasoning as above
        logger.error("Chat completion failed for question=%r", question, exc_info=exc)
        raise HTTPException(
            status_code=502,
            detail="Failed to generate an answer. Please try again shortly.",
        ) from exc

    answer_text = str(response.content)

    # Step 4: return the answer plus which documents it was grounded in.
    return QueryResponse(
        answer=answer_text,
        sources=collect_unique_sources(documents),
    )
