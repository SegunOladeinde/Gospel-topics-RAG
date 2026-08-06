"""Shared pytest fixtures for the LDS RAG test suite.

WHY THIS FILE EXISTS
--------------------
The FastAPI app uses a ``lifespan`` context manager that loads the real FAISS
and BM25 indexes from ``data/index/`` at server startup.  That directory does
not exist in CI (or any fresh clone), so ``TestClient(app)`` would immediately
crash with::

    RuntimeError: Index directory not found at data/index

These session-scoped fixtures intercept the two expensive startup calls *before*
``TestClient`` triggers the lifespan, replacing them with lightweight stubs:

- ``load_retriever``  → returns a ``MagicMock`` (never touches disk)
- ``ChatOpenAI(...)`` → returns a ``MagicMock`` (never contacts OpenAI)

The individual test modules can still patch ``app.state.retriever`` and
``app.state.chat_model`` at the attribute level for per-test assertions, as
they do today.  This conftest patch operates one level higher — at *import /
instantiation* time — which is what CI was missing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------

def _make_stub_retriever() -> MagicMock:
    """Return a minimal retriever stub that always yields an empty document list."""
    stub = MagicMock()
    stub.ainvoke = AsyncMock(return_value=[])
    return stub


def _make_stub_chat_model() -> MagicMock:
    """Return a minimal chat-model stub whose ainvoke returns a blank AIMessage."""
    from langchain_core.messages import AIMessage

    stub = MagicMock()
    stub.ainvoke = AsyncMock(return_value=AIMessage(content=""))
    return stub


# ---------------------------------------------------------------------------
# Session-scoped patch — runs once for the entire test session
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _patch_lifespan_dependencies():
    """Intercept load_retriever and ChatOpenAI before lifespan fires.

    ``autouse=True`` means every test in the suite automatically gets this
    fixture, so no individual test needs to import or reference it.

    ``scope="session"`` means the patch is applied once and kept for the
    whole test run — avoiding repeated disk-check failures on every function.
    """
    with (
        patch(
            "src.backend.main.load_retriever",
            return_value=_make_stub_retriever(),
        ),
        patch(
            "src.backend.main.ChatOpenAI",
            return_value=_make_stub_chat_model(),
        ),
    ):
        yield
