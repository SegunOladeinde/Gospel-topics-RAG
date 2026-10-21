"""Schema-level tests for POST /api/v1/query.

These tests verify Pydantic request validation *only* — they do not make
live OpenAI calls.  They patch the retriever and chat_model on app.state
so the endpoint can be exercised without any network dependencies.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.main import app

VALID_KEY = "test-key"


@pytest.fixture()
def client(monkeypatch):
    """Return a TestClient with APP_API_KEY wired to VALID_KEY."""
    monkeypatch.setenv("APP_API_KEY", VALID_KEY)
    with TestClient(app) as c:
        yield c


def _auth():
    return {"X-API-Key": VALID_KEY}


# ---------------------------------------------------------------------------
# Area 1: 422 Fix — long assistant messages must be accepted
# ---------------------------------------------------------------------------

def test_long_assistant_message_accepted(client):
    """A message body longer than the old 1500-char cap must not 422.

    Simulates the frontend replaying a previous AI response (which can
    easily be 3 000-5 000 characters) as part of the conversation history.
    """
    long_ai_reply = "A" * 4_000  # well above the old 1 500-char limit

    payload = {
        "messages": [
            {"role": "assistant", "content": long_ai_reply},
            {"role": "user", "content": "Can you expand on that?"},
        ]
    }

    # Patch the retriever to return no documents (safe 200, no LLM answer call)
    with patch.object(app.state, "retriever") as mock_retriever, \
         patch.object(app.state, "chat_model") as mock_chat:   # noqa: F841
        mock_retriever.ainvoke = AsyncMock(return_value=[])
        response = client.post("/api/v1/query", json=payload, headers=_auth())

    # Must be 200 (no context found path) rather than the old 422
    assert response.status_code == 200, response.json()


def test_message_at_new_limit_accepted(client):
    """A message of exactly 8 000 characters must be accepted.

    That's the cap for replayed history, e.g. a long earlier answer. The
    newest question has its own, lower limit (MAX_QUESTION_CHARS, see
    tests/test_history_and_usage.py::TestQuestionLength).
    """
    exactly_at_limit = "B" * 8_000

    payload = {"messages": [
        {"role": "user", "content": "What is faith?"},
        {"role": "assistant", "content": exactly_at_limit},
        {"role": "user", "content": "Tell me more."},
    ]}

    with patch.object(app.state, "retriever") as mock_retriever, \
         patch.object(app.state, "chat_model"):   # noqa: F841
        mock_retriever.ainvoke = AsyncMock(return_value=[])
        response = client.post("/api/v1/query", json=payload, headers=_auth())

    assert response.status_code == 200, response.json()


def test_message_over_new_limit_rejected(client):
    """A message of 8 001 characters must still be rejected with 422."""
    over_limit = "C" * 8_001

    payload = {"messages": [{"role": "user", "content": over_limit}]}
    response = client.post("/api/v1/query", json=payload, headers=_auth())

    assert response.status_code == 422, response.json()


# ---------------------------------------------------------------------------
# Area 1: 422 Fix — empty content is still rejected
# ---------------------------------------------------------------------------

def test_empty_content_rejected(client):
    """An empty message string must still produce a 422."""
    payload = {"messages": [{"role": "user", "content": ""}]}
    response = client.post("/api/v1/query", json=payload, headers=_auth())
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Existing validation: missing API key
# ---------------------------------------------------------------------------

def test_missing_api_key_rejected(client):
    """Requests without X-API-Key must get a 401."""
    payload = {"messages": [{"role": "user", "content": "Hello"}]}
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 401


def test_wrong_api_key_rejected(client):
    """Requests with a wrong X-API-Key must get a 401."""
    payload = {"messages": [{"role": "user", "content": "Hello"}]}
    response = client.post(
        "/api/v1/query", json=payload, headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Existing validation: empty messages list
# ---------------------------------------------------------------------------

def test_empty_messages_list_rejected(client):
    """An empty messages array must produce a 422 (min_length=1)."""
    payload = {"messages": []}
    response = client.post("/api/v1/query", json=payload, headers=_auth())
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Area 3: 3-Way intent routing
# ---------------------------------------------------------------------------
# All three tests patch route_question so we don't pay an OpenAI round-trip
# just to decide which branch to enter.

def test_chitchat_branch_skips_retrieval(client, monkeypatch):
    """CHITCHAT intent must bypass the retriever entirely."""
    from unittest.mock import patch as mock_patch
    import src.backend.main as backend

    with mock_patch.object(backend, "route_question", new=AsyncMock(return_value=backend.RouteDecision("CHITCHAT", "Hello!"))), \
         mock_patch.object(app.state, "retriever") as mock_ret, \
         mock_patch.object(app.state, "chat_model") as mock_chat:
        mock_ret.ainvoke = AsyncMock(return_value=[])           # should never be called
        mock_chat.ainvoke = AsyncMock(return_value=_make_ai_msg("Hello there!"))

        payload = {"messages": [{"role": "user", "content": "Hello!"}]}
        response = client.post("/api/v1/query", json=payload, headers=_auth())

    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []          # CHITCHAT returns empty sources
    mock_ret.ainvoke.assert_not_awaited() # retriever must NOT have been called


def test_rag_followup_branch_searches_the_standalone_question(client, monkeypatch):
    """RAG_FOLLOWUP must search the router's standalone question, not the raw follow-up."""
    from unittest.mock import patch as mock_patch
    import src.backend.main as backend
    from langchain_core.documents import Document

    fake_doc = Document(page_content="Chastity text.", metadata={"source": "s", "topic": "t"})
    decision = backend.RouteDecision("RAG_FOLLOWUP", "rewritten query")

    with mock_patch.object(backend, "route_question", new=AsyncMock(return_value=decision)), \
         mock_patch.object(app.state, "retriever") as mock_ret, \
         mock_patch.object(app.state, "chat_model") as mock_chat:
        mock_ret.ainvoke = AsyncMock(return_value=[fake_doc])
        mock_chat.ainvoke = AsyncMock(return_value=_make_ai_msg("Answer."))

        payload = {
            "messages": [
                {"role": "user", "content": "What is chastity?"},
                {"role": "assistant", "content": "It means..."},
                {"role": "user", "content": "Can you share scriptures?"},
            ]
        }
        response = client.post("/api/v1/query", json=payload, headers=_auth())

    assert response.status_code == 200
    assert response.json()["sources"] == ["s/t"]
    # The answer is built from the search for the REWRITTEN query.
    assert mock_ret.ainvoke.await_args_list[-1].args == ("rewritten query",)


def test_rag_new_topic_branch_searches_the_raw_question(client, monkeypatch):
    """RAG_NEW_TOPIC must search the question as asked, without a rewrite."""
    from unittest.mock import patch as mock_patch
    import src.backend.main as backend
    from langchain_core.documents import Document

    fake_doc = Document(page_content="Tithing text.", metadata={"source": "s", "topic": "tithing"})
    decision = backend.RouteDecision("RAG_NEW_TOPIC", "Tell me about tithing.")

    with mock_patch.object(backend, "route_question", new=AsyncMock(return_value=decision)), \
         mock_patch.object(app.state, "retriever") as mock_ret, \
         mock_patch.object(app.state, "chat_model") as mock_chat:
        mock_ret.ainvoke = AsyncMock(return_value=[fake_doc])
        mock_chat.ainvoke = AsyncMock(return_value=_make_ai_msg("Answer about tithing."))

        payload = {
            "messages": [
                {"role": "user", "content": "What is chastity?"},
                {"role": "assistant", "content": "It means..."},
                {"role": "user", "content": "Tell me about tithing."},
            ]
        }
        response = client.post("/api/v1/query", json=payload, headers=_auth())

    assert response.status_code == 200
    # Searched once, with the raw question (the speculative search is reused).
    mock_ret.ainvoke.assert_awaited_once_with("Tell me about tithing.")


def _make_ai_msg(text: str):
    """Return a minimal fake LangChain AI message for use in mock return values."""
    from langchain_core.messages import AIMessage
    return AIMessage(content=text)
