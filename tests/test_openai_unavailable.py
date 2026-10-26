"""When OpenAI can't serve us (insufficient_quota, rate limit, 5xx), users get
a friendly "temporarily unavailable" message instead of a 502, the question
isn't counted against their limit, and it's logged at ERROR level (Render's
failure emails pick that up).

Same in-memory SQLite setup as tests/test_history_and_usage.py; the OpenAI
errors are built locally, nothing calls OpenAI.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.backend.main as backend
from src.backend.database import Base, UserQueryLog
from src.backend.main import ASSISTANT_UNAVAILABLE_DETAIL, app, get_db, openai_unavailable

ENGINE = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
Session = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)
VALID_KEY = "test-key"
FAITH_DOC = Document(page_content="Faith is...", metadata={"source": "topical_guide", "topic": "faith"})


def _openai_error(status: int, code: str | None = None) -> openai.APIStatusError:
    """The error the OpenAI SDK raises for an HTTP status (as LangChain passes it on)."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"error": {"message": "test", "type": code, "code": code}}
    response = httpx.Response(status, request=request, json=body)
    error_class = {429: openai.RateLimitError, 500: openai.InternalServerError,
                   503: openai.InternalServerError}.get(status, openai.APIStatusError)
    return error_class("test", response=response, body=body["error"])


QUOTA = _openai_error(429, "insufficient_quota")
UNAVAILABLE = [QUOTA, _openai_error(429, "rate_limit_exceeded"), _openai_error(500), _openai_error(503)]


@pytest.fixture(autouse=True)
def reset_tables():
    Base.metadata.create_all(bind=ENGINE)
    yield
    Base.metadata.drop_all(bind=ENGINE)


@pytest.fixture()
def db():
    session = Session()
    try:
        yield session
    finally:
        session.close()


def _override_get_db():
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("APP_API_KEY", VALID_KEY)
    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


def _post(client, path, *, user_id="", ip="", search=None, answer=None, stream=None):
    """POST one question; search, answer (ainvoke) and stream (astream) default to working."""
    headers = {"X-API-Key": VALID_KEY}
    if user_id:
        headers["X-User-Id"] = user_id
    if ip:
        headers["X-Forwarded-For"] = ip
    model = MagicMock()
    model.ainvoke = answer or AsyncMock(return_value=AIMessage(content="Faith is trust in Christ."))
    if stream is not None:
        model.astream = stream
    with patch.object(backend, "route_question",
                      new=AsyncMock(return_value=backend.RouteDecision("RAG_NEW_TOPIC", "test question"))), \
         patch.object(app.state, "retriever") as retriever, \
         patch.object(app.state, "chat_model", model):
        retriever.ainvoke = search or AsyncMock(return_value=[FAITH_DOC])
        return client.post(path, json={"messages": [{"role": "user", "content": "What is faith?"}]},
                           headers=headers)


def _failing_stream(error, after=()):
    async def astream(_messages):
        for text in after:
            yield AIMessageChunk(content=text)
        raise error
    return astream


def _events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _counted(db, identifier):
    return db.query(UserQueryLog).filter_by(user_identifier=identifier).count()


# --- recognising the error ----------------------------------------------------

@pytest.mark.parametrize("error", UNAVAILABLE)
def test_quota_rate_limit_and_server_errors_count_as_unavailable(error):
    assert openai_unavailable(error) is error


def test_found_when_wrapped_in_another_exception():
    try:
        try:
            raise QUOTA
        except openai.RateLimitError as exc:
            raise RuntimeError("embedding failed") from exc
    except RuntimeError as wrapped:
        assert openai_unavailable(wrapped) is QUOTA


@pytest.mark.parametrize("error", [_openai_error(400), _openai_error(401), RuntimeError("db down"), None])
def test_other_failures_are_not_unavailable(error):
    assert openai_unavailable(error) is None


# --- /api/v1/query --------------------------------------------------------------

@pytest.mark.parametrize("error", UNAVAILABLE)
def test_chat_failure_is_a_friendly_503_not_a_502(client, error):
    r = _post(client, "/api/v1/query", user_id="q1", answer=AsyncMock(side_effect=error))
    assert r.status_code == 503
    assert r.json() == {"detail": ASSISTANT_UNAVAILABLE_DETAIL}


def test_embedding_failure_during_retrieval_is_a_friendly_503(client):
    r = _post(client, "/api/v1/query", user_id="q2", search=AsyncMock(side_effect=QUOTA))
    assert r.status_code == 503
    assert r.json()["detail"] == ASSISTANT_UNAVAILABLE_DETAIL


def test_other_failures_are_still_502(client):
    r = _post(client, "/api/v1/query", user_id="q3", answer=AsyncMock(side_effect=_openai_error(400)))
    assert r.status_code == 502
    assert r.json()["detail"] == backend.ANSWER_FAILED_DETAIL


def test_unavailable_answer_is_not_counted(client, db):
    _post(client, "/api/v1/query", user_id="q4", answer=AsyncMock(side_effect=QUOTA))
    _post(client, "/api/v1/query", user_id="q4", search=AsyncMock(side_effect=QUOTA))
    assert _counted(db, "q4") == 0
    usage = client.get("/api/v1/usage", headers={"X-API-Key": VALID_KEY, "X-User-Id": "q4"}).json()["usage"]
    assert usage["used"] == 0


def test_guest_can_still_ask_their_one_question_afterwards(client):
    failed = _post(client, "/api/v1/query", ip="5.6.7.8", answer=AsyncMock(side_effect=QUOTA))
    assert failed.status_code == 503
    answered = _post(client, "/api/v1/query", ip="5.6.7.8")
    assert answered.status_code == 200
    assert answered.json()["answer"] == "Faith is trust in Christ."


def test_logged_at_error_level(client, caplog):
    with caplog.at_level(logging.ERROR, logger=backend.logger.name):
        _post(client, "/api/v1/query", user_id="q5", answer=AsyncMock(side_effect=QUOTA))
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and "OpenAI unavailable" in r.getMessage()]
    assert len(errors) == 1
    assert "insufficient_quota" in errors[0].getMessage()
    assert "429" in errors[0].getMessage()


# --- /api/v1/query/stream ---------------------------------------------------------

def test_stream_sends_the_friendly_message_and_does_not_count(client, db):
    r = _post(client, "/api/v1/query/stream", user_id="s1", stream=_failing_stream(QUOTA))
    assert r.status_code == 200
    assert _events(r.text) == [("error", {"detail": ASSISTANT_UNAVAILABLE_DETAIL})]
    assert _counted(db, "s1") == 0


def test_stream_failing_mid_answer_is_not_counted(client, db):
    r = _post(client, "/api/v1/query/stream", user_id="s2",
              stream=_failing_stream(_openai_error(500), after=("Faith is ",)))
    assert _events(r.text) == [("token", {"text": "Faith is "}), ("error", {"detail": ASSISTANT_UNAVAILABLE_DETAIL})]
    assert _counted(db, "s2") == 0


def test_stream_embedding_failure_sends_the_friendly_message(client, db):
    r = _post(client, "/api/v1/query/stream", user_id="s3", search=AsyncMock(side_effect=QUOTA),
              stream=_failing_stream(AssertionError("not reached")))
    assert _events(r.text) == [("error", {"detail": ASSISTANT_UNAVAILABLE_DETAIL})]
    assert _counted(db, "s3") == 0
