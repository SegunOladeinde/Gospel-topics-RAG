"""Tests for POST /api/v1/query/stream (server-sent events).

Same in-memory SQLite setup as tests/test_history_and_usage.py; the chat
model's astream() is mocked, so nothing calls OpenAI.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.requests import Request

import src.backend.main as backend
from src.backend.database import Base, ChatConversation, ChatMessage, UserQueryLog
from src.backend.main import QueryRequest, app, get_db
from src.backend.usage import USER_LIMIT

ENGINE = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
Session = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)
VALID_KEY = "test-key"
FAITH_DOC = Document(page_content="Faith is...", metadata={"source": "topical_guide", "topic": "faith"})


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


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _model(chunks=("Faith is ", "trust in Christ."), fail_after=None):
    """A chat model whose astream yields the given chunks (optionally raising)."""
    async def astream(_messages):
        for i, text in enumerate(chunks):
            if fail_after is not None and i == fail_after:
                raise RuntimeError("model went away")
            yield AIMessageChunk(content=text)

    model = MagicMock()
    model.astream = astream
    return model


def _events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _stream(client, question="What is faith?", *, user_id="", ip="", docs=(FAITH_DOC,), model=None):
    headers = {"X-API-Key": VALID_KEY}
    if user_id:
        headers["X-User-Id"] = user_id
    if ip:
        headers["X-Forwarded-For"] = ip
    with patch.object(backend, "classify_intent", new=AsyncMock(return_value="RAG_NEW_TOPIC")), \
         patch.object(app.state, "retriever") as retriever, \
         patch.object(app.state, "chat_model", model or _model()):
        retriever.ainvoke = AsyncMock(return_value=list(docs))
        return client.post(
            "/api/v1/query/stream",
            json={"messages": [{"role": "user", "content": question}]},
            headers=headers,
        )


def test_streams_tokens_then_done(client):
    r = _stream(client, user_id="s1")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    assert [name for name, _ in events] == ["token", "token", "done"]
    assert "".join(data["text"] for name, data in events if name == "token") == "Faith is trust in Christ."

    done = events[-1][1]
    assert done["answer"] == "Faith is trust in Christ."
    assert done["sources"] == ["topical_guide/faith"]
    assert done["source_links"][0]["label"] == "Topical Guide: Faith"
    assert done["conversation_id"]
    assert done["usage"]["remaining"] == USER_LIMIT - 1


def test_turn_is_saved_after_the_stream_completes(client, db):
    done = _events(_stream(client, user_id="s2").text)[-1][1]
    messages = db.query(ChatMessage).filter_by(conversation_id=done["conversation_id"]).all()
    assert sorted((m.role, m.content) for m in messages) == [
        ("assistant", "Faith is trust in Christ."),
        ("user", "What is faith?"),
    ]
    assert db.query(UserQueryLog).filter_by(user_identifier="s2").count() == 1


def test_guest_stream_is_not_saved(client, db):
    done = _events(_stream(client, ip="9.8.7.6").text)[-1][1]
    assert done["conversation_id"] is None
    assert db.query(ChatConversation).count() == 0


def test_no_material_found_streams_the_fixed_reply(client):
    events = _events(_stream(client, user_id="s3", docs=()).text)
    assert [name for name, _ in events] == ["token", "done"]
    assert events[0][1]["text"] == backend.NO_MATERIAL_ANSWER
    assert events[1][1]["sources"] == []


def test_refusals_are_plain_json_before_any_streaming(client):
    for _ in range(USER_LIMIT):
        _stream(client, user_id="s4")
    blocked = _stream(client, user_id="s4")
    assert blocked.status_code == 429
    assert blocked.headers["content-type"].startswith("application/json")
    assert blocked.json()["detail"] == "USER_LIMIT_REACHED"
    assert blocked.json()["reset_at"]

    too_long = _stream(client, "a" * 2001, user_id="s5")
    assert too_long.status_code == 400
    assert "2,000" in too_long.json()["detail"]


def test_requires_api_key(client):
    r = client.post("/api/v1/query/stream", json={"messages": [{"role": "user", "content": "hi"}]},
                    headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_failure_mid_stream_sends_an_error_and_saves_nothing(client, db):
    events = _events(_stream(client, user_id="s6", model=_model(fail_after=1)).text)
    assert [name for name, _ in events] == ["token", "error"]
    assert events[-1][1]["detail"] == backend.ANSWER_FAILED_DETAIL
    # Not counted and not saved, like a failed non-streaming answer.
    assert db.query(UserQueryLog).filter_by(user_identifier="s6").count() == 0
    assert db.query(ChatMessage).count() == 0


@pytest.mark.anyio
async def test_stopping_early_counts_the_question_but_saves_nothing(db):
    """A client disconnect closes the stream mid-answer (simulated with aclose)."""
    request = Request({
        "type": "http", "method": "POST", "path": "/api/v1/query/stream",
        "headers": [], "client": ("1.2.3.4", 1234), "query_string": b"",
    })
    retriever = MagicMock()
    retriever.ainvoke = AsyncMock(return_value=[FAITH_DOC])
    with patch.object(backend, "classify_intent", new=AsyncMock(return_value="RAG_NEW_TOPIC")):
        response = await backend.query_stream(
            request=request,
            payload=QueryRequest(messages=[{"role": "user", "content": "What is faith?"}]),
            x_user_id="s7", x_forwarded_for="", x_user_role="",
            retriever=retriever, chat_model=_model(), db=db,
        )
        stream = response.body_iterator
        first = await stream.__anext__()
        assert "event: token" in first
        await stream.aclose()  # what a disconnect does

    assert db.query(UserQueryLog).filter_by(user_identifier="s7").count() == 1
    assert db.query(ChatMessage).count() == 0
