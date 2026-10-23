"""Tests for POST /api/v1/query/stream (server-sent events).

Same in-memory SQLite setup as tests/test_history_and_usage.py; the chat
model's astream() is mocked, so nothing calls OpenAI.
"""

import asyncio
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
    """The named events, skipping ": keep-alive" comments."""
    events = []
    for block in body.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _stream(client, question="What is faith?", *, user_id="", ip="", docs=(FAITH_DOC,), model=None, search=None):
    headers = {"X-API-Key": VALID_KEY}
    if user_id:
        headers["X-User-Id"] = user_id
    if ip:
        headers["X-Forwarded-For"] = ip
    with patch.object(backend, "route_question", new=AsyncMock(return_value=backend.RouteDecision("RAG_NEW_TOPIC", "test question"))), \
         patch.object(app.state, "retriever") as retriever, \
         patch.object(app.state, "chat_model", model or _model()):
        retriever.ainvoke = search or AsyncMock(return_value=list(docs))
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


def test_stream_starts_with_a_keep_alive_before_any_answer(client):
    r = _stream(client, user_id="k1")
    assert r.text.startswith(backend.SSE_KEEPALIVE)


def test_keep_alives_repeat_while_the_answer_is_prepared(client, monkeypatch):
    monkeypatch.setattr(backend, "STREAM_KEEPALIVE_SECONDS", 0.05)

    async def slow_search(_query):
        await asyncio.sleep(0.3)
        return [FAITH_DOC]

    r = _stream(client, user_id="k2", search=slow_search)
    assert r.text.count(backend.SSE_KEEPALIVE) >= 3
    assert [name for name, _ in _events(r.text)] == ["token", "token", "done"]


def test_retrieval_failure_after_the_keep_alive_is_an_error_event(client, db):
    r = _stream(client, user_id="k3", search=AsyncMock(side_effect=RuntimeError("db down")))
    assert r.status_code == 200
    events = _events(r.text)
    assert [name for name, _ in events] == ["error"]
    assert "search the document indexes" in events[0][1]["detail"]
    assert db.query(UserQueryLog).filter_by(user_identifier="k3").count() == 0


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
    with patch.object(backend, "route_question", new=AsyncMock(return_value=backend.RouteDecision("RAG_NEW_TOPIC", "test question"))):
        response = await backend.query_stream(
            request=request,
            payload=QueryRequest(messages=[{"role": "user", "content": "What is faith?"}]),
            x_user_id="s7", x_forwarded_for="", x_user_role="",
            retriever=retriever, chat_model=_model(), db=db,
        )
        stream = response.body_iterator
        assert await stream.__anext__() == backend.SSE_KEEPALIVE
        first = await stream.__anext__()
        assert "event: token" in first
        await stream.aclose()  # what a disconnect does

    assert db.query(UserQueryLog).filter_by(user_identifier="s7").count() == 1
    assert db.query(ChatMessage).count() == 0


# --- only the sources used, references checked -------------------------------

LIBERTY_DOC = Document(
    page_content="Liberty Jail held Joseph Smith in 1838. The Lord answered: \"thine adversity and thine "
                 "afflictions shall be but a small moment\" (Doctrine and Covenants 121:7).",
    metadata={"source": "church_history_topics", "topic": "liberty-jail"},
)
LIBERTY_TG = Document(page_content="Liberty: freedom.", metadata={"source": "topical_guide", "topic": "liberty"})


def test_stream_hides_the_used_line_and_shows_only_used_sources(client, db):
    model = _model(chunks=("Joseph was held in Liberty Jail. The Lord said his trials were \"but a small ",
                           "moment\" (D&C 121:8).\n\nUS", "ED: [", "2]"))
    r = _stream(client, "What happened at Liberty Jail?", user_id="s9", docs=(LIBERTY_TG, LIBERTY_DOC), model=model)
    events = _events(r.text)
    streamed = "".join(data["text"] for name, data in events if name == "token")
    assert "USED" not in streamed and "[2]" not in streamed

    done = events[-1][1]
    # The verse is corrected to the one the passage gives next to the quotation.
    assert done["answer"].endswith('"but a small moment" (Doctrine and Covenants 121:7).')
    assert done["sources"] == ["church_history_topics/liberty-jail"]  # not "Topical Guide: Liberty"
    saved = db.query(ChatMessage).filter_by(conversation_id=done["conversation_id"], role="assistant").one()
    assert saved.content == done["answer"]
    assert [s["id"] for s in saved.sources] == ["church_history_topics/liberty-jail"]


def test_without_a_used_line_the_first_sources_are_shown(client):
    docs = [Document(page_content=f"Passage {i}.", metadata={"source": "topical_guide", "topic": f"t{i}"})
            for i in range(5)]
    done = _events(_stream(client, user_id="s10", docs=docs).text)[-1][1]
    assert done["sources"] == ["topical_guide/t0", "topical_guide/t1", "topical_guide/t2"]


def test_non_streaming_answer_is_finalised_the_same_way(client):
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=AIMessageChunk(content="It was a hard winter.\nUSED: [2]"))
    with patch.object(backend, "route_question", new=AsyncMock(return_value=backend.RouteDecision("RAG_NEW_TOPIC", "q"))), \
         patch.object(app.state, "retriever") as retriever, patch.object(app.state, "chat_model", model):
        retriever.ainvoke = AsyncMock(return_value=[LIBERTY_TG, LIBERTY_DOC])
        body = client.post("/api/v1/query", json={"messages": [{"role": "user", "content": "Liberty Jail?"}]},
                           headers={"X-API-Key": VALID_KEY}).json()
    assert body["answer"] == "It was a hard winter."
    assert body["sources"] == ["church_history_topics/liberty-jail"]
