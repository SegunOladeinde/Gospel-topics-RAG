"""Tests for chat history (src/backend/history.py), question allowances
(src/backend/usage.py) and the endpoints that expose them.

Uses an in-memory SQLite database injected via a get_db override, like
tests/test_rate_limiting.py, so no Supabase credentials are needed.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.backend.main as backend
from src.backend.database import Base, ChatConversation, ChatMessage, UserQueryLog
from src.backend.history import TITLE_MAX_CHARS, make_title
from src.backend.main import app, get_db
from src.backend.usage import USER_LIMIT, USER_WINDOW, get_usage

ENGINE = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
Session = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)
VALID_KEY = "test-key"
NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


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


def _headers(user_id="", role="", ip="", api_key=VALID_KEY):
    headers = {"X-API-Key": api_key}
    if user_id:
        headers["X-User-Id"] = user_id
    if role:
        headers["X-User-Role"] = role
    if ip:
        headers["X-Forwarded-For"] = ip
    return headers


def _ask(client, question="What is faith?", *, conversation_id=None, history=(), docs=None, **identity):
    """POST one question with the RAG pipeline mocked out."""
    payload = {"messages": [*history, {"role": "user", "content": question}]}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if docs is None:
        docs = [Document(page_content="Faith is...", metadata={"source": "topical_guide", "topic": "faith"})]
    with patch.object(backend, "classify_intent", new=AsyncMock(return_value="RAG_NEW_TOPIC")), \
         patch.object(app.state, "retriever") as retriever, \
         patch.object(app.state, "chat_model") as chat_model:
        retriever.ainvoke = AsyncMock(return_value=docs)
        chat_model.ainvoke = AsyncMock(return_value=AIMessage(content=f"Answer to: {question}"))
        return client.post("/api/v1/query", json=payload, headers=_headers(**identity))


def _log(db, identifier, *, at, is_guest=False, is_admin=False):
    db.add(UserQueryLog(user_identifier=identifier, is_guest=is_guest, is_admin=is_admin, created_at=at))
    db.commit()


# =====================================================================
# get_usage()
# =====================================================================

class TestGetUsage:
    def test_guest_has_one_question_and_no_reset(self, db):
        assert get_usage(db, "1.2.3.4", is_guest=True, is_admin=False).remaining == 1
        _log(db, "1.2.3.4", at=NOW, is_guest=True)
        usage = get_usage(db, "1.2.3.4", is_guest=True, is_admin=False)
        assert (usage.used, usage.remaining, usage.reset_at) == (1, 0, None)

    def test_user_with_questions_left_has_no_reset(self, db):
        for minutes in range(3):
            _log(db, "u1", at=NOW - timedelta(minutes=minutes))
        usage = get_usage(db, "u1", is_guest=False, is_admin=False, now=NOW)
        assert (usage.limit, usage.used, usage.remaining, usage.reset_at) == (USER_LIMIT, 3, 4, None)

    def test_user_out_of_questions_resets_when_the_oldest_ages_out(self, db):
        oldest = NOW - timedelta(hours=5)
        _log(db, "u2", at=oldest)
        for minutes in range(1, USER_LIMIT):
            _log(db, "u2", at=NOW - timedelta(minutes=minutes))
        usage = get_usage(db, "u2", is_guest=False, is_admin=False, now=NOW)
        assert usage.remaining == 0
        assert usage.reset_at == oldest + USER_WINDOW

    def test_over_the_limit_resets_when_enough_age_out(self, db):
        """8 questions (two raced past the check): the 2nd oldest decides."""
        times = [NOW - timedelta(hours=h) for h in (10, 9, 1, 1, 1, 1, 1, 1)]
        for at in times:
            _log(db, "u3", at=at)
        usage = get_usage(db, "u3", is_guest=False, is_admin=False, now=NOW)
        assert usage.used == 8
        assert usage.reset_at == NOW - timedelta(hours=9) + USER_WINDOW

    def test_questions_older_than_the_window_and_admin_rows_dont_count(self, db):
        _log(db, "u4", at=NOW - USER_WINDOW - timedelta(minutes=1))
        _log(db, "u4", at=NOW, is_admin=True)
        _log(db, "u4", at=NOW)
        assert get_usage(db, "u4", is_guest=False, is_admin=False, now=NOW).used == 1

    def test_admin_has_no_limit(self, db):
        usage = get_usage(db, "a1", is_guest=False, is_admin=True)
        assert (usage.is_admin, usage.limit, usage.remaining, usage.reset_at) == (True, None, None, None)


# =====================================================================
# GET /api/v1/usage and the query response's usage / 429 body
# =====================================================================

class TestUsageEndpoint:
    def test_requires_api_key(self, client):
        assert client.get("/api/v1/usage", headers=_headers(api_key="wrong")).status_code == 401

    def test_guest_usage(self, client):
        body = client.get("/api/v1/usage", headers=_headers(ip="9.9.9.1")).json()
        assert body["usage"]["is_guest"] is True
        assert body["usage"]["remaining"] == 1

    def test_signed_in_usage_counts_down_with_each_answer(self, client):
        assert _ask(client, user_id="u10").json()["usage"]["remaining"] == USER_LIMIT - 1
        assert _ask(client, user_id="u10").json()["usage"]["remaining"] == USER_LIMIT - 2
        body = client.get("/api/v1/usage", headers=_headers(user_id="u10")).json()
        assert body["usage"]["used"] == 2

    def test_admin_usage(self, client):
        body = client.get("/api/v1/usage", headers=_headers(user_id="a10", role="admin")).json()
        assert body["usage"]["is_admin"] is True
        assert body["usage"]["limit"] is None

    def test_guest_cannot_claim_admin(self, client):
        body = client.get("/api/v1/usage", headers=_headers(ip="9.9.9.2", role="admin")).json()
        assert body["usage"]["is_admin"] is False

    def test_no_database_means_no_usage(self, monkeypatch):
        monkeypatch.setenv("APP_API_KEY", VALID_KEY)
        with TestClient(app) as c:  # no get_db override; conftest leaves DATABASE_URL empty
            assert c.get("/api/v1/usage", headers=_headers(user_id="u")).json() == {"usage": None}


class TestLimitResponse:
    def test_user_limit_includes_reset_time(self, client, db):
        for _ in range(USER_LIMIT):
            assert _ask(client, user_id="u20").status_code == 200
        oldest = db.query(UserQueryLog).filter_by(user_identifier="u20").order_by(UserQueryLog.created_at).first()

        blocked = _ask(client, user_id="u20")
        assert blocked.status_code == 429
        body = blocked.json()
        assert body["detail"] == "USER_LIMIT_REACHED"
        reset_at = datetime.fromisoformat(body["reset_at"])
        expected = oldest.created_at.replace(tzinfo=oldest.created_at.tzinfo or timezone.utc) + USER_WINDOW
        assert reset_at == expected
        assert body["usage"]["remaining"] == 0

    def test_guest_limit_has_no_reset_time(self, client):
        _ask(client, ip="9.9.9.3")
        body = _ask(client, ip="9.9.9.3").json()
        assert body["detail"] == "GUEST_LIMIT_REACHED"
        assert body["reset_at"] is None

    def test_admin_is_never_blocked(self, client):
        for _ in range(USER_LIMIT + 3):
            r = _ask(client, user_id="a20", role="admin")
            assert r.status_code == 200
            assert r.json()["usage"]["is_admin"] is True


# =====================================================================
# Chat history
# =====================================================================

class TestHistory:
    def test_signed_in_question_starts_a_saved_conversation(self, client):
        r = _ask(client, "What is faith?", user_id="u30")
        conversation_id = r.json()["conversation_id"]
        assert conversation_id

        listed = client.get("/api/v1/conversations", headers=_headers(user_id="u30")).json()["conversations"]
        assert [c["id"] for c in listed] == [conversation_id]
        assert listed[0]["title"] == "What is faith?"

        detail = client.get(f"/api/v1/conversations/{conversation_id}", headers=_headers(user_id="u30")).json()
        assert [(m["role"], m["content"]) for m in detail["messages"]] == [
            ("user", "What is faith?"),
            ("assistant", "Answer to: What is faith?"),
        ]
        assert detail["messages"][1]["sources"] == ["topical_guide/faith"]
        assert detail["messages"][1]["source_links"][0]["label"] == "Topical Guide: Faith"

    def test_follow_up_appends_to_the_same_conversation(self, client):
        first = _ask(client, "What is faith?", user_id="u31").json()
        history = [
            {"role": "user", "content": "What is faith?"},
            {"role": "assistant", "content": first["answer"]},
        ]
        second = _ask(client, "How do I grow it?", user_id="u31",
                      conversation_id=first["conversation_id"], history=history).json()
        assert second["conversation_id"] == first["conversation_id"]

        detail = client.get(f"/api/v1/conversations/{first['conversation_id']}",
                            headers=_headers(user_id="u31")).json()
        assert [m["content"] for m in detail["messages"] if m["role"] == "user"] == [
            "What is faith?", "How do I grow it?"
        ]
        listed = client.get("/api/v1/conversations", headers=_headers(user_id="u31")).json()["conversations"]
        assert len(listed) == 1

    def test_new_chat_keeps_the_old_one(self, client):
        """Asking without a conversation_id (New chat) starts a second chat; both are listed, newest first."""
        old = _ask(client, "What is faith?", user_id="u32").json()["conversation_id"]
        new = _ask(client, "What is repentance?", user_id="u32").json()["conversation_id"]
        assert old != new
        listed = client.get("/api/v1/conversations", headers=_headers(user_id="u32")).json()["conversations"]
        assert [c["id"] for c in listed] == [new, old]

    def test_guest_chats_are_not_saved(self, client, db):
        r = _ask(client, ip="9.9.9.4")
        assert r.json()["conversation_id"] is None
        assert db.query(ChatConversation).count() == 0
        assert client.get("/api/v1/conversations", headers=_headers(ip="9.9.9.4")).json() == {"conversations": []}

    def test_cannot_append_to_someone_elses_conversation(self, client, db):
        theirs = _ask(client, "Mine", user_id="owner").json()["conversation_id"]
        r = _ask(client, "Intruding", user_id="intruder", conversation_id=theirs).json()
        assert r["conversation_id"] != theirs
        owner_messages = db.query(ChatMessage).filter_by(conversation_id=theirs).all()
        assert [m.content for m in owner_messages if m.role == "user"] == ["Mine"]

    def test_cannot_read_someone_elses_conversation(self, client):
        theirs = _ask(client, "Mine", user_id="owner2").json()["conversation_id"]
        assert client.get(f"/api/v1/conversations/{theirs}", headers=_headers(user_id="other")).status_code == 404
        assert client.get(f"/api/v1/conversations/{theirs}", headers=_headers(ip="9.9.9.5")).status_code == 404
        listed = client.get("/api/v1/conversations", headers=_headers(user_id="other")).json()
        assert listed == {"conversations": []}

    def test_unknown_conversation_id_starts_a_new_one(self, client):
        r = _ask(client, user_id="u33", conversation_id="does-not-exist").json()
        assert r["conversation_id"] and r["conversation_id"] != "does-not-exist"

    def test_admin_chats_are_saved_too(self, client):
        assert _ask(client, user_id="a30", role="admin").json()["conversation_id"]

    def test_history_endpoints_require_api_key(self, client):
        assert client.get("/api/v1/conversations", headers=_headers(user_id="u", api_key="x")).status_code == 401
        assert client.get("/api/v1/conversations/abc", headers=_headers(user_id="u", api_key="x")).status_code == 401

    def test_blocked_question_is_not_saved(self, client, db):
        for _ in range(USER_LIMIT):
            _ask(client, user_id="u34")
        _ask(client, "One too many", user_id="u34")
        assert db.query(ChatMessage).filter_by(content="One too many").count() == 0


class TestDeleteConversation:
    def test_owner_can_delete(self, client, db):
        keep = _ask(client, "Keep me", user_id="u40").json()["conversation_id"]
        gone = _ask(client, "Delete me", user_id="u40").json()["conversation_id"]

        r = client.delete(f"/api/v1/conversations/{gone}", headers=_headers(user_id="u40"))
        assert r.status_code == 204
        listed = client.get("/api/v1/conversations", headers=_headers(user_id="u40")).json()["conversations"]
        assert [c["id"] for c in listed] == [keep]
        assert db.query(ChatMessage).filter_by(conversation_id=gone).count() == 0
        assert client.get(f"/api/v1/conversations/{gone}", headers=_headers(user_id="u40")).status_code == 404

    def test_deleting_does_not_give_questions_back(self, client):
        conversation_id = _ask(client, user_id="u41").json()["conversation_id"]
        client.delete(f"/api/v1/conversations/{conversation_id}", headers=_headers(user_id="u41"))
        usage = client.get("/api/v1/usage", headers=_headers(user_id="u41")).json()["usage"]
        assert usage["used"] == 1

    def test_cannot_delete_someone_elses(self, client):
        theirs = _ask(client, user_id="owner3").json()["conversation_id"]
        assert client.delete(f"/api/v1/conversations/{theirs}", headers=_headers(user_id="other")).status_code == 404
        assert client.delete(f"/api/v1/conversations/{theirs}", headers=_headers(ip="9.9.9.6")).status_code == 404
        assert client.get(f"/api/v1/conversations/{theirs}", headers=_headers(user_id="owner3")).status_code == 200

    def test_unknown_id_is_404(self, client):
        assert client.delete("/api/v1/conversations/nope", headers=_headers(user_id="u42")).status_code == 404

    def test_requires_api_key(self, client):
        assert client.delete("/api/v1/conversations/x", headers=_headers(user_id="u", api_key="bad")).status_code == 401


@pytest.mark.parametrize(
    ("question", "title"),
    [
        ("What is faith?", "What is faith?"),
        ("  What   is\nfaith? ", "What is faith?"),
    ],
)
def test_make_title_collapses_whitespace(question, title):
    assert make_title(question) == title


def test_make_title_truncates_long_questions():
    title = make_title("word " * 50)
    assert len(title) == TITLE_MAX_CHARS
    assert title.endswith("…")
