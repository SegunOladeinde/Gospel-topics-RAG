"""Unit tests for database-backed rate limiting.

Tests the UserQueryLog model, the guest rate limit (1 request total),
and the authenticated user rate limit (7 requests per 24 hours).

All tests use an in-memory SQLite database injected via FastAPI dependency
override, so no Supabase credentials are needed in CI.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.backend.database import Base, UserQueryLog
from src.backend.main import app, get_db

# ---------------------------------------------------------------------------
# In-memory SQLite engine shared across all tests in this module
# ---------------------------------------------------------------------------
TEST_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)

VALID_KEY = "test-key"


def _auth():
    return {"X-API-Key": VALID_KEY}


def _make_ai_msg(text: str = "Mock answer."):
    from langchain_core.messages import AIMessage
    return AIMessage(content=text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_tables():
    """Create all tables before each test and drop them afterward.

    This gives every test a completely clean database so they are
    independent of each other.
    """
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture()
def db_session():
    """Provide a test database session for direct manipulation."""
    session = TestSession()
    try:
        yield session
    finally:
        session.close()


def _override_get_db():
    """FastAPI dependency override: yields a session bound to the test DB."""
    session = TestSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(monkeypatch):
    """Return a TestClient with the DB dependency overridden and API key set."""
    monkeypatch.setenv("APP_API_KEY", VALID_KEY)
    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


def _patch_rag():
    """Context manager that patches retriever + chat model + classify_intent
    so the endpoint never calls OpenAI or touches real indexes."""
    import src.backend.main as backend

    return (
        patch.object(backend, "classify_intent", new=AsyncMock(return_value="RAG_NEW_TOPIC")),
        patch.object(app.state, "retriever", create=True),
        patch.object(app.state, "chat_model", create=True),
    )


def _send_query(client, *, user_id: str = "", forwarded_for: str = ""):
    """Helper that sends a single POST to /api/v1/query with the given
    identity headers and fully mocked RAG pipeline."""
    import src.backend.main as backend

    headers = _auth()
    if user_id:
        headers["X-User-Id"] = user_id
    if forwarded_for:
        headers["X-Forwarded-For"] = forwarded_for

    payload = {"messages": [{"role": "user", "content": "What is faith?"}]}

    with patch.object(backend, "classify_intent", new=AsyncMock(return_value="RAG_NEW_TOPIC")), \
         patch.object(app.state, "retriever") as mock_ret, \
         patch.object(app.state, "chat_model") as mock_chat:
        mock_ret.ainvoke = AsyncMock(return_value=[])
        mock_chat.ainvoke = AsyncMock(return_value=_make_ai_msg())
        response = client.post("/api/v1/query", json=payload, headers=headers)

    return response


# =====================================================================
# 1. UserQueryLog Model Tests
# =====================================================================

class TestUserQueryLogModel:
    """Verify the SQLAlchemy model works correctly in the test DB."""

    def test_create_guest_log(self, db_session):
        """A guest log entry can be inserted and queried."""
        entry = UserQueryLog(user_identifier="192.168.1.1", is_guest=True)
        db_session.add(entry)
        db_session.commit()

        result = db_session.query(UserQueryLog).filter_by(
            user_identifier="192.168.1.1"
        ).first()
        assert result is not None
        assert result.is_guest is True
        assert result.user_identifier == "192.168.1.1"
        assert result.id is not None  # auto-generated

    def test_create_authenticated_log(self, db_session):
        """An authenticated user log entry can be inserted and queried."""
        entry = UserQueryLog(user_identifier="clerk_user_abc123", is_guest=False)
        db_session.add(entry)
        db_session.commit()

        result = db_session.query(UserQueryLog).filter_by(
            user_identifier="clerk_user_abc123"
        ).first()
        assert result is not None
        assert result.is_guest is False

    def test_unique_ids(self, db_session):
        """Each entry gets a unique primary key."""
        e1 = UserQueryLog(user_identifier="ip1", is_guest=True)
        e2 = UserQueryLog(user_identifier="ip2", is_guest=True)
        db_session.add_all([e1, e2])
        db_session.commit()
        assert e1.id != e2.id


# =====================================================================
# 2. Guest Rate Limiting Tests (1 request cap)
# =====================================================================

class TestGuestRateLimit:
    """Guests (no X-User-Id header) are limited to exactly 1 request."""

    def test_guest_first_request_succeeds(self, client):
        """The very first guest request from an IP must succeed (200)."""
        response = _send_query(client, forwarded_for="10.0.0.1")
        assert response.status_code == 200

    def test_guest_second_request_blocked(self, client):
        """A second request from the same guest IP must return 429."""
        first = _send_query(client, forwarded_for="10.0.0.2")
        assert first.status_code == 200

        second = _send_query(client, forwarded_for="10.0.0.2")
        assert second.status_code == 429
        assert second.json()["detail"] == "GUEST_LIMIT_REACHED"

    def test_guest_different_ips_are_independent(self, client):
        """Different guest IPs each get their own single-request quota."""
        r1 = _send_query(client, forwarded_for="10.0.0.3")
        r2 = _send_query(client, forwarded_for="10.0.0.4")
        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_guest_third_request_still_blocked(self, client):
        """Subsequent requests beyond the second are also blocked."""
        _send_query(client, forwarded_for="10.0.0.5")
        _send_query(client, forwarded_for="10.0.0.5")
        third = _send_query(client, forwarded_for="10.0.0.5")
        assert third.status_code == 429
        assert third.json()["detail"] == "GUEST_LIMIT_REACHED"


# =====================================================================
# 3. Authenticated User Rate Limiting Tests (7 per 24h)
# =====================================================================

class TestUserRateLimit:
    """Clerk-authenticated users are limited to 7 requests per 24 hours."""

    def test_user_first_request_succeeds(self, client):
        """First authenticated request must succeed."""
        response = _send_query(client, user_id="user_001")
        assert response.status_code == 200

    def test_user_seven_requests_all_succeed(self, client):
        """Exactly 7 requests within 24 hours must all succeed."""
        for i in range(7):
            r = _send_query(client, user_id="user_002")
            assert r.status_code == 200, f"Request {i+1} failed unexpectedly"

    def test_user_eighth_request_blocked(self, client):
        """The 8th request within 24 hours must return 429."""
        for _ in range(7):
            r = _send_query(client, user_id="user_003")
            assert r.status_code == 200

        eighth = _send_query(client, user_id="user_003")
        assert eighth.status_code == 429
        assert eighth.json()["detail"] == "USER_LIMIT_REACHED"

    def test_different_users_have_independent_limits(self, client):
        """User A hitting the limit must not affect User B."""
        for _ in range(7):
            _send_query(client, user_id="user_A")

        blocked = _send_query(client, user_id="user_A")
        assert blocked.status_code == 429

        allowed = _send_query(client, user_id="user_B")
        assert allowed.status_code == 200

    def test_user_old_queries_do_not_count(self, client, db_session):
        """Queries older than 24 hours must not count toward the limit.

        We manually insert 7 old records (25 hours ago) into the test DB,
        then send a fresh request which should succeed because those old
        records fall outside the 24-hour window.
        """
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        for _ in range(7):
            entry = UserQueryLog(
                user_identifier="user_stale",
                is_guest=False,
            )
            db_session.add(entry)
        db_session.commit()

        # Manually backdate all entries
        db_session.query(UserQueryLog).filter_by(
            user_identifier="user_stale"
        ).update({"created_at": old_time})
        db_session.commit()

        response = _send_query(client, user_id="user_stale")
        assert response.status_code == 200


# =====================================================================
# 4. Query Log Insertion Tests
# =====================================================================

class TestQueryLogInsertion:
    """Verify that successful queries insert a log entry."""

    def test_guest_query_creates_log(self, client, db_session):
        """A successful guest query inserts a record in user_query_logs."""
        _send_query(client, forwarded_for="10.0.0.100")

        count = db_session.query(UserQueryLog).filter_by(
            user_identifier="10.0.0.100", is_guest=True
        ).count()
        assert count == 1

    def test_user_query_creates_log(self, client, db_session):
        """A successful authenticated query inserts a record."""
        _send_query(client, user_id="user_log_test")

        count = db_session.query(UserQueryLog).filter_by(
            user_identifier="user_log_test", is_guest=False
        ).count()
        assert count == 1

    def test_blocked_request_does_not_create_extra_log(self, client, db_session):
        """A 429-blocked request must NOT insert an additional log entry."""
        _send_query(client, forwarded_for="10.0.0.200")  # 1st succeeds, logged
        _send_query(client, forwarded_for="10.0.0.200")  # 2nd blocked

        count = db_session.query(UserQueryLog).filter_by(
            user_identifier="10.0.0.200"
        ).count()
        assert count == 1  # only the first successful request was logged
