"""Tests for the database layer: URL handling, startup without a database,
fail-open / fail-closed rate limiting, and (opt-in) a real Postgres run.

The SQLite-backed rate-limit tests live in test_rate_limiting.py. The
Postgres test at the bottom only runs when TEST_DATABASE_URL is set, e.g.

    TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres \
        uv run pytest -m postgres -v --no-cov
"""

import logging
import os
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from src.backend.database import (
    UserQueryLog,
    engine_kwargs,
    get_session_factory,
    init_db,
    normalize_database_url,
)
from src.backend.main import RATE_LIMIT_UNAVAILABLE_DETAIL, app, get_db

VALID_KEY = "test-key"
QUESTION = {"messages": [{"role": "user", "content": "What is faith?"}]}


@pytest.fixture(autouse=True)
def fresh_session_factory():
    """get_session_factory() is cached; reset it so each test sees its own env."""
    get_session_factory.cache_clear()
    yield
    get_session_factory.cache_clear()


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("APP_API_KEY", VALID_KEY)
    with TestClient(app) as c:
        yield c


def _ask(client, *, forwarded_for: str = "", user_id: str = ""):
    headers = {"X-API-Key": VALID_KEY}
    if forwarded_for:
        headers["X-Forwarded-For"] = forwarded_for
    if user_id:
        headers["X-User-Id"] = user_id
    return client.post("/api/v1/query", json=QUESTION, headers=headers)


# =====================================================================
# 1. URL normalisation and engine options
# =====================================================================

@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgres://u:p@host:5432/db", "postgresql+psycopg://u:p@host:5432/db"),
        ("postgresql://u:p@host:5432/db", "postgresql+psycopg://u:p@host:5432/db"),
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("sqlite://", "sqlite://"),
        ("sqlite:///./local.db", "sqlite:///./local.db"),
    ],
)
def test_normalize_database_url(url, expected):
    assert normalize_database_url(url) == expected


def test_engine_kwargs_direct_connection_keeps_prepared_statements():
    kwargs = engine_kwargs("postgresql+psycopg://u:p@db.example.supabase.co:5432/postgres")
    assert kwargs["pool_pre_ping"] is True
    assert kwargs["pool_size"] > 0
    assert kwargs["connect_args"] == {"connect_timeout": 5}


def test_engine_kwargs_transaction_pooler_disables_prepared_statements():
    kwargs = engine_kwargs("postgresql+psycopg://u:p@pooler.supabase.com:6543/postgres")
    assert kwargs["connect_args"] == {"connect_timeout": 5, "prepare_threshold": None}


def test_engine_kwargs_sqlite_gets_no_pool_options():
    assert engine_kwargs("sqlite://") == {}


def test_postgres_url_builds_psycopg_engine(monkeypatch):
    """The driver is importable: building the engine (no connection) works."""
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@localhost:6543/db")
    engine = get_session_factory().kw["bind"]
    assert engine.dialect.driver == "psycopg"


# =====================================================================
# 2. init_db()
# =====================================================================

def test_init_db_creates_table_and_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'logs.db'}")
    first = init_db()
    second = init_db()
    engine = first.kw["bind"]
    try:
        assert first is second
        inspector = inspect(engine)
        assert "user_query_logs" in inspector.get_table_names()
        index_names = {ix["name"] for ix in inspector.get_indexes("user_query_logs")}
        assert "ix_user_query_logs_rate_limit" in index_names
    finally:
        engine.dispose()


def test_init_db_without_database_url_returns_none(monkeypatch, caplog):
    monkeypatch.setenv("DATABASE_URL", "")
    with caplog.at_level(logging.WARNING):
        assert init_db() is None
    assert "DATABASE_URL is not set" in caplog.text


def test_init_db_with_invalid_url_returns_none(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "not a database url")
    assert init_db() is None


# =====================================================================
# 3. App starts and answers with no DATABASE_URL
# =====================================================================

def test_app_starts_and_answers_without_database(monkeypatch, client):
    monkeypatch.setenv("DATABASE_URL", "")
    assert app.state.db_session_factory is None

    # Rate limiting is off, so a guest isn't capped at one question.
    assert _ask(client, forwarded_for="10.9.0.1").status_code == 200
    assert _ask(client, forwarded_for="10.9.0.1").status_code == 200


# =====================================================================
# 4. Fail-open / fail-closed when the rate-limit query errors
# =====================================================================

@pytest.fixture()
def broken_db():
    """A session whose every query raises, as if the database went away."""
    session = MagicMock(spec=Session)
    session.query.side_effect = OperationalError("SELECT 1", {}, Exception("connection refused"))
    app.dependency_overrides[get_db] = lambda: session
    yield session
    app.dependency_overrides.pop(get_db, None)


@pytest.mark.parametrize("identity", [{"forwarded_for": "10.9.0.2"}, {"user_id": "user_db_down"}])
def test_db_error_fails_open_by_default(monkeypatch, client, broken_db, identity):
    monkeypatch.delenv("RATE_LIMIT_FAIL_CLOSED", raising=False)
    response = _ask(client, **identity)
    assert response.status_code == 200
    broken_db.rollback.assert_called()


def test_db_error_fails_closed_when_configured(monkeypatch, client, broken_db):
    monkeypatch.setenv("RATE_LIMIT_FAIL_CLOSED", "true")
    response = _ask(client, forwarded_for="10.9.0.3")
    assert response.status_code == 503
    assert response.json()["detail"] == RATE_LIMIT_UNAVAILABLE_DETAIL
    broken_db.rollback.assert_called_once()


def test_failed_rollback_still_fails_open(monkeypatch, client, broken_db):
    monkeypatch.delenv("RATE_LIMIT_FAIL_CLOSED", raising=False)
    broken_db.rollback.side_effect = OperationalError("ROLLBACK", {}, Exception("gone"))
    assert _ask(client, forwarded_for="10.9.0.4").status_code == 200


# =====================================================================
# 5. Real Postgres (opt-in)
# =====================================================================

@pytest.mark.postgres
@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="set TEST_DATABASE_URL to run against a real Postgres",
)
def test_postgres_init_db_and_guest_rate_limit(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setenv("APP_API_KEY", VALID_KEY)
    session_factory = init_db()
    assert session_factory is not None
    engine = session_factory.kw["bind"]
    assert "user_query_logs" in inspect(engine).get_table_names()

    guest_ip = f"pytest-{uuid.uuid4().hex}"  # unique, so reruns don't collide
    try:
        # The lifespan calls init_db() again and gets the same cached factory.
        with TestClient(app) as client:
            assert _ask(client, forwarded_for=guest_ip).status_code == 200
            second = _ask(client, forwarded_for=guest_ip)
            assert second.status_code == 429
            assert second.json()["detail"] == "GUEST_LIMIT_REACHED"
    finally:
        with session_factory() as db:
            db.query(UserQueryLog).filter(UserQueryLog.user_identifier == guest_ip).delete()
            db.commit()
        engine.dispose()
