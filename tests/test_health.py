import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.backend.main as backend
from src.backend.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_root_describes_the_service(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"service": "gospel-topics-rag", "docs": "/docs", "health": "/health"}


def test_health_without_database(client, monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", "0123456789abcdef0123456789abcdef01234567")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "commit": "0123456",
        "indexes_loaded": True,
        "db": "disabled",
        # The test stub retriever has no real chunk count.
        "retriever": {"backend": "faiss", "chunks": None},
    }


def test_health_commit_unknown_outside_render(client, monkeypatch):
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    assert client.get("/health").json()["commit"] == "unknown"


def test_health_db_ok(client, monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    monkeypatch.setattr(app.state, "db_session_factory", sessionmaker(bind=engine))
    body = client.get("/health").json()
    assert body["db"] == "ok"
    assert body["status"] == "ok"


def test_health_db_error_is_degraded_but_200(client, monkeypatch):
    session = MagicMock()
    session.__enter__.return_value.execute.side_effect = OperationalError(
        "SELECT 1", {}, Exception("password authentication failed for user secret")
    )
    monkeypatch.setattr(app.state, "db_session_factory", lambda: session)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["db"] == "error"
    assert response.json()["status"] == "degraded"
    # Error details stay in the server log, never in the response.
    assert "secret" not in response.text


def test_health_db_timeout(client, monkeypatch):
    monkeypatch.setattr(backend, "HEALTH_DB_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(app.state, "db_session_factory", object())
    monkeypatch.setattr(backend, "ping_db", lambda _factory: time.sleep(0.5) or "ok")
    assert client.get("/health").json()["db"] == "error"


def test_health_503_before_indexes_load(client, monkeypatch):
    monkeypatch.setattr(app.state, "retriever", None)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["indexes_loaded"] is False


def test_health_never_calls_openai(client, monkeypatch):
    chat_model = MagicMock()
    retriever = MagicMock()
    monkeypatch.setattr(app.state, "chat_model", chat_model)
    monkeypatch.setattr(app.state, "retriever", retriever)
    client.get("/health")
    chat_model.ainvoke.assert_not_called()
    retriever.ainvoke.assert_not_called()
