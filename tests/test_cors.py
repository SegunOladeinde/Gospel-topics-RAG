import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from src.backend.main import parse_allowed_origins

VERCEL = "https://gospel-topics-rag.vercel.app"


@pytest.mark.parametrize(
    "raw",
    [
        VERCEL,
        f"{VERCEL}/",
        f"  {VERCEL}/  ",
        f"{VERCEL},",
        f"{VERCEL}, {VERCEL}/",
    ],
)
def test_vercel_origin_variants_normalise(raw):
    assert parse_allowed_origins(raw) == [VERCEL]


def test_multiple_origins_keep_order():
    raw = f" {VERCEL}/ , http://localhost:3000 "
    assert parse_allowed_origins(raw) == [VERCEL, "http://localhost:3000"]


@pytest.mark.parametrize("raw", [None, "", " , "])
def test_unset_or_blank_falls_back_to_localhost(raw):
    expected = ["http://localhost:3000"] if raw is None or raw == "" else []
    assert parse_allowed_origins(raw) == expected


def test_parsed_origin_passes_a_real_preflight():
    """A trailing-slash value must still let the browser's Origin through."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=parse_allowed_origins(f" {VERCEL}/ "),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    client = TestClient(app)
    response = client.options(
        "/api/v1/query",
        headers={"Origin": VERCEL, "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == VERCEL

    blocked = client.options(
        "/api/v1/query",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in blocked.headers
