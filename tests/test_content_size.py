"""Tests for ContentSizeLimitMiddleware's Content-Length handling.

The middleware is called directly with a hand-built request, because HTTP
clients (including TestClient) overwrite Content-Length with the real body
size, so a malformed value can't be sent through them.
"""

import pytest
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from src.backend.main import ContentSizeLimitMiddleware


def _request(content_length: str | None) -> Request:
    headers = [] if content_length is None else [(b"content-length", content_length.encode())]
    return Request({"type": "http", "method": "POST", "path": "/api/v1/query", "headers": headers})


async def _call_next(request):
    return PlainTextResponse("passed through")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("value", ["abc", "-1", "1e9", "", "12 34"])
async def test_malformed_content_length_is_400(value):
    middleware = ContentSizeLimitMiddleware(app=None)
    response = await middleware.dispatch(_request(value), _call_next)
    assert response.status_code == 400


@pytest.mark.anyio
async def test_oversized_body_is_413():
    middleware = ContentSizeLimitMiddleware(app=None)
    response = await middleware.dispatch(_request(str(2 * 1024 * 1024)), _call_next)
    assert response.status_code == 413


@pytest.mark.anyio
@pytest.mark.parametrize("value", [None, "0", "512", str(1024 * 1024)])
async def test_acceptable_requests_pass_through(value):
    middleware = ContentSizeLimitMiddleware(app=None)
    response = await middleware.dispatch(_request(value), _call_next)
    assert response.status_code == 200
    assert response.body == b"passed through"
