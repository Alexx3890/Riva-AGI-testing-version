"""
Tests for voice gateway observability integration (middleware,
request ID injection, and metrics tracking).
"""

import asyncio
from pathlib import Path
import sys

# Ensure project root is in sys.path
_pkg_root = str(Path(__file__).resolve().parent.parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

try:
    import pytest
except ImportError:
    pytest = None

from observability.context import clear_context, get_request_id
from observability.middleware import RequestIDAndLoggingMiddleware


class DummyResponse:
    def __init__(self, content: str = "OK", status_code: int = 200):
        self.content = content
        self.status_code = status_code
        self.headers = {}


class DummyRequest:
    def __init__(self, path: str = "/health", method: str = "GET", headers: dict = None):
        self.url = type("URL", (), {"path": path})()
        self.method = method
        self.headers = headers or {}


async def async_test_request_id_middleware_injects_header():
    """Verify middleware injects X-Request-ID and cleans up context."""
    middleware = RequestIDAndLoggingMiddleware(app=None)
    request = DummyRequest(path="/health", method="GET")

    captured_rid = None

    async def call_next(req):
        nonlocal captured_rid
        captured_rid = get_request_id()
        return DummyResponse()

    response = await middleware.dispatch(request, call_next)

    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] == captured_rid
    assert captured_rid is not None
    assert get_request_id() is None


async def async_test_request_id_middleware_preserves_incoming_header():
    """Verify middleware respects an incoming X-Request-ID."""
    middleware = RequestIDAndLoggingMiddleware(app=None)
    request = DummyRequest(
        path="/api/test",
        method="POST",
        headers={"X-Request-ID": "client-trace-12345"},
    )

    async def call_next(req):
        assert get_request_id() == "client-trace-12345"
        return DummyResponse()

    response = await middleware.dispatch(request, call_next)
    assert response.headers["X-Request-ID"] == "client-trace-12345"


# Pytest wrappers if pytest is present
if pytest:
    @pytest.mark.asyncio
    async def test_request_id_middleware_injects_header():
        await async_test_request_id_middleware_injects_header()

    @pytest.mark.asyncio
    async def test_request_id_middleware_preserves_incoming_header():
        await async_test_request_id_middleware_preserves_incoming_header()


if __name__ == "__main__":
    asyncio.run(async_test_request_id_middleware_injects_header())
    asyncio.run(async_test_request_id_middleware_preserves_incoming_header())
    print("Voice speech observability tests passed successfully!")
