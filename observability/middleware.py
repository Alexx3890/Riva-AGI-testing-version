"""
FastAPI / Starlette middleware for request tracing, structured logging,
and operational metrics collection.
"""

import time
from typing import Any, Callable

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response
except ImportError:
    # Graceful fallback for non-web environments (CLI, workers, offline tests)
    class DummyBaseHTTPMiddleware:
        def __init__(self, app: Any = None, *args: Any, **kwargs: Any) -> None:
            self.app = app

    BaseHTTPMiddleware = DummyBaseHTTPMiddleware
    Request = Any
    Response = Any

from .context import clear_context, get_request_id, set_request_id
from .logging import get_logger
from .metrics import metrics

logger = get_logger("riva.access")


class RequestIDAndLoggingMiddleware(BaseHTTPMiddleware):
    """
    HTTP Middleware that:
    1. Extracts incoming 'X-Request-ID' or generates a unique UUID.
    2. Injects the request ID into contextvars for downstream logging.
    3. Records request start/end timing and metrics.
    4. Attaches 'X-Request-ID' to the outgoing response headers.
    """

    async def dispatch(self, request: Any, call_next: Callable) -> Any:
        # Extract or generate Request ID
        incoming_rid = getattr(request, "headers", {}).get("X-Request-ID")
        rid = set_request_id(incoming_rid)

        start_time = time.time()
        path = getattr(getattr(request, "url", None), "path", "/")
        method = getattr(request, "method", "GET")

        status_code = 500
        try:
            response = await call_next(request)
            status_code = getattr(response, "status_code", 200)
            if hasattr(response, "headers"):
                response.headers["X-Request-ID"] = rid
            return response
        except Exception as exc:
            metrics.record_error(error_class=exc.__class__.__name__, route=path)
            logger.error(
                f"Unhandled error handling {method} {path}: {exc}",
                exc_info=True,
                extra={"request_id": rid, "http_method": method, "path": path},
            )
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000.0
            metrics.record_request(
                method=method,
                route=path,
                status_code=status_code,
                duration_ms=duration_ms,
            )

            # Do not spam logs for high-frequency internal health checks if desired,
            # or log at debug level; log all other endpoints at INFO level
            log_level = logger.debug if path == "/health" else logger.info
            log_level(
                f"{method} {path} completed with {status_code} in {duration_ms:.2f}ms",
                extra={
                    "request_id": rid,
                    "http_method": method,
                    "path": path,
                    "status_code": status_code,
                    "latency_ms": round(duration_ms, 2),
                },
            )
            clear_context()
