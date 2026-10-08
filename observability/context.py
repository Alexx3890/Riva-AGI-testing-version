"""
Context management for request IDs and session tracing using contextvars.
Ensures thread-safe and async-safe propagation of IDs across coroutines.
"""

from contextvars import ContextVar
from typing import Optional
import uuid

_request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_session_id_ctx: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


def generate_request_id() -> str:
    """Generates a unique 12-char hex request ID or UUID4 hex."""
    return uuid.uuid4().hex[:16]


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Sets the current request ID in context.
    If none provided, generates a new unique ID.
    Returns the set request ID.
    """
    rid = request_id or generate_request_id()
    _request_id_ctx.set(rid)
    return rid


def get_request_id() -> Optional[str]:
    """Returns the current request ID from context or None."""
    return _request_id_ctx.get()


def set_session_id(session_id: Optional[str] = None) -> str:
    """
    Sets the current session ID (e.g. for WebSockets) in context.
    If none provided, generates a new unique ID.
    Returns the set session ID.
    """
    sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
    _session_id_ctx.set(sid)
    return sid


def get_session_id() -> Optional[str]:
    """Returns the current session ID from context or None."""
    return _session_id_ctx.get()


def clear_context() -> None:
    """Resets the context variables."""
    _request_id_ctx.set(None)
    _session_id_ctx.set(None)
