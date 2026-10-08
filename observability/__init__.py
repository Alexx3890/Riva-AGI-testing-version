"""
Observability Package for Riva-AGI.
Provides structured JSON logging, context-bound request IDs,
sensitive data masking, and operational metrics (Prometheus & JSON).
"""

from .context import (
    get_request_id,
    set_request_id,
    get_session_id,
    set_session_id,
    clear_context,
)
from .sanitizer import mask_sensitive_data, mask_sensitive_text
from .formatter import JSONFormatter, StructuredTextFormatter
from .logging import setup_logging, get_logger
from .metrics import MetricsCollector, metrics
from .middleware import RequestIDAndLoggingMiddleware

__all__ = [
    "get_request_id",
    "set_request_id",
    "get_session_id",
    "set_session_id",
    "clear_context",
    "mask_sensitive_data",
    "mask_sensitive_text",
    "JSONFormatter",
    "StructuredTextFormatter",
    "setup_logging",
    "get_logger",
    "MetricsCollector",
    "metrics",
    "RequestIDAndLoggingMiddleware",
]
