"""
Unit tests for the Observability package (structured JSON logging,
request IDs, contextvars, sensitive data masking, and metrics collection).
"""

import json
import logging
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

from observability.context import (
    clear_context,
    get_request_id,
    get_session_id,
    set_request_id,
    set_session_id,
)
from observability.formatter import JSONFormatter, StructuredTextFormatter
from observability.metrics import MetricsCollector
from observability.sanitizer import mask_sensitive_data, mask_sensitive_text


def test_context_request_and_session_ids():
    """Verify context variable isolation, setting, and cleanup."""
    clear_context()
    assert get_request_id() is None
    assert get_session_id() is None

    rid = set_request_id("custom_req_123")
    assert rid == "custom_req_123"
    assert get_request_id() == "custom_req_123"

    sid = set_session_id("custom_sess_456")
    assert sid == "custom_sess_456"
    assert get_session_id() == "custom_sess_456"

    clear_context()
    assert get_request_id() is None
    assert get_session_id() is None


def test_sanitizer_masks_credentials_and_tokens():
    """Verify that API keys, passwords, and tokens are scrubbed."""
    raw_text = "Calling API with key AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q and Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    masked = mask_sensitive_text(raw_text)
    assert "AIzaSy" not in masked
    assert "[REDACTED]" in masked

    data_payload = {
        "api_key": "secret123",
        "gemini_api_key": "AIzaSySecret",
        "user_query": "What is KIET?",
        "metadata": {
            "token": "tok_xyz",
            "safe_counter": 42,
        },
    }
    sanitized = mask_sensitive_data(data_payload)
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["gemini_api_key"] == "[REDACTED]"
    assert sanitized["user_query"] == "What is KIET?"
    assert sanitized["metadata"]["token"] == "[REDACTED]"
    assert sanitized["metadata"]["safe_counter"] == 42


def test_json_formatter_structure():
    """Verify that JSONFormatter produces valid structured JSON with required fields."""
    formatter = JSONFormatter(service_name="riva-test", environment="test", version="1.0.0")

    set_request_id("req_test_999")
    set_session_id("sess_test_888")

    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Processing test item with key AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q",
        args=(),
        exc_info=None,
    )
    record.custom_metric = 123.45

    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["level"] == "INFO"
    assert parsed["service"] == "riva-test"
    assert parsed["environment"] == "test"
    assert parsed["version"] == "1.0.0"
    assert parsed["request_id"] == "req_test_999"
    assert parsed["session_id"] == "sess_test_888"
    assert "AIzaSy" not in parsed["message"]
    assert parsed["extra"]["custom_metric"] == 123.45

    clear_context()


def test_structured_text_formatter():
    """Verify development text formatter formats cleanly."""
    formatter = StructuredTextFormatter(service_name="riva-dev")
    set_request_id("req_abc")
    record = logging.LogRecord(
        name="dev.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=5,
        msg="Dev message",
        args=(),
        exc_info=None,
    )
    out = formatter.format(record)
    assert "[req_abc]" in out
    assert "riva-dev" in out
    assert "Dev message" in out
    clear_context()


def test_metrics_collector():
    """Verify request, session, and operational metric recording."""
    collector = MetricsCollector(service_name="riva-test-metrics")

    # Record HTTP
    collector.record_request("GET", "/health", 200, 5.2)
    collector.record_request("GET", "/health", 200, 4.8)
    collector.record_request("POST", "/api/chat", 500, 120.0)

    # Record WebSocket
    collector.record_ws_session_start(voice="Aoede", language="en")
    assert collector.active_websocket_sessions == 1
    collector.record_ws_session_end(duration_s=30.5)
    assert collector.active_websocket_sessions == 0

    # Record Gemini call & error
    collector.record_gemini_call(model="gemini-flash", operation="synthesize", success=True, duration_ms=450.0)
    collector.record_error(error_class="QuotaExceededError", route="/api/chat")

    # Verify JSON output
    data = collector.get_metrics_json()
    assert data["service"] == "riva-test-metrics"
    assert len(data["http_requests"]) == 2
    assert data["websocket"]["total_sessions"] == 1
    assert data["websocket"]["avg_session_duration_s"] == 30.5
    assert len(data["gemini"]) == 1
    assert len(data["errors"]) == 1

    # Verify Prometheus output
    prom_text = collector.get_prometheus_metrics()
    assert "riva_http_requests_total" in prom_text
    assert 'route="/health"' in prom_text
    assert "riva_active_websocket_sessions" in prom_text
    assert "riva_gemini_api_calls_total" in prom_text
    assert "riva_errors_total" in prom_text


if __name__ == "__main__":
    test_context_request_and_session_ids()
    test_sanitizer_masks_credentials_and_tokens()
    test_json_formatter_structure()
    test_structured_text_formatter()
    test_metrics_collector()
    print("All unit tests passed successfully!")
