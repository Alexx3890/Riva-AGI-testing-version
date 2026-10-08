"""
Log formatters for structured JSON logging and development text format.
"""

from datetime import datetime, timezone
import json
import logging
import traceback
from typing import Any, Dict, Optional

from .context import get_request_id, get_session_id
from .sanitizer import mask_sensitive_data, mask_sensitive_text


class JSONFormatter(logging.Formatter):
    """
    Standard-library compatible JSON log formatter.
    Outputs one structured JSON record per line.
    """

    def __init__(
        self,
        service_name: str = "riva-agi",
        environment: Optional[str] = None,
        version: Optional[str] = None,
    ):
        super().__init__()
        self.service_name = service_name
        self.environment = environment or "development"
        self.version = version or "0.1.0"

    def format(self, record: logging.LogRecord) -> str:
        # Generate ISO-8601 UTC timestamp with trailing Z
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        iso_timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Base log fields
        log_entry: Dict[str, Any] = {
            "timestamp": iso_timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": mask_sensitive_text(record.getMessage()),
            "service": self.service_name,
            "environment": self.environment,
            "version": self.version,
        }

        # Tracing context (Request ID / Session ID)
        req_id = getattr(record, "request_id", None) or get_request_id()
        if req_id:
            log_entry["request_id"] = req_id

        sess_id = getattr(record, "session_id", None) or get_session_id()
        if sess_id:
            log_entry["session_id"] = sess_id

        # Source code location
        log_entry["location"] = f"{record.filename}:{record.lineno}"

        # Capture extra attributes attached to the LogRecord
        reserved_keys = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "request_id", "session_id"
        }
        extra_fields = {
            k: v for k, v in record.__dict__.items() if k not in reserved_keys
        }
        if extra_fields:
            log_entry["extra"] = mask_sensitive_data(extra_fields)

        # Exception / Traceback handling
        if record.exc_info:
            exc_type, exc_val, exc_tb = record.exc_info
            log_entry["error"] = {
                "type": getattr(exc_type, "__name__", str(exc_type)),
                "message": str(exc_val),
                "stack_trace": traceback.format_exception(exc_type, exc_val, exc_tb),
            }
        elif record.exc_text:
            log_entry["error"] = {
                "message": record.exc_text
            }

        return json.dumps(log_entry, default=str)


class StructuredTextFormatter(logging.Formatter):
    """
    Colored/formatted text output for interactive local developer shells.
    """

    def __init__(self, service_name: str = "riva-agi"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        req_id = getattr(record, "request_id", None) or get_request_id() or "-"
        sess_id = getattr(record, "session_id", None) or get_session_id()
        context_str = f"[{req_id}]" if not sess_id else f"[{req_id}|{sess_id}]"

        msg = mask_sensitive_text(record.getMessage())
        base = f"{dt} [{record.levelname:5}] {self.service_name} {context_str} {record.name}: {msg}"

        if record.exc_info:
            base += "\n" + "".join(traceback.format_exception(*record.exc_info))
        return base
