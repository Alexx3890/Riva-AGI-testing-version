"""
Logging configuration manager for Riva-AGI.
Supports environment-driven configuration (JSON for container/prod, text for dev).
"""

import logging
import os
import sys
from typing import Optional

from .formatter import JSONFormatter, StructuredTextFormatter


def setup_logging(
    service_name: Optional[str] = None,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Configures the root and Riva loggers.
    - service_name: Name of the microservice (default env SERVICE_NAME or 'riva-agi')
    - log_level: Logging level (default env LOG_LEVEL or 'INFO')
    - log_format: 'json' or 'text' (default env LOG_FORMAT or 'json')
    """
    service = service_name or os.getenv("SERVICE_NAME", "riva-agi")
    level_str = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    # In containers or when LOG_FORMAT=json, use JSONFormatter
    fmt_choice = (log_format or os.getenv("LOG_FORMAT", "json")).lower()
    env = os.getenv("ENVIRONMENT", os.getenv("ENV", "development"))
    version = os.getenv("APP_VERSION", "0.1.0")

    # Select formatter
    if fmt_choice == "json":
        formatter = JSONFormatter(service_name=service, environment=env, version=version)
    else:
        formatter = StructuredTextFormatter(service_name=service)

    # Configure root logger handler
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Silence overly verbose external libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    return logging.getLogger(service)


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor for named loggers."""
    return logging.getLogger(name)
