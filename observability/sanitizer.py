"""
Sensitive data sanitization and masking for logs and metrics.
Prevents leaking API keys, tokens, authorization headers, and personal data.
"""

import re
from typing import Any, Dict, List, Set, Union

# Common sensitive parameter/header names (case-insensitive)
SENSITIVE_KEYS: Set[str] = {
    "api_key",
    "apikey",
    "gemini_api_key",
    "authorization",
    "auth",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "private_key",
    "mongodb_uri",
    "connection_string",
}

# Regex to detect API keys and Bearer tokens in raw text
PATTERNS = [
    # Google AI / Gemini API keys (usually AIzaSy...)
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "AIza...[REDACTED]"),
    # General bearer tokens
    (re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE), "Bearer [REDACTED]"),
    # MongoDB URIs with credentials: mongodb(+srv)://user:pass@host
    (re.compile(r"mongodb(?:\+srv)?://([^:]+):([^@]+)@", re.IGNORECASE), r"mongodb://\1:***@"),
]


def mask_sensitive_text(text: str) -> str:
    """Masks known key signatures in arbitrary text strings."""
    if not isinstance(text, str):
        return str(text)

    sanitized = text
    for pattern, replacement in PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def mask_sensitive_data(data: Any, max_depth: int = 5) -> Any:
    """
    Recursively scrubs sensitive keys from dictionaries, lists, and primitives.
    Replaces values for sensitive keys with [REDACTED].
    """
    if max_depth <= 0:
        return data

    if isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            key_lower = str(k).lower().strip()
            if any(s in key_lower for s in SENSITIVE_KEYS):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = mask_sensitive_data(v, max_depth - 1)
        return cleaned

    if isinstance(data, list):
        return [mask_sensitive_data(item, max_depth - 1) for item in data]

    if isinstance(data, str):
        return mask_sensitive_text(data)

    return data
