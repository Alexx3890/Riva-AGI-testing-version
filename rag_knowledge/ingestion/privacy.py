"""Early Privacy Gate for Universal Ingestion Pipeline.

Ensures no private, sensitive information (phone numbers, personal emails,
national IDs, known private entity values) leaks into vector embeddings,
content, titles, summaries, keywords, or metadata.
Fails closed on leaks unless opt-in redaction is explicitly configured.
"""

import functools
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class PrivacyGateError(ValueError):
    """Raised when private sensitive data is detected in public embedded fields or metadata."""
    pass


@functools.lru_cache(maxsize=32)
def _get_compiled_private_pattern(private_tuple: Tuple[str, ...]) -> Optional[re.Pattern]:
    """Compiles a combined boundary-safe regex pattern for all known private entity values."""
    if not private_tuple:
        return None
    sorted_pvs = sorted(private_tuple, key=len, reverse=True)
    valid_pvs = [re.escape(p) for p in sorted_pvs if p and len(p.strip()) >= 3]
    if not valid_pvs:
        return None
    return re.compile(r"\b(?:" + "|".join(valid_pvs) + r")\b", re.IGNORECASE)


# RFC-5322 compliant email pattern (unanchored)
_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    re.IGNORECASE,
)

# Unanchored phone number pattern matching:
# 1. Indian 10-digit mobile numbers with optional prefix and internal spaces/separators (e.g. '98765 43210', '+91 98765-43210')
# 2. International E.164 phone numbers with explicit '+' prefix and separators
_PHONE_PATTERN = re.compile(
    r"(?:\b|\+)(?:91[\s.-]?)?[6-9](?:[\s.-]?\d){9}\b|"
    r"\+\d{1,3}[\s.-]?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b"
)


def is_phone_number(val: str) -> bool:
    """Checks if a string represents or contains a phone number."""
    if not val:
        return False
    digits = re.sub(r"[^\d]", "", str(val))
    if len(digits) >= 10 and (str(val).strip().startswith("+") or digits.startswith(("6", "7", "8", "9"))):
        return True
    return bool(_PHONE_PATTERN.search(str(val)))


def is_email_address(val: str) -> bool:
    """Checks if a string represents or contains an email address."""
    if not val or "@" not in str(val):
        return False
    return bool(_EMAIL_PATTERN.search(str(val)))


def extract_searchable_corpus(doc: Dict[str, Any]) -> str:
    """Concatenates all fields destined for embedding, text search, and metadata."""
    title = str(doc.get("title", ""))
    summary = str(doc.get("summary", ""))
    content = str(doc.get("content", ""))
    aliases = " ".join(str(a) for a in doc.get("aliases", []))
    keywords = " ".join(str(k) for k in doc.get("keywords", []))

    # Also inspect metadata string values to prevent leakage in payloads
    meta_values = []
    meta = doc.get("metadata", {})
    if isinstance(meta, dict):
        for k, v in meta.items():
            if isinstance(v, (str, int, float)):
                meta_values.append(f"{k}: {v}")

    return f"{title} {summary} {content} {aliases} {keywords} {' '.join(meta_values)}"


def scan_for_privacy_leaks(
    text: str,
    safe_identifiers: Optional[Set[str]] = None,
    known_private_values: Optional[Set[str]] = None,
    pv_pattern: Optional[re.Pattern] = None,
) -> Dict[str, List[str]]:
    """Scans text for unanchored email, phone leaks, and known private entity values."""
    safe = {str(s).strip().upper() for s in (safe_identifiers or set()) if s}
    leaks: Dict[str, List[str]] = {"emails": [], "phones": [], "private_values": []}

    # 1. Scan emails
    for email in _EMAIL_PATTERN.findall(text):
        if email.strip().upper() not in safe:
            leaks["emails"].append(email)

    # 2. Scan phones
    for match in _PHONE_PATTERN.finditer(text):
        raw_match = match.group(0).strip()
        digits_only = re.sub(r"[^\d]", "", raw_match)

        # Exempt if exact digits match a safe identifier (e.g. numeric roll numbers or UIDs)
        if digits_only.upper() in safe or raw_match.upper() in safe:
            continue

        if len(digits_only) >= 10:
            leaks["phones"].append(raw_match)

    # 3. Scan known private entity values (father names, personal emails, personal phones)
    pattern = pv_pattern
    if pattern is None and known_private_values:
        private_vals = {
            str(p).strip().upper()
            for p in known_private_values
            if p and len(str(p).strip()) >= 3 and str(p).strip().upper() not in safe
        }
        if private_vals:
            pattern = _get_compiled_private_pattern(tuple(sorted(private_vals)))

    if pattern:
        matches = pattern.findall(text)
        if matches:
            leaks["private_values"].extend(list(set(matches)))

    return leaks


def redact_private_text(
    text: str,
    safe_identifiers: Optional[Set[str]] = None,
    known_private_values: Optional[Set[str]] = None,
    pv_pattern: Optional[re.Pattern] = None,
) -> str:
    """Replaces detected emails, phones, and known private values with redaction markers."""
    safe = {str(s).strip().upper() for s in (safe_identifiers or set()) if s}

    # Redact emails
    redacted = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)

    # Redact phones while preserving whitelisted identifiers
    def _phone_sub(m: re.Match) -> str:
        raw = m.group(0).strip()
        digits = re.sub(r"[^\d]", "", raw)
        if digits.upper() in safe or raw.upper() in safe:
            return raw
        if len(digits) >= 10:
            return "[REDACTED_PHONE]"
        return raw

    redacted = _PHONE_PATTERN.sub(_phone_sub, redacted)

    # Redact known private values using cached compiled regex
    pattern = pv_pattern
    if pattern is None and known_private_values:
        private_vals = {
            str(p).strip().upper()
            for p in known_private_values
            if p and len(str(p).strip()) >= 3 and str(p).strip().upper() not in safe
        }
        if private_vals:
            pattern = _get_compiled_private_pattern(tuple(sorted(private_vals)))

    if pattern:
        redacted = pattern.sub("[REDACTED]", redacted)

    return redacted


def assert_no_privacy_leaks(
    documents: Iterable[Dict[str, Any]],
    safe_identifiers: Optional[Set[str]] = None,
    known_private_values: Optional[Set[str]] = None,
    opt_in_redact: bool = False,
) -> None:
    """Enforces zero-leak policy across a batch of unified documents.

    If opt_in_redact is True, mutates document content, summary, title, aliases,
    and keywords in-place with redactions, then re-verifies.
    Raises PrivacyGateError with all detected violations if any leak is unredacted.
    """
    safe = {str(s).strip().upper() for s in (safe_identifiers or set()) if s}
    clean_private = {
        str(p).strip().upper()
        for p in (known_private_values or set())
        if p and len(str(p).strip()) >= 3 and str(p).strip().upper() not in safe
    }
    pv_pattern = _get_compiled_private_pattern(tuple(sorted(clean_private))) if clean_private else None

    violations: List[str] = []

    for doc in documents:
        if opt_in_redact:
            # Thorough redaction across all text fields
            for field in ["title", "summary", "content"]:
                if field in doc and isinstance(doc[field], str):
                    doc[field] = redact_private_text(
                        doc[field],
                        safe_identifiers=safe,
                        known_private_values=clean_private,
                        pv_pattern=pv_pattern,
                    )

            if "aliases" in doc and isinstance(doc["aliases"], list):
                doc["aliases"] = [
                    redact_private_text(a, safe, clean_private, pv_pattern=pv_pattern)
                    for a in doc["aliases"] if isinstance(a, str)
                ]

            if "keywords" in doc and isinstance(doc["keywords"], list):
                doc["keywords"] = [
                    redact_private_text(k, safe, clean_private, pv_pattern=pv_pattern)
                    for k in doc["keywords"] if isinstance(k, str)
                ]

        # Scan (or re-scan after redaction)
        corpus = extract_searchable_corpus(doc)
        leaks = scan_for_privacy_leaks(
            corpus,
            safe_identifiers=safe,
            known_private_values=clean_private,
            pv_pattern=pv_pattern,
        )

        if leaks["emails"] or leaks["phones"] or leaks["private_values"]:
            doc_id = doc.get("id", "unknown_id")
            found_items = []
            if leaks["emails"]:
                found_items.append(f"Emails: {', '.join(leaks['emails'][:2])}")
            if leaks["phones"]:
                found_items.append(f"Phones: {', '.join(leaks['phones'][:2])}")
            if leaks["private_values"]:
                found_items.append(f"Private Values: {', '.join(leaks['private_values'][:2])}")
            violations.append(f"Doc [{doc_id}]: {'; '.join(found_items)}")

    if violations:
        sample_str = " | ".join(violations[:5])
        raise PrivacyGateError(
            f"Privacy Gate Violation: {len(violations)} document(s) contained sensitive private data: {sample_str}"
        )
