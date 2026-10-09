"""Data ingestion engine for rag_knowledge."""

import argparse
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import csv
import hashlib
import json
import openpyxl

from ..storage.qdrant_storage import get_global_qdrant_store
from .. import load_env
from .privacy import assert_no_privacy_leaks, is_phone_number, is_email_address, PrivacyGateError
from .joiner import StudentEntityJoiner, make_opaque_ref_id

logger = logging.getLogger("rag.ingest")

_ID_COLUMN_PATTERN = re.compile(
    r"\b(?:id|uid|roll|urn|reg|registration|serial|sno|s_no|code|candidate_id|applicant_id|student_id|enrollment)\b",
    re.IGNORECASE,
)

_SENSITIVE_HEADER_KEYWORDS = (
    "email", "phone", "mobile", "contact",
    "father", "mother", "parent", "guardian",
    "gender", "address", "dob", "birth",
    "bank", "account", "ifsc", "caste", "aadhar", "pan", "salary", "income",
)


def is_sensitive_header(header: str) -> bool:
    """Checks if a column header refers to private contact, parental, or financial data."""
    h_low = str(header).lower()
    return any(k in h_low for k in _SENSITIVE_HEADER_KEYWORDS)


def normalize_whitespace(text: Optional[str]) -> str:
    """Collapses multiple spaces and strips whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def to_title_case(name: str) -> str:
    """Converts student/mentor names to clean Title Case."""
    clean = normalize_whitespace(name)
    if not clean:
        return ""
    parts = clean.split(" ")
    capitalized = []
    for p in parts:
        if p.upper() in {"I", "II", "III", "IV", "CSE", "CSIT", "IT", "ECE", "ME", "AIML", "AI"}:
            capitalized.append(p.upper())
        else:
            capitalized.append(p.capitalize())
    return " ".join(capitalized)


def clean_identifier(val: Optional[str]) -> str:
    """Sanitizes an ID or roll number (uppercase, alphanumeric stripped of spaces)."""
    if not val:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(val)).upper()


def get_active_sheet(wb: openpyxl.Workbook) -> Optional[Any]:
    """Safely retrieves the active worksheet, falling back to the first sheet if wb.active is None."""
    if wb.active is not None:
        return wb.active
    if wb.worksheets:
        return wb.worksheets[0]
    if wb.sheetnames:
        return wb[wb.sheetnames[0]]
    return None


def load_uid_mapping(filepath: Path) -> Dict[str, Dict[str, Any]]:
    """Loads UID.xlsx into a dictionary keyed by UID."""
    if not filepath.exists():
        logger.warning(f"UID file not found: {filepath}")
        return {}

    logger.info(f"Reading UID mapping from {filepath.name}...")
    wb = openpyxl.load_workbook(filepath, data_only=True)
    try:
        sheet = get_active_sheet(wb)
        if sheet is None:
            logger.warning(f"No valid sheet found in {filepath.name}")
            return {}

        records: Dict[str, Dict[str, Any]] = {}

        for row in list(sheet.iter_rows(values_only=True))[1:]:
            if not row or len(row) < 6:
                continue
            _, name_raw, father_raw, gender_raw, batch_raw, uid_raw = row[:6]
            uid = clean_identifier(uid_raw)
            if not uid or uid.lower() in {"studentuid", "uid", "none"}:
                continue

            records[uid] = {
                "uid": uid,
                "name": normalize_whitespace(name_raw),
                "father_name": normalize_whitespace(father_raw),
                "gender": normalize_whitespace(gender_raw).upper(),
                "batch_name": normalize_whitespace(batch_raw),
            }

        logger.info(f"Loaded {len(records)} entries from {filepath.name}")
        return records
    finally:
        wb.close()


def load_nominal_roll(filepath: Path) -> Dict[str, Dict[str, Any]]:
    """Loads Nominal Roll List (.xlsx) keyed by Student UID."""
    if not filepath.exists():
        logger.warning(f"Nominal roll file not found: {filepath}")
        return {}

    logger.info(f"Reading Nominal Roll from {filepath.name}...")
    wb = openpyxl.load_workbook(filepath, data_only=True)
    try:
        sheet = get_active_sheet(wb)
        if sheet is None:
            logger.warning(f"No valid sheet found in {filepath.name}")
            return {}

        rows = list(sheet.iter_rows(values_only=True))

        header_idx = -1
        for i, r in enumerate(rows):
            if r and any("Student UID" in str(cell or "") for cell in r):
                header_idx = i
                break

        if header_idx == -1:
            logger.error(f"Could not locate header row in {filepath.name}")
            return {}

        records: Dict[str, Dict[str, Any]] = {}
        for r in rows[header_idx + 1 :]:
            if not r or len(r) < 7:
                continue
            uid = clean_identifier(r[1])
            if not uid or uid.lower() in {"studentuid", "uid", "none", "sno"}:
                continue

            name = normalize_whitespace(r[2])
            sem = normalize_whitespace(r[3])
            sec = normalize_whitespace(r[4])
            phone = normalize_whitespace(r[5])
            email = normalize_whitespace(r[6]).lower()
            father = normalize_whitespace(r[7]) if len(r) > 7 else ""
            mentor = normalize_whitespace(r[8]) if len(r) > 8 else ""

            records[uid] = {
                "uid": uid,
                "name": name,
                "semester": sem,
                "section": sec,
                "phone": phone,
                "email": email,
                "father_name": father,
                "mentor": mentor,
            }

        logger.info(f"Loaded {len(records)} entries from Nominal Roll {filepath.name}")
        return records
    finally:
        wb.close()


def load_student_list(filepath: Path) -> List[Dict[str, Any]]:
    """Loads B.Tech 1st Year Master Student List (.xlsx)."""
    if not filepath.exists():
        logger.warning(f"Student list file not found: {filepath}")
        return []

    logger.info(f"Reading Student Master List from {filepath.name}...")
    wb = openpyxl.load_workbook(filepath, data_only=True)
    try:
        sheet = get_active_sheet(wb)
        if sheet is None:
            logger.warning(f"No valid sheet found in {filepath.name}")
            return []

        rows = list(sheet.iter_rows(values_only=True))

        if not rows:
            return []

        records: List[Dict[str, Any]] = []
        for r in rows[1:]:
            if not r or len(r) < 8:
                continue
            roll_no = clean_identifier(r[7])
            name = normalize_whitespace(r[1])
            if not name and not roll_no:
                continue
            if roll_no.lower() in {"rollnumber", "rollno", "sno"} or name.lower() in {"displayname", "name", "studentname"}:
                continue

            degree = normalize_whitespace(r[2])
            branch = normalize_whitespace(r[3])
            batch = normalize_whitespace(r[4])
            year = normalize_whitespace(r[5])
            section = normalize_whitespace(r[6])
            status = normalize_whitespace(r[8]) if len(r) > 8 else "ACTIVE"
            email = normalize_whitespace(r[9]).lower() if len(r) > 9 else ""
            phone = normalize_whitespace(r[10]) if len(r) > 10 else ""
            category = normalize_whitespace(r[11]) if len(r) > 11 else ""
            adm_date = normalize_whitespace(r[12]) if len(r) > 12 else ""
            adm_cat = normalize_whitespace(r[13]) if len(r) > 13 else ""

            records.append({
                "roll_number": roll_no,
                "name": name,
                "degree": degree,
                "branch": branch,
                "academic_batch": batch,
                "year": year,
                "section": section,
                "admission_status": status,
                "email": email,
                "phone": phone,
                "category": category,
                "date_of_admission": adm_date,
                "admission_category": adm_cat,
            })

        logger.info(f"Loaded {len(records)} entries from Student List {filepath.name}")
        return records
    finally:
        wb.close()


def get_institution_name() -> str:
    load_env()
    return os.getenv("INSTITUTION_NAME", "").strip()


def get_default_branch() -> str:
    load_env()
    return os.getenv("DEFAULT_BRANCH", "").strip()


def get_default_batch() -> str:
    load_env()
    return os.getenv("DEFAULT_BATCH", "").strip()


def get_default_degree() -> str:
    load_env()
    return os.getenv("DEFAULT_DEGREE", "").strip()


def get_known_core_members() -> Set[str]:
    load_env()
    val = os.getenv("CORE_MEMBERS", "").strip()
    return {m.strip().upper() for m in val.split(",") if m.strip()} if val else set()


def build_unified_student_documents(
    nominal_records: Dict[str, Dict[str, Any]],
    uid_records: Dict[str, Dict[str, Any]],
    master_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Unifies and de-duplicates all student records into standard RAG knowledge documents."""
    documents: List[Dict[str, Any]] = []
    processed_uids: Set[str] = set()
    processed_roll_numbers: Set[str] = set()

    master_by_name: Dict[str, Dict[str, Any]] = {}
    master_by_email: Dict[str, Dict[str, Any]] = {}
    for mr in master_records:
        n_key = normalize_whitespace(mr.get("name", "")).upper()
        if n_key and n_key not in master_by_name:
            master_by_name[n_key] = mr
        e_key = mr.get("email", "").strip().lower()
        if e_key and "@" in e_key and e_key not in master_by_email:
            master_by_email[e_key] = mr

    name_counts = Counter(normalize_whitespace(m.get("name", "")).upper() for m in master_records)
    ambiguous_count = sum(1 for k in master_by_name if name_counts.get(k, 0) > 1)
    master_by_name = {k: v for k, v in master_by_name.items() if name_counts.get(k, 0) == 1}
    if ambiguous_count > 0:
        logger.info(f"Filtered {ambiguous_count} non-unique names from name-only fallback to avoid namesake mismatch.")

    all_uids = set(nominal_records.keys()) | set(uid_records.keys())
    for uid in sorted(all_uids):
        nom = nominal_records.get(uid, {})
        u_info = uid_records.get(uid, {})

        raw_name = nom.get("name") or u_info.get("name") or ""
        display_name = to_title_case(raw_name)
        if not display_name:
            continue

        name_upper = raw_name.upper().strip()
        matched_master = master_by_email.get(nom.get("email", "")) or master_by_name.get(name_upper) or {}

        roll_no = matched_master.get("roll_number", "")
        branch = matched_master.get("branch") or u_info.get("batch_name") or get_default_branch()
        sec = nom.get("section") or matched_master.get("section") or ""
        mentor = to_title_case(nom.get("mentor", ""))
        email = nom.get("email") or matched_master.get("email") or ""
        batch = matched_master.get("academic_batch") or get_default_batch()
        sem = nom.get("semester") or matched_master.get("year") or ""
        gender = u_info.get("gender", "")
        father = to_title_case(nom.get("father_name") or u_info.get("father_name") or "")

        aliases: List[str] = [uid]
        if roll_no:
            aliases.append(roll_no)
        if display_name:
            aliases.extend([display_name, display_name.upper(), display_name.lower()])

        keywords: List[str] = ["student", branch]
        if sec:
            keywords.extend([sec, f"section {sec}"])
        if mentor:
            keywords.extend([f"mentor {mentor}", mentor])
        if sem:
            keywords.append(f"sem {sem}")

        inst_name = get_institution_name()
        summary_parts = [f"{display_name} is a student in {branch}"]
        if sec:
            summary_parts.append(f"section {sec}")
        if mentor:
            summary_parts.append(f"mentored by {mentor}")
        if inst_name:
            summary_parts.append(f"at {inst_name}.")
        else:
            summary_parts[-1] = summary_parts[-1] + "."
        summary = " ".join(summary_parts)

        content_lines = [
            f"Student Name: {display_name}",
            f"UID: {uid}",
        ]
        degree_val = matched_master.get("degree") or get_default_degree()
        if degree_val:
            content_lines.append(f"Degree: {degree_val}")
        content_lines.append(f"Branch: {branch}")
        if sec:
            content_lines.append(f"Class Section: {sec}")
        if sem:
            content_lines.append(f"Semester / Year: Semester {sem}")
        if mentor:
            content_lines.append(f"Faculty Mentor: {mentor}")
        if batch:
            content_lines.append(f"Academic Batch: {batch}")

        doc_id = f"student_{uid.lower()}"
        doc = {
            "id": doc_id,
            "category": "student",
            "title": f"{display_name} - {branch}",
            "aliases": sorted(list(set(aliases))),
            "keywords": sorted(list(set(keywords))),
            "summary": summary,
            "content": "\n".join(content_lines),
            "metadata": {
                "uid": uid,
                "roll_number": roll_no,
                "name": display_name,
                "branch": branch,
                "section": sec,
                "semester": sem,
                "mentor": mentor,
                "academic_batch": batch,
                "entity_ref_id": make_opaque_ref_id(roll_no or uid),
            },
            "is_active": True,
        }
        documents.append(doc)
        processed_uids.add(uid)
        if roll_no:
            processed_roll_numbers.add(roll_no)

    for mr in master_records:
        roll_no = mr.get("roll_number", "")
        if not roll_no or roll_no in processed_roll_numbers:
            continue

        raw_name = mr.get("name", "")
        if raw_name.upper().strip() in get_known_core_members():
            continue
        display_name = to_title_case(raw_name)
        if not display_name:
            continue

        branch = mr.get("branch") or get_default_branch()
        sec = mr.get("section", "")
        batch = mr.get("academic_batch") or get_default_batch()
        degree = mr.get("degree") or get_default_degree()
        year = mr.get("year", "")
        email = mr.get("email", "")
        category = mr.get("category", "")
        status = mr.get("admission_status", "ACTIVE")

        phone = mr.get("phone", "")
        adm_date = mr.get("date_of_admission", "")
        adm_cat = mr.get("admission_category", "")

        aliases = [roll_no, display_name, display_name.upper(), display_name.lower()]

        keywords = ["student", branch]
        if sec:
            keywords.extend([sec, f"section {sec}"])
        if batch:
            keywords.append(batch)

        inst_name = get_institution_name()
        inst_suffix = f" at {inst_name}." if inst_name else "."
        batch_info = f", Batch {batch}" if batch else ""
        sec_info = f"Section {sec}" if sec else ""
        paren_info = f" ({sec_info}{batch_info})" if (sec_info or batch_info) else ""
        year_info = f"{year} " if year else ""
        summary = f"{display_name} is a {year_info}{degree} student in {branch}{paren_info}{inst_suffix}".strip()

        content_lines = [
            f"Student Name: {display_name}",
            f"University Roll Number: {roll_no}",
            f"Degree: {degree}",
            f"Branch: {branch}",
            f"Class Section: {sec}",
            f"Current Year: Year {year}",
            f"Academic Batch: {batch}",
            f"Admission Status: {status}",
        ]

        doc_id = f"student_{roll_no.lower()}"
        doc = {
            "id": doc_id,
            "category": "student",
            "title": f"{display_name} - {branch}",
            "aliases": sorted(list(set(aliases))),
            "keywords": sorted(list(set(keywords))),
            "summary": summary,
            "content": "\n".join(content_lines),
            "metadata": {
                "roll_number": roll_no,
                "name": display_name,
                "branch": branch,
                "section": sec,
                "degree": degree,
                "academic_batch": batch,
                "admission_status": status,
                "entity_ref_id": make_opaque_ref_id(roll_no),
            },
            "is_active": True,
        }
        documents.append(doc)
        processed_roll_numbers.add(roll_no)

    logger.info(f"Generated {len(documents)} unified student knowledge documents.")
    return documents


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def assert_no_private_in_embedded_fields(
    documents: List[Dict[str, Any]],
    safe_identifiers: Optional[Set[str]] = None,
    known_private_values: Optional[Set[str]] = None,
    opt_in_redact: bool = False,
) -> None:
    """Enforces zero-leak policy across embedded fields and payloads using the unanchored privacy gate."""
    assert_no_privacy_leaks(
        documents,
        safe_identifiers=safe_identifiers,
        known_private_values=known_private_values,
        opt_in_redact=opt_in_redact,
    )


def load_json_documents(filepath: Path) -> List[Dict[str, Any]]:
    """Loads documents or Q&A pairs from a JSON file."""
    if not filepath.is_file():
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    items = data if isinstance(data, list) else [data]
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("question") or f"{filepath.stem} #{idx + 1}"
        content = item.get("content") or item.get("answer") or str(item)
        summary = item.get("summary") or item.get("answer") or content[:200]
        doc_id = str(item.get("id") or f"doc_{filepath.stem}_{idx + 1}")
        category = item.get("category") or "general"
        keywords = item.get("keywords") or [category]
        aliases = item.get("aliases") or [title]

        documents.append({
            "id": doc_id,
            "title": title,
            "category": category,
            "summary": summary,
            "content": content,
            "aliases": aliases,
            "keywords": keywords,
            "metadata": item.get("metadata", {}),
            "is_active": bool(item.get("is_active", True)),
        })
    return documents


def load_csv_documents(filepath: Path) -> List[Dict[str, Any]]:
    """Loads knowledge documents from a CSV or TSV file."""
    if not filepath.is_file():
        return []
    delimiter = "\t" if filepath.suffix.lower() == ".tsv" else ","
    documents = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for idx, row in enumerate(reader):
            r = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            title = r.get("title") or r.get("name") or r.get("question") or f"Record #{idx + 1}"
            summary = r.get("summary") or r.get("description") or ""
            content = r.get("content") or r.get("details") or "\n".join(f"{k}: {v}" for k, v in row.items() if v)
            if not summary:
                summary = content[:200]
            doc_id = r.get("id") or f"csv_{filepath.stem}_{idx + 1}"
            category = r.get("category") or "general"

            documents.append({
                "id": doc_id,
                "title": title,
                "category": category,
                "summary": summary,
                "content": content,
                "aliases": [title],
                "keywords": [category],
                "metadata": dict(row),
                "is_active": True,
            })
    return documents


def load_text_or_markdown(filepath: Path) -> List[Dict[str, Any]]:
    """Loads plain text or Markdown files into sectioned knowledge documents."""
    if not filepath.is_file():
        return []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    if not text:
        return []

    sections = re.split(r"\n(?=#{1,3}\s+)", text)
    documents = []
    for idx, sec in enumerate(sections):
        lines = [line.strip() for line in sec.strip().split("\n") if line.strip()]
        if not lines:
            continue
        first_line = lines[0].lstrip("#").strip()
        title = first_line if first_line else f"{filepath.stem} #{idx + 1}"
        content = sec.strip()
        summary = lines[1] if len(lines) > 1 else content[:200]
        doc_id = f"doc_{filepath.stem}_{idx + 1}"

        documents.append({
            "id": doc_id,
            "title": title,
            "category": "document",
            "summary": summary,
            "content": content,
            "aliases": [title, filepath.stem],
            "keywords": ["document", filepath.stem],
            "metadata": {"source": filepath.name, "section_index": idx},
            "is_active": True,
        })
    return documents


def load_pdf_documents(filepath: Path) -> List[Dict[str, Any]]:
    """Loads knowledge documents from a PDF file."""
    if not filepath.is_file():
        return []

    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed. Run `pip install pypdf` to process PDF files.")
        return []

    try:
        reader = PdfReader(str(filepath))
        text_pages = [p.extract_text() or "" for p in reader.pages]
        raw_full = "\n\n".join(text_pages)
        full_text = re.sub(r"[^\x00-\x7F]+", "-", raw_full).strip()
        if not full_text:
            logger.warning(f"No extractable text found in PDF: {filepath.name}")
            return []

        lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        row_pattern = re.compile(
            r"^(\d+)\s+(.+?)\s+(\d{1,2}(?:st|nd|rd|th)?(?:\s*-\s*\d{1,2}(?:st|nd|rd|th)?)?\s+[A-Za-z]{3,}\s+\d{4})\s+(.+)$"
        )
        schedule_items = [row_pattern.match(l) for l in lines if row_pattern.match(l)]

        documents = []
        if schedule_items:
            headers = [l for l in lines if not row_pattern.match(l) and "s.no" not in l.lower()]
            overview_title = " - ".join(headers[:3]) if headers else filepath.stem

            documents.append({
                "id": f"pdf_{filepath.stem}_overview",
                "title": overview_title,
                "category": "schedule",
                "summary": f"Schedule and activity overview for {overview_title}.",
                "content": full_text,
                "aliases": [overview_title, filepath.stem],
                "keywords": ["schedule", "activities", "events", filepath.stem],
                "metadata": {"source": filepath.name},
                "is_active": True,
            })

            for m in schedule_items:
                s_no, act_type, planned_date, rest = m.groups()
                clean_title = re.sub(r"[^\x00-\x7F]+", "-", rest.strip()).strip("- ")
                title = f"{act_type}: {clean_title}" if clean_title else f"{act_type} #{s_no}"
                summary = f"{title} is scheduled for {planned_date}."
                documents.append({
                    "id": f"event_{filepath.stem}_{s_no}",
                    "title": title,
                    "category": "event",
                    "summary": summary,
                    "content": f"Event: {title}\nType: {act_type}\nDate: {planned_date}\nDetails: {m.group(0)}",
                    "aliases": [title, clean_title, act_type],
                    "keywords": ["event", act_type.lower(), filepath.stem],
                    "metadata": {"source": filepath.name, "s_no": s_no, "date": planned_date},
                    "is_active": True,
                })
        else:
            for idx, p_text in enumerate(text_pages):
                clean_p = re.sub(r"[^\x00-\x7F]+", "-", p_text).strip()
                if not clean_p:
                    continue
                p_lines = [l.strip() for l in clean_p.splitlines() if l.strip()]
                content_lines = [l for l in p_lines if not re.match(r"^.+?\|\s*Page\s*\d+$", l, re.IGNORECASE)]
                if content_lines:
                    first_line = content_lines[0]
                    if len(content_lines) > 1 and len(first_line) < 45 and not first_line.endswith("."):
                        title = f"{first_line} - {content_lines[1]}"
                    else:
                        title = first_line
                    summary = content_lines[1] if len(content_lines) > 1 else clean_p[:200]
                else:
                    title = p_lines[0] if p_lines else f"{filepath.stem} Part {idx + 1}"
                    summary = clean_p[:200]

                category = "report" if "report" in filepath.stem.lower() else "document"
                documents.append({
                    "id": f"pdf_{filepath.stem}_{idx + 1}",
                    "title": title[:120],
                    "category": category,
                    "summary": summary[:250],
                    "content": clean_p,
                    "aliases": [title[:120], filepath.stem],
                    "keywords": [category, "document", filepath.stem] + filepath.stem.replace("_", " ").split(),
                    "metadata": {"source": filepath.name, "page": idx + 1},
                    "is_active": True,
                })

        return documents
    except Exception as e:
        logger.error(f"Error parsing PDF {filepath.name}: {e}")
        return []


def get_vision_api_key() -> str:
    """Retrieves dedicated vision API key, falling back to general GEMINI_API_KEY."""
    load_env()
    return os.getenv("GEMINI_VISION_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()


def get_vision_model() -> str:
    """Retrieves dedicated vision model from environment."""
    load_env()
    return (
        os.getenv("GEMINI_VISION_MODEL", "").strip()
        or os.getenv("GEMINI_RAG_MODEL", "").strip()
        or os.getenv("GEMINI_MODEL", "").strip()
        or os.getenv("GEMINI_DEFAULT_MODEL", "").strip()
    )


def get_vision_fallback_models() -> list[str]:
    """Retrieves fallback models for vision transcription."""
    load_env()
    raw = os.getenv("GEMINI_VISION_FALLBACK_MODELS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    from ..clients.gemini_client import get_fallback_gemini_models
    return get_fallback_gemini_models()


def is_cloud_vision_permitted(filepath: Path, allow_cloud_vision: bool = False) -> bool:
    """Checks whether cloud vision is permitted for this specific file or run."""
    if allow_cloud_vision:
        return True
    raw_env = os.getenv("ALLOW_CLOUD_VISION", "0").strip().lower()
    if raw_env in ("1", "true", "yes", "all"):
        allowlist = os.getenv("ALLOW_CLOUD_VISION_ALLOWLIST", "").strip()
        if not allowlist:
            return True
        allowed_tokens = [tok.strip().lower() for tok in allowlist.split(",") if tok.strip()]
        target_str = f"{filepath.parent.name}/{filepath.name}".lower()
        return any(tok in target_str for tok in allowed_tokens)
    return False


def extract_image_text(filepath: Path, allow_cloud_vision: bool = False) -> Tuple[str, str, Any]:
    """Extracts textual and structured content from an image via local OCR (default) or opt-in Gemini Vision.

    Returns:
        (extracted_text, extraction_method, confidence)
    """
    load_env()
    cloud_opted_in = is_cloud_vision_permitted(filepath, allow_cloud_vision=allow_cloud_vision)

    # 1. Default: Local OCR (pytesseract) to protect student privacy
    try:
        import pytesseract
        from PIL import Image
        with Image.open(filepath) as img:
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                confidence = None
                try:
                    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                    confs = [float(c) for c in data.get("conf", []) if str(c).strip() not in ("", "-1")]
                    if confs:
                        confidence = round(sum(confs) / len(confs) / 100.0, 2)
                except Exception:
                    pass
                conf_display = f"{confidence:.2f}" if confidence is not None else "unknown"
                logger.info(f"Extracted OCR text from {filepath.name} via local pytesseract (confidence={conf_display}).")
                return ocr_text, "local_ocr", confidence
    except Exception as e:
        logger.warning(f"Local OCR failed or unavailable for {filepath.name}: {e}")

    # 2. Opt-in: Gemini Vision API (only if explicitly allowed via flag or ALLOW_CLOUD_VISION=1 / allowlist)
    if cloud_opted_in:
        api_key = get_vision_api_key()
        if api_key:
            try:
                import base64
                import urllib.error
                import urllib.request

                ext = filepath.suffix.lower().lstrip(".")
                mime_map = {
                    "png": "image/png",
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "webp": "image/webp",
                    "gif": "image/gif",
                    "bmp": "image/bmp",
                }
                mime_type = mime_map.get(ext, "image/png")

                with open(filepath, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")

                primary_model = get_vision_model()
                fallback_models = get_vision_fallback_models()
                model_candidates = ([primary_model] if primary_model else []) + fallback_models
                valid_models = []
                for m in model_candidates:
                    if m and "live" not in m.lower() and m not in valid_models:
                        valid_models.append(m)

                prompt_text = (
                    "Transcribe and extract all content from this document/image accurately. "
                    "Include all questions, numbered items, formulas, answers, solutions, tables, "
                    "headings, and details in clean Markdown format."
                )

                for model_name in valid_models:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                    payload = {
                        "contents": [
                            {
                                "parts": [
                                    {"inlineData": {"mimeType": mime_type, "data": b64_data}},
                                    {"text": prompt_text},
                                ]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.0,
                            "maxOutputTokens": 4096,
                        },
                    }
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                        method="POST",
                    )
                    try:
                        with urllib.request.urlopen(req, timeout=30.0) as resp:
                            res = json.loads(resp.read().decode("utf-8"))
                            cand = res.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            if cand.strip():
                                logger.info(f"Successfully extracted text from image {filepath.name} using Gemini Vision ({model_name}).")
                                # Gemini Vision does not return an OCR word confidence score; record as unknown
                                return cand.strip(), "gemini_vision", None
                    except urllib.error.HTTPError as he:
                        logger.warning(f"Gemini Vision call for {model_name} HTTP {he.code}: {he.reason}")
                        continue
                    except Exception as ex:
                        logger.warning(f"Gemini Vision call for {model_name} failed: {ex}")
                        continue
            except Exception as e:
                logger.warning(f"Error invoking Gemini Vision for {filepath.name}: {e}")
    else:
        logger.info(f"Cloud vision is disabled for {filepath.name}. To enable, pass --cloud-vision or configure ALLOW_CLOUD_VISION.")

    # 3. If extraction failed or yielded nothing, report and skip rather than creating a junk record
    logger.warning(f"No textual content could be extracted from image {filepath.name}. Skipping indexing.")
    return "", "none", None


def load_image_documents(filepath: Path, allow_cloud_vision: bool = False) -> List[Dict[str, Any]]:
    """Loads knowledge documents from an image file using multimodal Vision or local OCR."""
    if not filepath.is_file():
        return []

    try:
        res = extract_image_text(filepath, allow_cloud_vision=allow_cloud_vision)
    except TypeError:
        res = extract_image_text(filepath)
    if isinstance(res, tuple):
        raw_text, method, confidence = res
    else:
        raw_text, method, confidence = str(res or ""), "custom", None

    if not raw_text or not raw_text.strip():
        # Do not index empty images or fallback junk records
        return []

    conf_record = confidence if confidence is not None else "unknown"
    img_meta: Dict[str, Any] = {
        "source": filepath.name,
        "file_type": "image",
        "extraction_method": method,
        "confidence": conf_record,
    }
    if isinstance(confidence, (int, float)) and confidence < 0.6:
        img_meta["flagged_for_review"] = True

    try:
        from PIL import Image
        with Image.open(filepath) as img:
            img_meta["width"] = img.size[0]
            img_meta["height"] = img.size[1]
            img_meta["dimensions"] = f"{img.size[0]}x{img.size[1]}"
            img_meta["format"] = img.format or filepath.suffix.lstrip(".").upper()
    except Exception:
        pass

    documents = []
    clean_stem = re.sub(r"[^a-zA-Z0-9]+", "_", filepath.stem).strip("_").lower()

    # Split into sections if image contains multiple questions or headings
    raw_sections = [s.strip() for s in re.split(r"\n\s*---\s*\n", raw_text) if s.strip()]
    if len(raw_sections) <= 1:
        raw_sections = [s.strip() for s in re.split(r"\n(?=#{1,3}\s+)", raw_text) if s.strip()]

    extra_keywords = ["image", "transcription", clean_stem]
    if isinstance(confidence, (int, float)) and confidence < 0.6:
        extra_keywords.append("needs_review")

    if len(raw_sections) > 1:
        # Full content overview
        documents.append({
            "id": f"img_{clean_stem}_full",
            "title": f"{filepath.stem.replace('_', ' ').title()} - Full Content",
            "category": "document",
            "summary": f"Image transcription of {filepath.name} with {len(raw_sections)} sections.",
            "content": raw_text,
            "aliases": [filepath.stem, filepath.name],
            "keywords": extra_keywords,
            "metadata": dict(img_meta),
            "is_active": True,
        })
        for idx, sec in enumerate(raw_sections):
            lines = [l.strip() for l in sec.split("\n") if l.strip()]
            sec_title = lines[0].lstrip("#* -").rstrip("*").strip() if lines else f"Section {idx + 1}"
            if len(sec_title) > 80:
                sec_title = sec_title[:80] + "..."
            summary = lines[1] if len(lines) > 1 else sec[:200]
            sec_meta = dict(img_meta)
            sec_meta["section_index"] = idx + 1

            documents.append({
                "id": f"img_{clean_stem}_{idx + 1}",
                "title": f"{filepath.stem.replace('_', ' ').title()} - {sec_title}",
                "category": "document",
                "summary": summary[:250],
                "content": sec,
                "aliases": [sec_title, filepath.stem],
                "keywords": extra_keywords + [w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", sec_title)[:5]],
                "metadata": sec_meta,
                "is_active": True,
            })
    else:
        first_line = raw_text.splitlines()[0].lstrip("#* -").strip() if raw_text else filepath.stem
        title = f"{filepath.stem.replace('_', ' ').title()}"
        if first_line and len(first_line) < 60:
            title = f"{title}: {first_line}"

        documents.append({
            "id": f"img_{clean_stem}_1",
            "title": title,
            "category": "document",
            "summary": raw_text[:250].replace("\n", " "),
            "content": raw_text,
            "aliases": [filepath.stem, filepath.name],
            "keywords": extra_keywords,
            "metadata": dict(img_meta),
            "is_active": True,
        })

    return documents


def load_generic_tabular_dataset(
    filepath: Path,
    max_row_docs: int = 1500,
    safe_identifiers: Optional[Set[str]] = None,
    known_private_values: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Universal tabular ingestion engine for CSV, TSV, and Excel spreadsheets.

    Dynamically ingests any arbitrary table without hardcoded column names or file rules:
    1. Reads headers and rows across sheets.
    2. Identifies key columns (entity names, statuses, waiting lists, attendance, categories).
    3. Produces a Master Dataset Overview document.
    4. Automatically generates dedicated Aggregation Documents:
       - Waiting List breakdown (when status/outcome values contain 'wait', 'waitlist', or 'reserve').
       - Attendance summary (when attendance fields with 'P', 'A', 'Present', 'Absent' exist).
       - Status / Outcome distribution & rosters.
       - Category / Domain / Department groupings.
    5. Produces entity-level row documents with privacy-safe masking.
    """
    if not filepath.is_file():
        return []

    ext = filepath.suffix.lower()
    table_name = filepath.stem.replace("_", " ").replace("-", " ").title()
    safe_clean = {str(s).strip().upper() for s in (safe_identifiers or set()) if s}
    private_digits = {
        re.sub(r"[^\d]", "", str(p))
        for p in (known_private_values or set())
        if p and len(re.sub(r"[^\d]", "", str(p))) >= 10
    }

    sheet_datasets: List[Tuple[str, List[str], List[List[str]]]] = []

    if ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        rows: List[List[str]] = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for r in reader:
                if any(r):
                    rows.append([str(c or "").strip() for c in r])
        if len(rows) >= 2:
            sheet_datasets.append((table_name, rows[0], rows[1:]))

    elif ext in (".xlsx", ".xls"):
        if ext == ".xls":
            logger.warning(f"File {filepath.name} is a legacy binary Excel format (.xls). openpyxl does not support binary .xls. Please convert to .xlsx or .csv.")
            return []
        try:
            wb = openpyxl.load_workbook(filepath, data_only=True)
            for sname in wb.sheetnames:
                sheet = wb[sname]
                s_rows: List[List[str]] = []
                for r in sheet.iter_rows(values_only=True):
                    if any(c is not None and str(c).strip() for c in r):
                        s_rows.append([str(c or "").strip() if c is not None else "" for c in r])
                if len(s_rows) >= 2:
                    sub_title = table_name if len(wb.sheetnames) == 1 else f"{table_name} ({sname})"
                    sheet_datasets.append((sub_title, s_rows[0], s_rows[1:]))
            wb.close()
        except Exception as e:
            logger.warning(f"Could not read workbook {filepath.name}: {e}")
            return []

    if not sheet_datasets:
        return []

    documents: List[Dict[str, Any]] = []

    for sub_title, raw_headers, data_rows in sheet_datasets:
        headers = [normalize_whitespace(h) for h in raw_headers]
        if not headers or not data_rows:
            continue

        # 1. Identify primary entity name / title column (exclude email/phone headers)
        name_col_idx = -1
        for i, h in enumerate(headers):
            h_low = h.lower()
            if "email" in h_low or "phone" in h_low or "mobile" in h_low:
                continue
            if h_low in ("name", "student name", "candidate name", "full name", "title", "applicant name"):
                name_col_idx = i
                break
        if name_col_idx == -1:
            for i, h in enumerate(headers):
                h_low = h.lower()
                if "email" in h_low or "phone" in h_low or "mobile" in h_low:
                    continue
                if any(k in h_low for k in ["name", "candidate", "applicant", "student", "title", "event"]):
                    name_col_idx = i
                    break
        if name_col_idx == -1:
            for i, h in enumerate(headers):
                h_low = h.lower()
                if "email" in h_low or "phone" in h_low or "mobile" in h_low:
                    continue
                vals = [r[i] for r in data_rows if len(r) > i and r[i]]
                if len(vals) > 0 and len(set(vals)) / len(vals) > 0.4 and not all(v.replace(".", "", 1).isdigit() for v in vals):
                    name_col_idx = i
                    break
        if name_col_idx == -1:
            name_col_idx = 0

        def _safe_name(raw: str) -> str:
            clean = raw.split("@")[0] if "@" in raw else raw
            clean = re.sub(r"\+?\d{10,12}", "", clean).strip()
            return clean if clean else "Candidate"

        # 2. Discover categorical columns dynamically (exclude private/contact columns)
        categorical_cols: Dict[int, Tuple[str, Counter]] = {}
        for c_idx, h in enumerate(headers):
            h_low = h.lower()
            if any(k in h_low for k in ["email", "phone", "mobile", "contact", "father", "parent", "guardian", "gender", "address", "dob", "birth"]):
                continue
            vals = [r[c_idx] for r in data_rows if len(r) > c_idx and r[c_idx] and r[c_idx].lower() not in ("none", "not found", "-", "nan")]
            if not vals:
                continue
            distinct = set(vals)
            if 1 <= len(distinct) <= 30 and (len(distinct) / len(vals) <= 0.65 or len(distinct) <= 10):
                categorical_cols[c_idx] = (h, Counter(vals))

        # 3. Build Table Master Overview Document
        safe_headers = [
            h for h in headers
            if not any(k in h.lower() for k in ["email", "phone", "mobile", "contact", "father", "parent", "guardian", "gender", "address", "dob", "birth"])
        ]
        col_list_str = ", ".join(safe_headers or headers)
        overview_lines = [
            f"Dataset: {sub_title}",
            f"Source File: {filepath.name}",
            f"Total Records: {len(data_rows)}",
            f"Available Attributes: {col_list_str}",
            "",
            "Key Summary Statistics:",
        ]
        for c_idx, (h, cnt) in categorical_cols.items():
            overview_lines.append(f"\nBreakdown of {h} ({len(cnt)} distinct categories):")
            for val, count in cnt.most_common(12):
                overview_lines.append(f"* {val}: {count} records")

        clean_slug = re.sub(r"[^a-zA-Z0-9]+", "_", sub_title).strip("_").lower()
        documents.append({
            "id": f"table_{clean_slug}_overview",
            "title": f"{sub_title} - Dataset Overview",
            "category": "dataset_overview",
            "summary": f"Overview of {sub_title} ({len(data_rows)} records, columns: {col_list_str[:160]}).",
            "content": "\n".join(overview_lines),
            "aliases": [sub_title, f"{sub_title} Overview", f"{sub_title} Summary"],
            "keywords": ["dataset", "table", "overview", "summary"] + clean_slug.split("_"),
            "metadata": {"source": filepath.name, "total_records": len(data_rows)},
            "is_active": True,
        })

        # Helper to filter out sensitive attributes for waitlist and aggregations
        def _get_safe_row_attributes(r_row: List[str]) -> List[str]:
            attrs = []
            for ci in range(len(r_row)):
                if ci != name_col_idx and ci < len(headers) and r_row[ci]:
                    if is_sensitive_header(headers[ci]):
                        continue
                    v = str(r_row[ci]).strip()
                    if is_email_address(v) or is_phone_number(v):
                        continue
                    attrs.append(f"{headers[ci]}: {v}")
            return attrs

        # 4. Generate Targeted Aggregation Documents (Waiting List, Attendance, Status, Categories)
        for c_idx, (h, cnt) in categorical_cols.items():
            h_low = h.lower()
            clean_h_key = re.sub(r"[^a-zA-Z0-9]+", "_", h).strip("_").lower()

            # A. Dedicated Waiting List Document (with column filtering and part chunking)
            has_waiting = any("wait" in str(v).lower() for v in cnt)
            if has_waiting:
                wl_rows = [r for r in data_rows if len(r) > c_idx and "wait" in str(r[c_idx]).lower()]
                wl_entries = []
                for i, r in enumerate(wl_rows, 1):
                    entity = _safe_name(r[name_col_idx]) if len(r) > name_col_idx and r[name_col_idx] else f"Record {i}"
                    safe_attrs = _get_safe_row_attributes(r)
                    attr_str = f" ({', '.join(safe_attrs[:5])})" if safe_attrs else ""
                    wl_entries.append(f"{i}. {entity}{attr_str}")

                wl_chunk_size = 40
                if len(wl_rows) <= wl_chunk_size:
                    wl_lines = [
                        f"{sub_title} - Waiting List ({h})",
                        "",
                        f"Total Students on Waiting List: {len(wl_rows)} students",
                        "",
                        "List of Waitlisted Entries:",
                    ]
                    wl_lines.extend(wl_entries)
                    wl_lines.extend([
                        "",
                        f"Summary: Exactly {len(wl_rows)} entries are placed on the waiting list in {sub_title}.",
                    ])

                    documents.append({
                        "id": f"table_{clean_slug}_waiting_list",
                        "title": f"{sub_title} - Waiting List ({h})",
                        "category": "aggregation",
                        "summary": f"There are exactly {len(wl_rows)} students on the waiting list (waitlist) in {sub_title}.",
                        "content": "\n".join(wl_lines),
                        "aliases": [f"{sub_title} Waiting List", f"{sub_title} Waitlist", "Waiting List"],
                        "keywords": ["waiting", "list", "waitlist", "waitlisted", clean_slug],
                        "metadata": {"source": filepath.name, "waitlisted_count": len(wl_rows)},
                        "is_active": True,
                    })
                else:
                    num_parts = (len(wl_rows) + wl_chunk_size - 1) // wl_chunk_size
                    for part_idx in range(num_parts):
                        chunk_entries = wl_entries[part_idx * wl_chunk_size : (part_idx + 1) * wl_chunk_size]
                        chunk_lines = [
                            f"{sub_title} - Waiting List ({h}) (Part {part_idx + 1}/{num_parts})",
                            "",
                            f"Total Students on Waiting List: {len(wl_rows)} students (Displaying {len(chunk_entries)} in this part)",
                            "",
                            "List of Waitlisted Entries:",
                        ]
                        chunk_lines.extend(chunk_entries)
                        chunk_lines.extend([
                            "",
                            f"Summary: Exactly {len(wl_rows)} total entries on the waiting list in {sub_title} (Part {part_idx + 1}/{num_parts}).",
                        ])
                        documents.append({
                            "id": f"table_{clean_slug}_waiting_list_part{part_idx + 1}",
                            "title": f"{sub_title} - Waiting List ({h}) (Part {part_idx + 1}/{num_parts}, Total: {len(wl_rows)})",
                            "category": "aggregation",
                            "summary": f"Waiting list for {sub_title} ({len(wl_rows)} total, Part {part_idx + 1}/{num_parts}).",
                            "content": "\n".join(chunk_lines),
                            "aliases": [f"{sub_title} Waiting List Part {part_idx + 1}", f"{sub_title} Waitlist"],
                            "keywords": ["waiting", "list", "waitlist", "waitlisted", clean_slug],
                            "metadata": {"source": filepath.name, "waitlisted_count": len(wl_rows), "part": part_idx + 1, "total_parts": num_parts},
                            "is_active": True,
                        })

            # B. Dedicated Attendance Document (with part chunking for large rosters)
            is_attendance = "attendance" in h_low or (len(cnt) <= 4 and set(cnt.keys()).issubset({"P", "A", "Present", "Absent", "p", "a"}))
            if is_attendance and len(cnt) <= 6:
                p_rows = [r for r in data_rows if len(r) > c_idx and str(r[c_idx]).upper() in ("P", "PRESENT")]
                p_entries = [
                    f"{i}. {_safe_name(r[name_col_idx]) if len(r) > name_col_idx and r[name_col_idx] else f'Entry {i}'}"
                    for i, r in enumerate(p_rows, 1)
                ]

                att_chunk_size = 40
                if len(p_entries) <= att_chunk_size:
                    att_lines = [
                        f"{sub_title} - Attendance Summary ({h})",
                        "",
                        f"Total Attendance Entries: {sum(cnt.values())}",
                        "",
                        "Attendance Breakdown:",
                    ]
                    for val, count in cnt.most_common():
                        att_lines.append(f"* {val}: {count} candidates")
                    if p_entries:
                        att_lines.append(f"\nCandidates Marked Present / 'P' ({len(p_entries)} total):")
                        att_lines.extend(p_entries)

                    documents.append({
                        "id": f"table_{clean_slug}_attendance_{clean_h_key}",
                        "title": f"{sub_title} - Attendance Summary ({h})",
                        "category": "aggregation",
                        "summary": f"Attendance summary for {sub_title} ({h}): " + ", ".join(f"{k}: {v}" for k, v in cnt.items()),
                        "content": "\n".join(att_lines),
                        "aliases": [f"{sub_title} Attendance", "Attendance Summary", "Attendance"],
                        "keywords": ["attendance", "present", "absent", "roster", clean_slug],
                        "metadata": {"source": filepath.name, "present_count": len(p_rows) if p_rows else 0},
                        "is_active": True,
                    })
                else:
                    num_parts = (len(p_entries) + att_chunk_size - 1) // att_chunk_size
                    for part_idx in range(num_parts):
                        chunk_entries = p_entries[part_idx * att_chunk_size : (part_idx + 1) * att_chunk_size]
                        att_lines = [
                            f"{sub_title} - Attendance Summary ({h}) (Part {part_idx + 1}/{num_parts})",
                            "",
                            f"Total Attendance Entries: {sum(cnt.values())} (Displaying {len(chunk_entries)} present in this part)",
                            "",
                            "Attendance Breakdown:",
                        ]
                        for val, count in cnt.most_common():
                            att_lines.append(f"* {val}: {count} candidates")
                        att_lines.append(f"\nCandidates Marked Present (Part {part_idx + 1}/{num_parts}, Total: {len(p_entries)}):")
                        att_lines.extend(chunk_entries)

                        documents.append({
                            "id": f"table_{clean_slug}_attendance_{clean_h_key}_part{part_idx + 1}",
                            "title": f"{sub_title} - Attendance Summary ({h}) (Part {part_idx + 1}/{num_parts}, Total: {len(p_entries)})",
                            "category": "aggregation",
                            "summary": f"Attendance summary for {sub_title} ({h}) (Part {part_idx + 1}/{num_parts}, {len(p_entries)} total present).",
                            "content": "\n".join(att_lines),
                            "aliases": [f"{sub_title} Attendance Part {part_idx + 1}", "Attendance Summary"],
                            "keywords": ["attendance", "present", "absent", "roster", clean_slug],
                            "metadata": {"source": filepath.name, "present_count": len(p_rows), "part": part_idx + 1, "total_parts": num_parts},
                            "is_active": True,
                        })

            # C. Category / Domain / Branch Breakdown
            is_category = any(k in h_low for k in ["domain", "branch", "department", "category", "role", "section", "status", "result"])
            if is_category and len(cnt) <= 25:
                cat_lines = [
                    f"{sub_title} - Breakdown by {h}",
                    "",
                    f"Distribution across {len(cnt)} categories:",
                ]
                for val, count in cnt.most_common():
                    cat_lines.append(f"* {val}: {count} records")

                by_cat = defaultdict(list)
                for r in data_rows:
                    if len(r) > c_idx and r[c_idx]:
                        entity = _safe_name(r[name_col_idx]) if len(r) > name_col_idx and r[name_col_idx] else "Entry"
                        by_cat[r[c_idx]].append(entity)

                for cat_val, names in sorted(by_cat.items(), key=lambda x: len(x[1]), reverse=True):
                    cat_slug = re.sub(r"[^a-zA-Z0-9]+", "_", cat_val).strip("_").lower()
                    chunk_size = 40
                    if len(names) <= chunk_size:
                        cat_lines.append(f"\n### {cat_val} ({len(names)} records):")
                        cat_lines.append(", ".join(names))
                    else:
                        num_parts = (len(names) + chunk_size - 1) // chunk_size
                        for part_idx in range(num_parts):
                            chunk_names = names[part_idx * chunk_size : (part_idx + 1) * chunk_size]
                            chunk_lines = [
                                f"{sub_title} - {h}: {cat_val} (Part {part_idx + 1}/{num_parts})",
                                "",
                                f"Total records in {cat_val}: {len(names)} (Displaying {len(chunk_names)} in this part)",
                                "",
                                ", ".join(chunk_names),
                            ]
                            documents.append({
                                "id": f"table_{clean_slug}_{clean_h_key}_{cat_slug}_part{part_idx + 1}",
                                "title": f"{sub_title} - {h}: {cat_val} (Part {part_idx + 1}/{num_parts}, Total: {len(names)})",
                                "category": "aggregation",
                                "summary": f"{sub_title} members for {cat_val} in {h} (Part {part_idx + 1}/{num_parts}, {len(names)} total).",
                                "content": "\n".join(chunk_lines),
                                "aliases": [f"{sub_title} {cat_val}", f"{cat_val} {h}"],
                                "keywords": [h.lower(), clean_slug, cat_slug, "breakdown"],
                                "metadata": {"source": filepath.name, "column": h, "category_value": cat_val, "total_records": len(names)},
                                "is_active": True,
                            })

                documents.append({
                    "id": f"table_{clean_slug}_{clean_h_key}",
                    "title": f"{sub_title} - Breakdown by {h}",
                    "category": "aggregation",
                    "summary": f"{sub_title} breakdown across {len(cnt)} groups in {h}.",
                    "content": "\n".join(cat_lines),
                    "aliases": [f"{sub_title} {h}", f"{sub_title} by {h}"],
                    "keywords": [h.lower(), clean_slug, "breakdown", "category"],
                    "metadata": {"source": filepath.name, "column": h},
                    "is_active": True,
                })

        # 5. Row-level Structured Entity Documents (Process all rows with stable IDs and identifier exemptions)
        for idx, r in enumerate(data_rows):
            entity_name = r[name_col_idx] if len(r) > name_col_idx and r[name_col_idx] else f"Record #{idx + 1}"
            if not entity_name or entity_name.lower() in ("not found", "none", "nan"):
                continue

            row_fields = []
            row_meta: Dict[str, Any] = {"source": filepath.name, "row_index": idx + 1}
            id_val = None

            # Collect contact values from sensitive columns in this row to detect accidental duplicate entries in ID columns
            row_contact_digits = set()
            for ci, h in enumerate(headers):
                if ci < len(r) and r[ci]:
                    val_str = str(r[ci]).strip()
                    if is_sensitive_header(h):
                        digs = re.sub(r"[^\d]", "", val_str)
                        if len(digs) >= 10:
                            row_contact_digits.add(digs)

            for ci, h in enumerate(headers):
                if ci < len(r) and r[ci]:
                    val = str(r[ci]).strip()

                    # Check if column is an identifier column using word-boundary matching
                    # Prevents substrings in words like 'residence', 'valid', 'paid', 'guide' from matching 'id'
                    is_contact_header = is_sensitive_header(h)
                    is_id_col = bool(_ID_COLUMN_PATTERN.search(h)) and not is_contact_header

                    # Strictly filter private contact details from both metadata payload and embedded text
                    if is_contact_header:
                        continue
                    if is_email_address(val):
                        continue

                    digits = re.sub(r"[^\d]", "", val)
                    is_known_phone = bool(len(digits) >= 10 and (digits in row_contact_digits or digits in private_digits))
                    val_is_safe = bool(safe_clean and (val.upper() in safe_clean or clean_identifier(val).upper() in safe_clean))

                    # If in an ID column, filter out if it actually matches a known phone number
                    # from the row/master list (and is not an approved safe identifier). Otherwise allow valid IDs.
                    if is_id_col:
                        if is_known_phone and not val_is_safe:
                            continue
                        if not id_val:
                            id_val = clean_identifier(val)
                    else:
                        # Non-ID columns are strictly checked for phone numbers
                        if is_phone_number(val) and not val_is_safe:
                            continue
                        if len(digits) >= 10 and (val.startswith("+") or digits.startswith(("6", "7", "8", "9"))) and not val_is_safe:
                            continue

                    row_meta[h] = val
                    row_fields.append(f"* **{h}**: {val}")

            if not row_fields:
                continue

            entity_title = f"{entity_name} - {sub_title}"
            row_content = f"### {entity_name}\n**Dataset**: {sub_title}\n\n" + "\n".join(row_fields)
            clean_entity_id = re.sub(r"[^a-zA-Z0-9]+", "_", entity_name).strip("_").lower()

            # Deterministic, position-independent ID based on unique identifier or normalized entity name
            if id_val:
                doc_seed = f"{clean_slug}_{id_val}"
            else:
                doc_seed = f"{clean_slug}_{clean_entity_id}"
            stable_doc_id = f"rec_{clean_slug[:16]}_{hashlib.sha256(doc_seed.encode()).hexdigest()[:16]}"

            documents.append({
                "id": stable_doc_id,
                "title": entity_title[:120],
                "category": "record",
                "summary": f"Record for {entity_name} in {sub_title}. " + "; ".join(row_fields[:3])[:180],
                "content": row_content,
                "aliases": [entity_name, entity_title[:120]],
                "keywords": ["record", clean_slug] + entity_name.lower().split(),
                "metadata": row_meta,
                "is_active": True,
            })

    return documents


def load_source_documents(
    source: Path,
    data_type: str = "auto",
    allow_cloud_vision: bool = False,
    safe_identifiers: Optional[Set[str]] = None,
    known_private_values: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Loads knowledge documents from a file or folder supporting JSON, CSV, Markdown, text, PDF, and Excel."""
    source_path = Path(source)
    dtype = (data_type or "auto").lower()

    if source_path.is_file():
        ext = source_path.suffix.lower()
        if dtype == "json" or ext == ".json":
            return load_json_documents(source_path)
        if ext in (".csv", ".tsv", ".xlsx", ".xls"):
            return load_generic_tabular_dataset(
                source_path,
                safe_identifiers=safe_identifiers,
                known_private_values=known_private_values,
            )
        if dtype in ("doc", "text", "markdown") or ext in (".md", ".txt"):
            return load_text_or_markdown(source_path)
        if dtype == "pdf" or ext == ".pdf":
            return load_pdf_documents(source_path)
        if dtype in ("image", "vision") or ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"):
            return load_image_documents(source_path, allow_cloud_vision=allow_cloud_vision)
        return []

    if source_path.is_dir():
        docs: List[Dict[str, Any]] = []
        handled_files: Set[Path] = set()
        active_safe_ids = set(safe_identifiers) if safe_identifiers else set()
        active_private_vals = set(known_private_values) if known_private_values else set()

        # 1. Process unified student datasets using canonical StudentEntityJoiner
        has_uid = (source_path / "UID.xlsx").exists()
        has_nominal = any("Nominal" in f.name for f in source_path.glob("*.xlsx"))
        has_student = any("STUDENT" in f.name for f in source_path.glob("*.xlsx"))
        if dtype in ("student", "auto") and (has_uid or has_nominal or has_student):
            joiner = StudentEntityJoiner(source_path)
            student_docs = joiner.to_knowledge_documents()
            if student_docs:
                docs.extend(student_docs)
                active_safe_ids.update(joiner.safe_identifiers)
                active_private_vals.update(joiner.known_private_values)
                if safe_identifiers is not None:
                    safe_identifiers.update(joiner.safe_identifiers)
                if known_private_values is not None:
                    known_private_values.update(joiner.known_private_values)

                uid_path = source_path / "UID.xlsx"
                if uid_path.exists():
                    handled_files.add(uid_path)
                for p in source_path.glob("*Nominal*.xlsx"):
                    handled_files.add(p)
                for p in source_path.glob("*STUDENT*.xlsx"):
                    handled_files.add(p)

        if dtype == "student":
            return docs

        # 2. Process all remaining files in directory using universal processors
        for file in sorted(source_path.iterdir()):
            if not file.is_file() or file in handled_files:
                continue
            ext = file.suffix.lower()
            if ext == ".json":
                docs.extend(load_json_documents(file))
            elif ext in (".csv", ".tsv", ".xlsx", ".xls"):
                docs.extend(load_generic_tabular_dataset(
                    file,
                    safe_identifiers=active_safe_ids,
                    known_private_values=active_private_vals,
                ))
            elif ext in (".md", ".txt"):
                docs.extend(load_text_or_markdown(file))
            elif ext == ".pdf":
                docs.extend(load_pdf_documents(file))
            elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"):
                docs.extend(load_image_documents(file, allow_cloud_vision=allow_cloud_vision))
        return docs

    return []


def run_ingestion(
    source_dir: Path,
    dry_run: bool = False,
    limit: Optional[int] = None,
    batch_size: int = 100,
    data_type: str = "auto",
    redact: bool = False,
    allow_cloud_vision: bool = False,
) -> int:
    """Coordinates reading datasets, building documents, and upserting into Qdrant."""
    load_env()
    cleaned_source = str(source_dir).strip(' "\'\r\n\t') if isinstance(source_dir, (str, Path)) else source_dir
    source_path = Path(cleaned_source)

    safe_identifiers: Set[str] = set()
    known_private_values: Set[str] = set()

    documents = load_source_documents(
        source_path,
        data_type=data_type,
        allow_cloud_vision=allow_cloud_vision,
        safe_identifiers=safe_identifiers,
        known_private_values=known_private_values,
    )

    if not documents and not source_path.is_file():
        candidates = [
            source_path,
            Path(__file__).resolve().parent.parent.parent / "local_data",
            Path.cwd() / "local_data",
            Path.cwd() / "rag_knowledge" / "local_data",
        ]
        for c in candidates:
            if c.exists():
                documents = load_source_documents(
                    c,
                    data_type=data_type,
                    allow_cloud_vision=allow_cloud_vision,
                    safe_identifiers=safe_identifiers,
                    known_private_values=known_private_values,
                )
                if documents:
                    source_path = c
                    break

    if not documents:
        logger.error(f"No valid knowledge data found in {source_path}")
        return 0

    assert_no_private_in_embedded_fields(
        documents,
        safe_identifiers=safe_identifiers,
        known_private_values=known_private_values,
        opt_in_redact=redact,
    )

    if limit and limit > 0:
        documents = documents[:limit]
        logger.info(f"Limited document ingestion to first {limit} records.")

    print(f"\n=======================================================")
    print(f"Target Backend: QDRANT VECTOR DATABASE")
    print(f"Total Unified Knowledge Documents: {len(documents)}")
    print(f"=======================================================\n")

    if dry_run:
        print("[DRY-RUN MODE] Showing sample documents that would be upserted:\n")
        sample_count = min(3, len(documents))
        for i in range(sample_count):
            d = documents[i]
            print(f"--- Document #{i+1}: [{d['id']}] ---")
            print(f"Title:    {d['title']}")
            print(f"Category: {d.get('category', 'general')}")
            print(f"Aliases:  {', '.join(d.get('aliases', [])[:6])}")
            print(f"Keywords: {', '.join(d.get('keywords', [])[:6])}")
            print(f"Summary:  {d['summary']}")
            print(f"Content Preview:\n{d.get('content', '')[:250]}...")
            print()
        print(f"[DRY-RUN COMPLETE] Zero changes were made to Qdrant.")
        return len(documents)

    store = get_global_qdrant_store()
    store.timeout = float(os.getenv("QDRANT_INGEST_TIMEOUT", "30.0"))
    if not store.is_available():
        print(f"[ERROR] Qdrant storage is unreachable. Verify Qdrant configuration in .env.")
        sys.exit(1)

    print(f"Upserting {len(documents)} documents to Qdrant Cloud (collection: '{store.collection_name}') in batches of {batch_size}...")
    total_upserted = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        try:
            count = store.upsert_documents(batch)
            total_upserted += count
            print(f"  * Batch {i // batch_size + 1}: Upserted {count} documents (Progress: {total_upserted}/{len(documents)})")
        except Exception as e:
            logger.error(f"Error upserting batch {i // batch_size + 1}: {e}", exc_info=True)

    if total_upserted != len(documents):
        print(f"\n[INGESTION FAILURE] Only {total_upserted}/{len(documents)} documents were successfully upserted.")
        sys.exit(1)

    print(f"\n[INGESTION SUCCESS] Successfully upserted all {total_upserted} documents into Qdrant Cloud!")
    return total_upserted


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="RAG Knowledge Data Ingestion Pipeline (Qdrant)")
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("RAG_DATA_SOURCE", "data/raw"),
        help="Path to folder or file (supports .xlsx, .json, .csv, .md, .txt, .pdf, .png)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="auto",
        choices=["auto", "student", "json", "csv", "doc", "pdf", "image", "table"],
        help="Data type format (default: auto)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate ingestion and display sample documents without updating database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to ingest",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=150,
        help="Bulk upsert batch size",
    )
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Opt-in to redacting detected personal emails and phone numbers instead of failing (PRD S4)",
    )
    parser.add_argument(
        "--cloud-vision",
        action="store_true",
        help="Opt-in to Google Gemini Cloud Vision for processing images (default: local OCR only)",
    )
    parser.add_argument(
        "--clear",
        "--delete-all",
        action="store_true",
        help="Delete all documents from the Qdrant Cloud collection and reset it empty",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Confirm destructive operations such as --clear without interactive prompt",
    )

    args = parser.parse_args()

    if getattr(args, "clear", False):
        store = get_global_qdrant_store()
        if not store.is_available():
            print(f"[ERROR] Qdrant storage is unreachable. Verify Qdrant configuration in .env.")
            sys.exit(1)

        # Safety check: Refuse to clear active live collection alias or if live alias is unconfigured
        live_alias = os.getenv("QDRANT_LIVE_COLLECTION", "").strip() or os.getenv("LIVE_COLLECTION", "").strip()
        if not live_alias:
            print(f"[ERROR] Safety policy violation: Neither 'QDRANT_LIVE_COLLECTION' nor 'LIVE_COLLECTION' is configured.")
            print(f"Refusing to execute destructive --clear on '{store.collection_name}' without a protected live collection policy.")
            sys.exit(1)
        if store.collection_name.strip().lower() == live_alias.lower():
            logger.error(f"Refusing to clear live production collection '{store.collection_name}'.")
            print(f"[ERROR] Refusing to clear collection '{store.collection_name}' because it matches active LIVE collection alias ({live_alias}).")
            sys.exit(1)

        # Require explicit confirmation
        if not getattr(args, "yes", False):
            if sys.stdin.isatty():
                ans = input(f"Are you sure you want to completely clear target collection '{store.collection_name}'? (yes/no): ").strip().lower()
                if ans != "yes":
                    print("[ABORT] Clear cancelled by user.")
                    sys.exit(0)
            else:
                print(f"[ERROR] --clear requires confirmation. Pass --yes flag to confirm clearing target collection '{store.collection_name}'.")
                sys.exit(1)

        ok = store.clear_all()
        if ok:
            print(f"[SUCCESS] All documents successfully deleted from Qdrant Cloud (collection: '{store.collection_name}'). The database is completely cleared.")
            sys.exit(0)
        else:
            print(f"[ERROR] Failed to clear Qdrant collection '{store.collection_name}'.")
            sys.exit(1)

    raw_source = (args.source or "").strip(' "\'\r\n\t')
    raw_path = Path(raw_source)
    package_dir = Path(__file__).resolve().parent.parent

    candidates = [
        raw_path,
        Path.cwd() / raw_source,
        package_dir.parent / raw_source,
        package_dir / raw_source,
    ]
    if raw_source.startswith("rag_knowledge/") or raw_source.startswith("rag_knowledge\\"):
        sub_rel = raw_source[len("rag_knowledge") + 1:]
        candidates.append(package_dir / sub_rel)
        candidates.append(Path.cwd() / sub_rel)

    source_dir = None
    for cand in candidates:
        if cand.is_file() or cand.is_dir():
            source_dir = cand
            break

    if source_dir is None:
        source_dir = raw_path

    try:
        total = run_ingestion(
            source_dir=source_dir,
            dry_run=args.dry_run,
            limit=args.limit,
            batch_size=args.batch_size,
            data_type=args.type,
            redact=args.redact,
            allow_cloud_vision=args.cloud_vision,
        )
        if not args.dry_run and total == 0:
            sys.exit(1)
    except PrivacyGateError as pge:
        print(f"\n[ERROR] Ingestion blocked by Privacy Gate:")
        print(f"{pge}")
        print(f"\nTo automatically redact detected contact details before upserting, re-run with --redact:")
        print(f"  .venv\\Scripts\\python.exe -m rag_knowledge.ingestion --source \"{raw_source}\" --redact\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
