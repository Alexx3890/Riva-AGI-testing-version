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
import json
import openpyxl

from ..storage.qdrant_storage import get_global_qdrant_store
from .. import load_env

logger = logging.getLogger("rag.ingest")


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


INSTITUTION_NAME = os.getenv("INSTITUTION_NAME", "").strip()
DEFAULT_BRANCH = os.getenv("DEFAULT_BRANCH", "Engineering").strip()
DEFAULT_BATCH = os.getenv("DEFAULT_BATCH", "").strip()
DEFAULT_DEGREE = os.getenv("DEFAULT_DEGREE", "B.Tech").strip()

CORE_MEMBERS_ENV = os.getenv("CORE_MEMBERS", "").strip()
KNOWN_CORE_MEMBERS: Set[str] = (
    {m.strip().upper() for m in CORE_MEMBERS_ENV.split(",") if m.strip()}
    if CORE_MEMBERS_ENV
    else set()
)


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
        branch = matched_master.get("branch") or u_info.get("batch_name") or DEFAULT_BRANCH
        sec = nom.get("section") or matched_master.get("section") or ""
        mentor = to_title_case(nom.get("mentor", ""))
        email = nom.get("email") or matched_master.get("email") or ""
        batch = matched_master.get("academic_batch") or DEFAULT_BATCH
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

        summary_parts = [f"{display_name} is a student in {branch}"]
        if sec:
            summary_parts.append(f"section {sec}")
        if mentor:
            summary_parts.append(f"mentored by {mentor}")
        if INSTITUTION_NAME:
            summary_parts.append(f"at {INSTITUTION_NAME}.")
        else:
            summary_parts[-1] = summary_parts[-1] + "."
        summary = " ".join(summary_parts)

        content_lines = [
            f"Student Name: {display_name}",
            f"UID: {uid}",
        ]
        degree_val = matched_master.get("degree") or DEFAULT_DEGREE
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
                "private": {
                    "email": email,
                    "phone": nom.get("phone", ""),
                    "gender": gender,
                    "father_name": father,
                },
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
        if raw_name.upper().strip() in KNOWN_CORE_MEMBERS:
            continue
        display_name = to_title_case(raw_name)
        if not display_name:
            continue

        branch = mr.get("branch") or DEFAULT_BRANCH
        sec = mr.get("section", "")
        batch = mr.get("academic_batch") or DEFAULT_BATCH
        degree = mr.get("degree") or DEFAULT_DEGREE
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

        inst_suffix = f" at {INSTITUTION_NAME}." if INSTITUTION_NAME else "."
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
                "private": {
                    "email": email,
                    "phone": phone,
                    "category": category,
                    "date_of_admission": adm_date,
                    "admission_category": adm_cat,
                },
            },
            "is_active": True,
        }
        documents.append(doc)
        processed_roll_numbers.add(roll_no)

    logger.info(f"Generated {len(documents)} unified student knowledge documents.")
    return documents


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def assert_no_private_in_embedded_fields(documents: List[Dict[str, Any]]) -> None:
    """Enforces that no email addresses or contact details leak into vector-embedded fields."""
    bad = []
    for d in documents:
        embedded = " ".join([
            d.get("title", ""),
            d.get("summary", ""),
            d.get("content", ""),
            " ".join(d.get("aliases", [])),
            " ".join(d.get("keywords", [])),
        ])
        if _EMAIL_RE.search(embedded):
            bad.append(d.get("id", "unknown"))
    if bad:
        raise ValueError(f"Private data found in embedded fields for: {bad[:5]} (+{max(0, len(bad) - 5)} more)")


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

    # If a companion tabular file exists (e.g. CSV/XLSX sibling for scanned tables), prefer parsing it
    for companion_ext in (".csv", ".tsv", ".xlsx", ".xls"):
        sibling = filepath.with_suffix(companion_ext)
        if sibling.exists():
            return load_generic_tabular_dataset(sibling)

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
                clean_title = re.split(r"\b(?:H Block|Auditorium|Block|Lab)\b", rest)[0].strip()
                clean_title = re.sub(r"[^\x00-\x7F]+", "-", clean_title).strip("- ")
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


def load_generic_tabular_dataset(filepath: Path, max_row_docs: int = 1500) -> List[Dict[str, Any]]:
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
            logger.error(f"Error reading workbook {filepath.name}: {e}")
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

        # 2. Discover categorical columns dynamically (exclude email/phone columns)
        categorical_cols: Dict[int, Tuple[str, Counter]] = {}
        for c_idx, h in enumerate(headers):
            h_low = h.lower()
            if "email" in h_low or "phone" in h_low or "mobile" in h_low:
                continue
            vals = [r[c_idx] for r in data_rows if len(r) > c_idx and r[c_idx] and r[c_idx].lower() not in ("none", "not found", "-", "nan")]
            if not vals:
                continue
            distinct = set(vals)
            if 1 < len(distinct) <= 30 and (len(distinct) / len(vals) <= 0.65 or len(distinct) <= 10):
                categorical_cols[c_idx] = (h, Counter(vals))

        # 3. Build Table Master Overview Document
        col_list_str = ", ".join(headers)
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

        # 4. Generate Targeted Aggregation Documents (Waiting List, Attendance, Status, Categories)
        for c_idx, (h, cnt) in categorical_cols.items():
            h_low = h.lower()
            clean_h_key = re.sub(r"[^a-zA-Z0-9]+", "_", h).strip("_").lower()

            # A. Dedicated Waiting List Document
            has_waiting = any("wait" in str(v).lower() for v in cnt)
            if has_waiting:
                wl_rows = [r for r in data_rows if len(r) > c_idx and "wait" in str(r[c_idx]).lower()]
                wl_lines = [
                    f"{sub_title} - Waiting List ({h})",
                    "",
                    f"Total Students on Waiting List: {len(wl_rows)} students",
                    "",
                    "List of Waitlisted Entries:",
                ]
                for i, r in enumerate(wl_rows, 1):
                    entity = _safe_name(r[name_col_idx]) if len(r) > name_col_idx and r[name_col_idx] else f"Record {i}"
                    attrs = []
                    for ci in range(len(r)):
                        if ci != name_col_idx and ci < len(headers) and r[ci]:
                            v = r[ci]
                            if "@" not in v and not re.match(r"^\+?\d{10,12}$", v):
                                attrs.append(f"{headers[ci]}: {v}")
                    wl_lines.append(f"{i}. {entity} (" + ", ".join(attrs[:5]) + ")")

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
                    "aliases": [f"{sub_title} Waiting List", f"{sub_title} Waitlist", "Waiting List", "NextGen Waiting List", "NextGen Waitlist"],
                    "keywords": ["waiting", "list", "waitlist", "waitlisted", clean_slug, "students"],
                    "metadata": {"source": filepath.name, "waitlisted_count": len(wl_rows)},
                    "is_active": True,
                })

            # B. Dedicated Attendance Document (Orientation / Event Attendance)
            is_attendance = "attendance" in h_low or (len(cnt) <= 4 and set(cnt.keys()).issubset({"P", "A", "Present", "Absent", "p", "a"}))
            if is_attendance and len(cnt) <= 6:
                att_lines = [
                    f"{sub_title} - Attendance Summary ({h})",
                    "",
                    f"Total Attendance Entries: {sum(cnt.values())}",
                    "",
                    "Attendance Breakdown:",
                ]
                for val, count in cnt.most_common():
                    att_lines.append(f"* {val}: {count} candidates")

                p_rows = [r for r in data_rows if len(r) > c_idx and str(r[c_idx]).upper() in ("P", "PRESENT")]
                if p_rows:
                    att_lines.append(f"\nCandidates Marked Present / 'P' ({len(p_rows)} total):")
                    for i, r in enumerate(p_rows, 1):
                        entity = _safe_name(r[name_col_idx]) if len(r) > name_col_idx and r[name_col_idx] else f"Entry {i}"
                        att_lines.append(f"{i}. {entity}")

                documents.append({
                    "id": f"table_{clean_slug}_attendance_{clean_h_key}",
                    "title": f"{sub_title} - Attendance Summary ({h})",
                    "category": "aggregation",
                    "summary": f"Attendance summary for {sub_title} ({h}): " + ", ".join(f"{k}: {v}" for k, v in cnt.items()),
                    "content": "\n".join(att_lines),
                    "aliases": [f"{sub_title} Attendance", f"{sub_title} Orientation Attendance", "Orientation Attendance"],
                    "keywords": ["attendance", "present", "absent", "p", "orientation", clean_slug],
                    "metadata": {"source": filepath.name, "present_count": len(p_rows) if p_rows else 0},
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
                    cat_lines.append(f"\n### {cat_val} ({len(names)} records):")
                    cat_lines.append(", ".join(names[:40]))

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

        # 5. Row-level Structured Entity Documents
        for idx, r in enumerate(data_rows[:max_row_docs]):
            entity_name = r[name_col_idx] if len(r) > name_col_idx and r[name_col_idx] else f"Record #{idx + 1}"
            if not entity_name or entity_name.lower() in ("not found", "none", "nan"):
                continue

            row_fields = []
            row_meta: Dict[str, Any] = {"source": filepath.name, "row_index": idx + 1}
            for ci, h in enumerate(headers):
                if ci < len(r) and r[ci]:
                    val = r[ci]
                    row_meta[h] = val
                    # Mask personal emails & phone numbers from vector-embedded text
                    if "@" in val or re.match(r"^\+?\d{10,12}$", val):
                        continue
                    row_fields.append(f"* **{h}**: {val}")

            if not row_fields:
                continue

            entity_title = f"{entity_name} - {sub_title}"
            row_content = f"### {entity_name}\n**Dataset**: {sub_title}\n\n" + "\n".join(row_fields)
            clean_entity_id = re.sub(r"[^a-zA-Z0-9]+", "_", entity_name).strip("_").lower()

            documents.append({
                "id": f"rec_{clean_slug}_{idx + 1}_{clean_entity_id}"[:64],
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


def load_source_documents(source: Path, data_type: str = "auto") -> List[Dict[str, Any]]:
    """Loads knowledge documents from a file or folder supporting JSON, CSV, Markdown, text, PDF, and Excel."""
    source_path = Path(source)
    dtype = (data_type or "auto").lower()

    if source_path.is_file():
        ext = source_path.suffix.lower()
        if dtype == "json" or ext == ".json":
            return load_json_documents(source_path)
        if ext in (".csv", ".tsv", ".xlsx", ".xls"):
            return load_generic_tabular_dataset(source_path)
        if dtype in ("doc", "text", "markdown") or ext in (".md", ".txt"):
            return load_text_or_markdown(source_path)
        if dtype == "pdf" or ext == ".pdf":
            return load_pdf_documents(source_path)
        return []

    if source_path.is_dir():
        docs: List[Dict[str, Any]] = []
        handled_files: Set[Path] = set()

        # 1. Process unified student datasets if present
        has_uid = (source_path / "UID.xlsx").exists()
        has_nominal = any("Nominal" in f.name for f in source_path.glob("*.xlsx"))
        if dtype in ("student", "auto") and (has_uid or has_nominal):
            uid_path = source_path / "UID.xlsx"
            nom_paths = list(source_path.glob("*Nominal*.xlsx"))
            master_paths = list(source_path.glob("*STUDENT*.xlsx"))

            if uid_path.exists():
                handled_files.add(uid_path)
            for p in nom_paths:
                handled_files.add(p)
            for p in master_paths:
                handled_files.add(p)

            uid_records = load_uid_mapping(uid_path) if uid_path.exists() else {}
            nom_records = load_nominal_roll(nom_paths[0]) if nom_paths else {}
            master_records = load_student_list(master_paths[0]) if master_paths else []
            if uid_records or nom_records or master_records:
                docs.extend(build_unified_student_documents(nom_records, uid_records, master_records))

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
                pdf_sibling = file.with_suffix(".pdf")
                if pdf_sibling.exists():
                    handled_files.add(pdf_sibling)
                docs.extend(load_generic_tabular_dataset(file))
            elif ext in (".md", ".txt"):
                docs.extend(load_text_or_markdown(file))
            elif ext == ".pdf":
                csv_sibling = file.with_suffix(".csv")
                xlsx_sibling = file.with_suffix(".xlsx")
                if csv_sibling.exists():
                    handled_files.add(csv_sibling)
                    docs.extend(load_generic_tabular_dataset(csv_sibling))
                elif xlsx_sibling.exists():
                    handled_files.add(xlsx_sibling)
                    docs.extend(load_generic_tabular_dataset(xlsx_sibling))
                else:
                    docs.extend(load_pdf_documents(file))
        return docs

    return []


def run_ingestion(
    source_dir: Path,
    dry_run: bool = False,
    limit: Optional[int] = None,
    batch_size: int = 100,
    data_type: str = "auto",
) -> int:
    """Coordinates reading datasets, building documents, and upserting into Qdrant."""
    load_env()
    source_path = Path(source_dir)

    documents = load_source_documents(source_path, data_type=data_type)

    if not documents and not source_path.is_file():
        candidates = [
            source_path,
            Path(__file__).resolve().parent.parent.parent / "local_data",
            Path.cwd() / "local_data",
            Path.cwd() / "rag_knowledge" / "local_data",
        ]
        for c in candidates:
            if c.exists():
                documents = load_source_documents(c, data_type=data_type)
                if documents:
                    source_path = c
                    break

    if not documents:
        logger.error(f"No valid knowledge data found in {source_path}")
        return 0

    assert_no_private_in_embedded_fields(documents)

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
        help="Path to folder or file (supports .xlsx, .json, .csv, .md, .txt)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="auto",
        choices=["auto", "student", "json", "csv", "doc"],
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

    args = parser.parse_args()
    raw_path = Path(args.source)
    package_dir = Path(__file__).resolve().parent.parent

    if raw_path.is_file() or raw_path.is_dir():
        source_dir = raw_path
    elif (package_dir / args.source).exists():
        source_dir = package_dir / args.source
    elif (package_dir.parent / args.source).exists():
        source_dir = package_dir.parent / args.source
    else:
        source_dir = package_dir / args.source

    total = run_ingestion(
        source_dir=source_dir,
        dry_run=args.dry_run,
        limit=args.limit,
        batch_size=args.batch_size,
        data_type=args.type,
    )
    if not args.dry_run and total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
