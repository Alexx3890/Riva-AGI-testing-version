"""Data Ingestion Engine for rag_knowledge.

Parses, cleans, unifies, and upserts student and institutional datasets
from Excel files (.xlsx) into Qdrant Cloud for zero-latency RAG retrieval.

Supported datasets:
1. Nominal Roll List (CSIT / Department cohorts with UID, mentor, section)
2. UID List (UID, student name, father name, gender, batch)
3. B.Tech Student Master List (All branches, university roll number, branch, batch, admission status)

Usage:
    python -m rag_knowledge.ingestion.ingest --dry-run
    python -m rag_knowledge.ingestion.ingest --dry-run --limit 5
    python -m rag_knowledge.ingestion.ingest
    python -m rag_knowledge.ingestion.ingest --source rag_knowledge/data/raw
"""

import argparse
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    # Capitalize each word properly while keeping Roman numerals and acronyms
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
            # Expected row format: ('UID', 'Student Name', 'Father Name ', ' Gender', 'Batch Name', 'UID')
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
            # ('S.No', 'Student UID', 'Student Name', 'Sem', 'Sec', 'Student Phone Number', 'Student Email', 'Father Name', 'Mentor')
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
        # Header format: ('S.No.', 'Display Name ', 'Degree', 'Branch', 'Academic Batch', 'Current Year', 'Class Section', 'Roll Number', 'Admission Status', 'Email ID', 'Mobile Number', 'Category', 'Date of Admission', 'Admission Category')
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


KNOWN_CORE_MEMBERS: Set[str] = {
    "ANKIT SINGH TOMAR",
    "CHIRAG TEJASVI",
    "RAJ OJHA",
    "AYUSH PATHAK",
}


def build_unified_student_documents(
    nominal_records: Dict[str, Dict[str, Any]],
    uid_records: Dict[str, Dict[str, Any]],
    master_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Unifies and de-duplicates all student records into standard RAG knowledge documents."""
    documents: List[Dict[str, Any]] = []
    processed_uids: Set[str] = set()
    processed_roll_numbers: Set[str] = set()

    # Index master records by normalized name and email for cross-matching
    master_by_name: Dict[str, Dict[str, Any]] = {}
    master_by_email: Dict[str, Dict[str, Any]] = {}
    for mr in master_records:
        n_key = normalize_whitespace(mr.get("name", "")).upper()
        if n_key and n_key not in master_by_name:
            master_by_name[n_key] = mr
        e_key = mr.get("email", "").strip().lower()
        if e_key and "@" in e_key and e_key not in master_by_email:
            master_by_email[e_key] = mr

    # Prevent namesake mismatch: only fallback to name match if name is unique across master list
    name_counts = Counter(normalize_whitespace(m.get("name", "")).upper() for m in master_records)
    ambiguous_count = sum(1 for k in master_by_name if name_counts.get(k, 0) > 1)
    master_by_name = {k: v for k, v in master_by_name.items() if name_counts.get(k, 0) == 1}
    if ambiguous_count > 0:
        logger.info(f"Filtered {ambiguous_count} non-unique names from name-only fallback to avoid namesake mismatch.")

    # 1. First process Nominal Roll students (rich with UID, section, mentor)
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
        branch = matched_master.get("branch") or u_info.get("batch_name") or "Computer Science and Information Technology"
        sec = nom.get("section") or matched_master.get("section") or ""
        mentor = to_title_case(nom.get("mentor", ""))
        email = nom.get("email") or matched_master.get("email") or ""
        batch = matched_master.get("academic_batch") or "2025-2029"
        sem = nom.get("semester") or matched_master.get("year") or "I"
        gender = u_info.get("gender", "")
        father = to_title_case(nom.get("father_name") or u_info.get("father_name") or "")

        # Aliases for search (10x search weight)
        aliases: List[str] = [uid]
        if roll_no:
            aliases.append(roll_no)
        if display_name:
            aliases.extend([display_name, display_name.upper(), display_name.lower()])

        # Keywords for search (5x search weight)
        keywords: List[str] = ["student", "kiet", "btech", "b.tech", branch]
        if sec:
            keywords.extend([sec, f"section {sec}"])
        if mentor:
            keywords.extend([f"mentor {mentor}", mentor])
        if sem:
            keywords.append(f"sem {sem}")

        # Voice-friendly spoken summary
        summary_parts = [f"{display_name} is a B.Tech student in {branch}"]
        if sec:
            summary_parts.append(f"section {sec}")
        if mentor:
            summary_parts.append(f"mentored by {mentor}")
        summary_parts.append("at K.I.E.T. Deemed to be University, Ghaziabad.")
        summary = " ".join(summary_parts)

        # Structured detailed content (public non-contact profile facts only)
        content_lines = [
            f"Student Name: {display_name}",
            f"UID: {uid}",
        ]
        if roll_no:
            content_lines.append(f"University Roll Number: {roll_no}")
        content_lines.append("Degree: Bachelor of Technology (B.Tech)")
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

    # 2. Process all remaining students from B.Tech 1st Year Master List
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

        branch = mr.get("branch", "Engineering")
        sec = mr.get("section", "")
        batch = mr.get("academic_batch", "2025-2029")
        degree = mr.get("degree", "Bachelor of Technology")
        year = mr.get("year", "I")
        email = mr.get("email", "")
        category = mr.get("category", "")
        status = mr.get("admission_status", "ACTIVE")

        phone = mr.get("phone", "")
        adm_date = mr.get("date_of_admission", "")
        adm_cat = mr.get("admission_category", "")

        aliases = [roll_no, display_name, display_name.upper(), display_name.lower()]

        keywords = ["student", "kiet", "btech", "b.tech", branch]
        if sec:
            keywords.extend([sec, f"section {sec}"])
        if batch:
            keywords.append(batch)

        summary = (
            f"{display_name} is a 1st-year {degree} student in {branch} "
            f"(Section {sec}, Batch {batch}) at K.I.E.T. Deemed to be University, Ghaziabad."
        )

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


def run_ingestion(
    source_dir: Path,
    dry_run: bool = False,
    limit: Optional[int] = None,
    batch_size: int = 100,
    backend: str = "qdrant",
) -> int:
    """Coordinates reading datasets, building documents, and upserting into Qdrant."""
    load_env()

    # File paths resolution with automatic fallbacks
    if not (source_dir / "UID.xlsx").exists():
        candidates = [
            source_dir,
            Path(__file__).resolve().parent.parent.parent / "local_data",
            Path.cwd() / "local_data",
            Path.cwd() / "rag_knowledge" / "local_data",
        ]
        for c in candidates:
            if (c / "UID.xlsx").exists():
                source_dir = c
                break

    uid_path = source_dir / "UID.xlsx"
    nom_path = source_dir / "Nominal Roll List University 2026-27 Odd sem.xlsx"
    master_path = source_dir / "STUDENT LIST B.TECH I YEAR UPDATED.xlsx"

    # 1. Parse Excel files
    uid_records = load_uid_mapping(uid_path) if uid_path.exists() else {}
    nom_records = load_nominal_roll(nom_path) if nom_path.exists() else {}
    master_records = load_student_list(master_path) if master_path.exists() else []

    if not uid_records and not nom_records and not master_records:
        logger.error(f"No valid Excel datasets found in {source_dir}")
        return 0

    # 2. Build unified student documents
    documents = build_unified_student_documents(nom_records, uid_records, master_records)

    # Enforce strict privacy guard before dry-run or upsert
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
            print(f"Aliases:  {', '.join(d.get('aliases', [])[:6])}")
            print(f"Keywords: {', '.join(d.get('keywords', [])[:6])}")
            print(f"Summary:  {d['summary']}")
            print(f"Content Preview:\n{d.get('content', '')[:250]}...")
            print()
        print(f"[DRY-RUN COMPLETE] Zero changes were made to Qdrant.")
        return len(documents)

    # 3. Upsert into Qdrant Cloud
    store = get_global_qdrant_store()
    store.timeout = 30.0
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
        default="data/raw",
        help="Path to folder containing .xlsx student datasets (default: data/raw)",
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
        help="Limit number of documents to ingest (useful for testing)",
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

    if raw_path.is_dir():
        source_dir = raw_path
    elif (package_dir / args.source).is_dir():
        source_dir = package_dir / args.source
    elif (package_dir.parent / args.source).is_dir():
        source_dir = package_dir.parent / args.source
    else:
        source_dir = package_dir / args.source

    total = run_ingestion(
        source_dir=source_dir,
        dry_run=args.dry_run,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    if not args.dry_run and total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
