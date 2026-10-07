"""Unit tests for student dataset unification, privacy guards, and namesake filtering in ingest.py."""

import pytest
from rag_knowledge.ingestion.ingest import (
    assert_no_private_in_embedded_fields,
    build_unified_student_documents,
    get_active_sheet,
    load_uid_mapping,
    load_nominal_roll,
    load_student_list,
)


def test_build_documents_and_privacy_guard_clean():
    """Builds documents from tiny in-memory records, runs guard, and checks privacy separation."""
    nominal_records = {
        "2630BTECH001": {
            "uid": "2630BTECH001",
            "name": "Aarav Sharma",
            "semester": "I",
            "section": "A",
            "phone": "9876543210",
            "email": "aarav.sharma@kiet.edu",
            "father_name": "Rajesh Sharma",
            "mentor": "Dr. Sunita Rao",
        }
    }
    uid_records = {
        "2630BTECH001": {
            "uid": "2630BTECH001",
            "name": "Aarav Sharma",
            "father_name": "Rajesh Sharma",
            "gender": "MALE",
            "batch_name": "CSIT",
        }
    }
    master_records = [
        {
            "roll_number": "26300101001",
            "name": "Aarav Sharma",
            "degree": "B.Tech",
            "branch": "Computer Science and Information Technology",
            "academic_batch": "2025-2029",
            "year": "I",
            "section": "A",
            "admission_status": "ACTIVE",
            "email": "aarav.sharma@kiet.edu",
            "phone": "9876543210",
            "category": "GEN",
            "date_of_admission": "2025-08-01",
            "admission_category": "Direct",
        },
        {
            "roll_number": "26300101002",
            "name": "Priya Verma",
            "degree": "B.Tech",
            "branch": "Electronics and Communication Engineering",
            "academic_batch": "2025-2029",
            "year": "I",
            "section": "B",
            "admission_status": "ACTIVE",
            "email": "priya.verma@kiet.edu",
            "phone": "9876543211",
            "category": "OBC",
            "date_of_admission": "2025-08-02",
            "admission_category": "JEE",
        },
    ]

    docs = build_unified_student_documents(nominal_records, uid_records, master_records)
    assert len(docs) == 2

    # 1. Guard passes with zero errors on freshly built documents
    assert_no_private_in_embedded_fields(docs)

    # 2. Check nominal roll built document
    nom_doc = next(d for d in docs if d["id"] == "student_2630btech001")
    assert nom_doc["metadata"]["private"]["email"] == "aarav.sharma@kiet.edu"
    assert nom_doc["metadata"]["private"]["phone"] == "9876543210"
    assert nom_doc["metadata"]["private"]["gender"] == "MALE"
    assert nom_doc["metadata"]["private"]["father_name"] == "Rajesh Sharma"

    # Verify no '@' character in embedded fields
    for field in ["content", "summary"]:
        assert "@" not in nom_doc[field], f"Found '@' in {field}: {nom_doc[field]}"
    for alias in nom_doc["aliases"]:
        assert "@" not in alias, f"Found '@' in alias: {alias}"

    # 3. Check master list built document
    master_doc = next(d for d in docs if d["id"] == "student_26300101002")
    assert master_doc["metadata"]["private"]["email"] == "priya.verma@kiet.edu"
    assert master_doc["metadata"]["private"]["phone"] == "9876543211"
    assert master_doc["metadata"]["private"]["category"] == "OBC"
    assert master_doc["metadata"]["private"]["date_of_admission"] == "2025-08-02"
    assert master_doc["metadata"]["private"]["admission_category"] == "JEE"

    for field in ["content", "summary"]:
        assert "@" not in master_doc[field], f"Found '@' in {field}: {master_doc[field]}"
    for alias in master_doc["aliases"]:
        assert "@" not in alias, f"Found '@' in alias: {alias}"


def test_assert_no_private_in_embedded_fields_catches_leak():
    """Guard raises ValueError if email or private info leaks into content, summary, or aliases."""
    leaky_doc = {
        "id": "student_test_leaky",
        "title": "Test Student",
        "summary": "Student summary",
        "content": "Official Email: leak@example.com",
        "aliases": ["TEST101"],
        "keywords": ["student"],
    }
    with pytest.raises(ValueError, match="Private data found in embedded fields"):
        assert_no_private_in_embedded_fields([leaky_doc])

    leaky_alias_doc = {
        "id": "student_test_alias_leak",
        "title": "Test Student",
        "summary": "Student summary",
        "content": "Clean profile content",
        "aliases": ["leak@example.com"],
        "keywords": ["student"],
    }
    with pytest.raises(ValueError, match="Private data found in embedded fields"):
        assert_no_private_in_embedded_fields([leaky_alias_doc])


def test_namesake_ambiguity_skips_non_unique_names():
    """Verifies that common names shared across multiple master records are skipped during name-only fallback."""
    nominal_records = {
        "2630BTECH999": {
            "uid": "2630BTECH999",
            "name": "Kishan Singh",
            "semester": "I",
            "section": "C",
            "phone": "9999999999",
            "email": "different_unmatched_email@kiet.edu",
            "father_name": "Father Singh",
            "mentor": "Mentor",
        }
    }
    uid_records = {}
    master_records = [
        {
            "roll_number": "26300101091",
            "name": "Kishan Singh",
            "degree": "B.Tech",
            "branch": "CSE",
            "email": "kishan1@kiet.edu",
        },
        {
            "roll_number": "26300101092",
            "name": "Kishan Singh",
            "degree": "B.Tech",
            "branch": "ECE",
            "email": "kishan2@kiet.edu",
        },
    ]

    docs = build_unified_student_documents(nominal_records, uid_records, master_records)
    nom_doc = next(d for d in docs if d["id"] == "student_2630btech999")

    # Because "Kishan Singh" appears twice in master records and email did not match,
    # the nominal builder must NOT associate ambiguous roll_number or branch to this student.
    assert nom_doc["metadata"]["roll_number"] == "", "Ambiguous namesake should not bind roll_number"
    assert nom_doc["metadata"]["branch"] == "Computer Science and Information Technology"


def test_get_active_sheet_fallback():
    """Verifies that get_active_sheet falls back to worksheets[0] when wb.active is None."""
    import openpyxl

    wb = openpyxl.Workbook()
    # In openpyxl, if active tab index is missing or out-of-bounds, wb.active returns None
    setattr(wb, "_active_sheet_index", 999)
    assert wb.active is None

    sheet = get_active_sheet(wb)
    assert sheet is not None
    assert sheet == wb.worksheets[0]


def test_excel_loaders_handle_none_sheet(tmp_path, monkeypatch):
    """Ensures load_uid_mapping, load_nominal_roll, and load_student_list do not crash when sheet is None."""
    import openpyxl

    dummy_xlsx = tmp_path / "dummy.xlsx"
    wb = openpyxl.Workbook()
    wb.save(dummy_xlsx)
    wb.close()

    # Force get_active_sheet to return None as if the file has no worksheets
    monkeypatch.setattr("rag_knowledge.ingestion.ingest.get_active_sheet", lambda _wb: None)

    uid_res = load_uid_mapping(dummy_xlsx)
    assert uid_res == {}

    nom_res = load_nominal_roll(dummy_xlsx)
    assert nom_res == {}

    master_res = load_student_list(dummy_xlsx)
    assert master_res == []

