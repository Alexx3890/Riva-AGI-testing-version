"""Ingestion subpackage for processing and loading institutional datasets."""

from .ingest import (
    assert_no_private_in_embedded_fields,
    build_unified_student_documents,
    main,
    run_ingestion,
)

__all__ = [
    "run_ingestion",
    "build_unified_student_documents",
    "assert_no_private_in_embedded_fields",
    "main",
]
