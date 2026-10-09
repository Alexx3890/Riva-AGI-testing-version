"""Word Document (.docx) Reader.

Extracts text hierarchically based on Headings (H1-H3), preserving
lists and parsing tables into individual structured row units.
Records source file, section, and table metadata for provenance.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rag.ingest.readers.word")


class WordDocxReader:
    """Parses Word .docx files into structured raw document units."""

    def can_read(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".docx"

    def read(self, file_path: Path) -> List[Dict[str, Any]]:
        """Reads a Word .docx file into structured section and table units."""
        if not file_path.is_file():
            return []

        try:
            import docx
        except ImportError:
            logger.error("python-docx is not installed. Run `pip install python-docx`.")
            return []

        try:
            doc = docx.Document(str(file_path))
        except Exception as e:
            logger.error(f"Failed to open Word file {file_path.name}: {e}")
            raise

        units: List[Dict[str, Any]] = []
        current_section = file_path.stem
        current_paragraphs: List[str] = []
        section_idx = 1

        def _flush_section():
            nonlocal section_idx, current_paragraphs
            if current_paragraphs:
                content = "\n\n".join(current_paragraphs).strip()
                if content:
                    units.append({
                        "unit_type": "prose",
                        "title": f"{current_section} (Section {section_idx})",
                        "section": current_section,
                        "content": content,
                        "provenance": {
                            "source_file": file_path.name,
                            "section": current_section,
                            "section_index": section_idx,
                        },
                    })
                    section_idx += 1
                current_paragraphs = []

        # 1. Process paragraphs
        for p in doc.paragraphs:
            text = (p.text or "").strip()
            if not text:
                continue

            style_name = (p.style.name if p.style else "").lower()
            if "heading" in style_name:
                _flush_section()
                current_section = text
            else:
                current_paragraphs.append(text)

        _flush_section()

        # 2. Process tables
        for t_idx, table in enumerate(doc.tables, 1):
            if not table.rows:
                continue

            # First row as header
            headers = [c.text.strip() for c in table.rows[0].cells]
            table_rows = []
            for row in table.rows[1:]:
                row_vals = [c.text.strip() for c in row.cells]
                if any(row_vals):
                    row_dict = {
                        (headers[i] if i < len(headers) and headers[i] else f"Col_{i+1}"): val
                        for i, val in enumerate(row_vals)
                    }
                    table_rows.append(row_dict)

            if table_rows:
                table_title = f"{file_path.stem} - Table {t_idx}"
                rows_text = "\n".join(
                    " | ".join(f"{k}: {v}" for k, v in r.items() if v) for r in table_rows
                )
                units.append({
                    "unit_type": "table",
                    "title": table_title,
                    "section": current_section,
                    "content": f"### {table_title}\n\n" + rows_text,
                    "provenance": {
                        "source_file": file_path.name,
                        "table_index": t_idx,
                        "row_count": len(table_rows),
                    },
                })

        logger.info(f"WordDocxReader extracted {len(units)} units from {file_path.name}")
        return units
