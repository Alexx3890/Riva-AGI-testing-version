# RAG Knowledge Subsystem (`rag_knowledge`)

Voice-optimized Retrieval-Augmented Generation (RAG) service powered by **Qdrant Vector Database** and **Google Gemini**.

---

## 1. Setup

Set your credentials in `.env` (see `.env.example`):

```ini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_VISION_API_KEY=your_gemini_vision_api_key  # Optional dedicated key for vision OCR
QDRANT_URL=https://your-cluster-id.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=riva_knowledge
```

Install requirements:
```bash
pip install -r rag_knowledge/requirements.txt
```

---

## 2. Ingestion (Upload Data)

Ingest files or directories with universal format detection:

```bash
# Ingest with privacy redaction (recommended)
python -m rag_knowledge.ingestion --source "path/to/file_or_folder" --redact

# Preview sample documents without writing to cloud
python -m rag_knowledge.ingestion --source "path/to/file_or_folder" --dry-run

# Wipe the cloud database clean
python -m rag_knowledge.ingestion --clear
```

**Supported Formats:**
- **Spreadsheets**: `.xlsx`, `.xls`, `.csv`, `.tsv` (automatic rosters & category breakdowns).
- **Documents**: `.pdf`, `.docx`, `.md`, `.txt`, `.json`.
- **Images**: `.png`, `.jpg`, `.jpeg`, `.webp` (multimodal vision extraction for questions, formulas, and diagrams).

---

## 3. Querying

### From Terminal
```bash
python -m rag_knowledge "What is the probability of rolling an even number?"
```

### In Python Code
```python
import asyncio
from rag_knowledge import query_rag

async def main():
    answer = await query_rag("When is the next workshop?")
    print(answer)

asyncio.run(main())
```

---

## 4. Tests

```bash
pytest rag_knowledge/tests
```
