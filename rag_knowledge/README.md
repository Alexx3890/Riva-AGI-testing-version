# RAG Knowledge Subsystem (`rag_knowledge`)

A modular Retrieval-Augmented Generation (RAG) service providing fast vector lookup via Qdrant and conversational synthesis with Google Gemini.

---

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install -r rag_knowledge/requirements.txt
   ```

2. **Configure environment:**
   Copy `.env.example` to `.env` and set your credentials:
   ```ini
   GEMINI_API_KEY=your_gemini_api_key
   QDRANT_URL=https://your-cluster.qdrant.io
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_COLLECTION=riva_knowledge
   ```

---

## Usage

### In Python
```python
import asyncio
from rag_knowledge import query_rag

async def main():
    answer = await query_rag("When is the next workshop scheduled?")
    print(answer)

asyncio.run(main())
```

### From CLI
```bash
# Query knowledge base
python -m rag_knowledge "When is the next workshop scheduled?"

# List registered documents
python -m rag_knowledge --list
```

### Data Ingestion
Upsert documents from files or directories (`.xlsx`, `.json`, `.csv`, `.md`, `.txt`):
```bash
# Dry run preview (auto-detects format)
python -m rag_knowledge.ingestion.ingest --source data/raw/ --dry-run

# Upsert specific files or folder
python -m rag_knowledge.ingestion.ingest --source data/raw/faq.json
python -m rag_knowledge.ingestion.ingest --source data/raw/handbook.md
python -m rag_knowledge.ingestion.ingest --source data/raw/students/ --type student
```

---

## Running Tests

Execute the hermetic unit test suite:
```bash
pytest rag_knowledge/tests -v
```
