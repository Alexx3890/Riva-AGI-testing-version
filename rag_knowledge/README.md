# RAG Knowledge Subsystem (`rag_knowledge`)

A fast, self-contained Retrieval-Augmented Generation (RAG) package built for Riva. It gives conversational assistants and voice agents instant access to factual knowledge (students, mentors, departments, and campus info) with natural spoken responses powered by Google Gemini.

---

## How It Works (At a Glance)

```text
User Question ("Who is Nikhil Kumar?")
               │
               ▼
┌──────────────────────────────────────────────┐
│  1. Hybrid Retrieval                         │
│     • FastEmbed (local 384-d dense vector)   │
│     • Exact ID / Roll No boost (score 100.0) │
│     • Qdrant Cloud vector search             │
└──────────────────────┬───────────────────────┘
                       │ Matched Documents
                       ▼
┌──────────────────────────────────────────────┐
│  2. Gemini Synthesis (gemini-flash-lite)     │
│     • Concise 2–3 sentence spoken answer     │
│     • Direct factual fallback if key unset   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
Spoken Answer / Structured Facts to User
```

---

## Quick Setup

### 1. Configure Environment Variables
Copy `.env.example` to `.env` in your project root or `rag_knowledge/`:

```bash
cp rag_knowledge/.env.example .env
```

Fill in your credentials:
```ini
# Google Gemini (Optional: system returns facts directly if omitted)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest
GEMINI_TIMEOUT=4.0

# Qdrant Cloud (Vector Database)
QDRANT_URL=https://your-cluster-id.us-east-1-1.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_WRITE_API_KEY=your_qdrant_write_api_key_here
QDRANT_COLLECTION=riva_knowledge

# Embedding Model (Optional, defaults to BAAI/bge-small-en-v1.5)
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### 2. Install Dependencies
```bash
pip install -r rag_knowledge/requirements.txt
```

### 3. Pre-warm Embedding Model (Optional)
The system uses `BAAI/bge-small-en-v1.5` via FastEmbed ONNX. On the very first run, it downloads the model (~130 MB) to `~/.cache/fastembed/`. You can pre-cache it ahead of time:
```bash
python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"
```

---

## Project Structure

The package is flat and self-contained under `rag_knowledge/`:

```text
rag_knowledge/
├── __init__.py           # Public exports (query_rag, get_rag_service, etc.)
├── __main__.py           # CLI runner (python -m rag_knowledge)
├── cli.py                # Command-line interface logic
├── prompts.py            # Prompt templates, versioning & input sanitization
├── service.py            # High-level RAG service (retrieval + synthesis)
├── clients/
│   ├── __init__.py
│   └── gemini_client.py  # Zero-dependency Gemini caller via urllib
├── ingestion/
│   ├── __init__.py
│   └── ingest.py         # Excel data ingestion & privacy guard pipeline
├── retrieval/
│   ├── __init__.py
│   └── retriever.py      # Knowledge retrieval & context string formatting
├── storage/
│   ├── __init__.py
│   └── qdrant_storage.py # Qdrant Cloud client, FastEmbed vectors & hybrid search
├── data/
│   └── raw/              # Raw spreadsheets (.xlsx) — gitignored
├── tests/                # Hermetic unit test suite (36 tests)
│   ├── conftest.py
│   ├── test_gemini_client.py
│   ├── test_ingest.py
│   ├── test_prompts.py
│   ├── test_qdrant_storage.py
│   ├── test_retriever.py
│   └── test_service.py
├── docs/                 # Documentation notes
├── .env.example          # Environment variable template
├── .gitignore            # Ignores secrets, cache, and data/*
├── README.md             # This guide
├── requirements.txt      # Production runtime dependencies
└── requirements-dev.txt  # Development & test dependencies
```

---

## How to Use

### 1. From the Command Line (CLI)

Ask a natural language question:
```bash
python -m rag_knowledge "Tell me about Nikhil Kumar"
```

Look up by student UID or university roll number:
```bash
python -m rag_knowledge "2630BTECH0121"
```

Inspect match candidates and similarity scores (verbose mode):
```bash
python -m rag_knowledge "Who is mentored by Dr. Meeta Chaudhry?" -v
```

List stored documents:
```bash
python -m rag_knowledge --list
```

### 2. In Python Code

```python
import asyncio
from rag_knowledge import query_rag

async def main():
    # Ask a question and get a conversational answer
    answer = await query_rag("Tell me about Nikhil Kumar")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
```

Directly access the Qdrant store:
```python
from rag_knowledge.storage import get_global_qdrant_store

store = get_global_qdrant_store()

# Search knowledge records
results = store.search("computer science mentors", limit=3)
for doc, score in results:
    print(f"[{score:.2f}] {doc['title']}: {doc['summary']}")
```

### 3. Ingesting Institutional Data

Place your Excel spreadsheets (`.xlsx`) in `rag_knowledge/data/raw/`:
- Nominal roll lists
- UID mappings
- Student master lists

Preview ingestion without writing to the database (dry run):
```bash
python -m rag_knowledge.ingestion.ingest --dry-run
python -m rag_knowledge.ingestion.ingest --dry-run --limit 5
```

Perform live upsert to Qdrant Cloud:
```bash
python -m rag_knowledge.ingestion.ingest
```

---

## Key Features Explained

### 1. Hybrid Search (Semantic + Exact Match)
Queries undergo a two-phase lookup:
1. **Semantic Vector Search**: Uses FastEmbed's `BAAI/bge-small-en-v1.5` (384 dimensions) to find relevant records based on meaning (`score >= 0.45`).
2. **Exact Identifier Boosting**: Detects roll numbers or IDs (e.g., `2630BTECH0121`) and boosts exact matches to a score of `100.0`, ensuring 100% precision for ID lookups.
3. **Runner-Up Gap Guard**: If an exact match is found, weaker semantic runner-ups are suppressed to avoid confusing the conversational synthesizer.

### 2. Built-in Privacy Guard
Before records are embedded and stored in Qdrant:
- Highly sensitive fields (such as Father's Name, personal gender markers, and admission remarks) are stripped out of the public vector embedding payload.
- Namesake ambiguity protection automatically drops name-only lookups for duplicate names to avoid confusing two students with the same name.
- Raw spreadsheet files in `rag_knowledge/data/raw/` are strictly gitignored and never committed.

### 3. Dual Resilient Fallbacks
- **Missing or Rate-Limited Gemini**: If `GEMINI_API_KEY` is not provided or Gemini hits a quota, the system falls back to returning the verified factual summary directly. Conversations never crash.
- **Database Offline**: If Qdrant Cloud is unreachable, a friendly notification (`"The knowledge database is currently unavailable. Please try again shortly."`) is returned with no uncaught exceptions.

---

## Running Tests

The test suite is **100% hermetic**: tests mock all external network calls and Qdrant connections, requiring no live API keys or cloud credentials.

Run all tests:
```bash
pytest rag_knowledge/tests/ -v
```

Or using `uv`:
```bash
uv run pytest rag_knowledge/tests/ -v
```

All 36 unit tests should pass with zero configuration.

---

## Troubleshooting

| Issue | What Happened | How to Fix |
| :--- | :--- | :--- |
| **"The knowledge database is currently unavailable"** | Could not connect to Qdrant Cloud. | Check `QDRANT_URL` and `QDRANT_API_KEY` in your `.env` file. Verify internet connectivity. |
| **Direct summary returned instead of conversational answer** | `GEMINI_API_KEY` is missing or invalid. | Add a valid key from [Google AI Studio](https://aistudio.google.com/) to your `.env`. |
| **First run takes a few seconds** | FastEmbed is downloading the ONNX embedding model (~130 MB). | Allow it to complete once. Subsequent runs load instantly from local cache. |
| **"No valid Excel datasets found" during ingestion** | Raw `.xlsx` files are missing from `data/raw/`. | Place your spreadsheets in `rag_knowledge/data/raw/` or pass `--source <path>`. |
