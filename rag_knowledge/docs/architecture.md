# RAG Knowledge Architecture

This document describes the internal design of the `rag_knowledge` subsystem.

```
                    +-----------------------+
                    |      User Query       |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    |    RAGService         |
                    | (service.py)          |
                    +-----------+-----------+
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
  +-----------------------+             +-----------------------+
  |  KnowledgeRetriever   |             |    GeminiRAGClient    |
  |  (retriever.py)       |             |  (gemini_client.py)   |
  +-----------+-----------+             +-----------+-----------+
              |                                     |
              v                                     v
  +-----------------------+             +-----------------------+
  | Qdrant Vector Store   |             |  Google Gemini API    |
  | (qdrant_storage.py)   |             | (generativelanguage)  |
  +-----------------------+             +-----------------------+
```

## Components

### 1. `KnowledgeRetriever` (`retrieval/retriever.py`)
- Vector-backed semantic retrieval engine querying Qdrant Cloud or local Qdrant collection.
- Dense embeddings generated with FastEmbed (`BAAI/bge-small-en-v1.5`), paired with exact alias/roll/UID scoring.
- Hybrid token and exact match ranking with reciprocal rank fusion (RRF).
- Returns top-k matching documents ranked by relevance score.

### 2. `GeminiRAGClient` (`clients/gemini_client.py`)
- Communicates directly with Google Generative Language REST API (`models/{model}:generateContent`).
- Uses standard library `urllib.request` inside an async worker executor to ensure zero event loop blocking and zero third-party dependencies.
- Injects a voice-tuned prompt directing Gemini to answer within 2–3 spoken sentences strictly grounded in retrieved facts.
- Fail-safe: if request fails or key is missing, returns `None` so the caller gracefully falls back.

### 3. `RAGService` (`service.py`)
- Coordinates the retrieval and generation workflow.
- Falls back gracefully to raw document content if Gemini is unconfigured or encounters an error.
