"""Pytest configuration and hermetic isolation fixtures for rag_knowledge."""

import os
import pytest
from unittest.mock import MagicMock

import rag_knowledge.storage.qdrant_storage as qdrant_mod
import rag_knowledge.service as service_mod


@pytest.fixture
def anyio_backend():
    """Pins anyio to asyncio for all async test cases."""
    return "asyncio"


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """Enforces hermetic test isolation by clearing real external credentials from environment."""
    # Prevent tests from discovering live Qdrant or Gemini credentials
    monkeypatch.setenv("qdrant_cluster_endpoint", "")
    monkeypatch.delenv("qdrant_cluster_endpoint", raising=False)
    monkeypatch.setenv("qdrant_api", "")
    monkeypatch.delenv("qdrant_api", raising=False)
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("QDRANT_API_KEY", "")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("QDRANT_WRITE_API_KEY", "")
    monkeypatch.delenv("QDRANT_WRITE_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_RAG_MODEL", raising=False)
    monkeypatch.setenv("RAG_LOAD_CWD_ENV", "false")
    monkeypatch.setenv("RAG_DISABLE_LOAD_ENV", "true")

    # Reset any cached global stores or services
    qdrant_mod._global_qdrant_store = None
    service_mod._default_service = None

    yield

    # Clean up afterwards
    qdrant_mod._global_qdrant_store = None
    service_mod._default_service = None
