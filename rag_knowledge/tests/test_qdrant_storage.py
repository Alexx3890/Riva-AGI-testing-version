"""Unit tests for Qdrant vector storage backend, alignment, filtering, and cooldown."""

import time
import uuid
from unittest.mock import MagicMock, patch
import pytest

from rag_knowledge.storage.qdrant_storage import QdrantKnowledgeStore, string_to_point_id


def test_string_to_point_id():
    """Verifies deterministic UUID generation from string document IDs."""
    pid1 = string_to_point_id("student_2630btech2206")
    pid2 = string_to_point_id("student_2630btech2206")
    pid3 = string_to_point_id("student_2025r0111101103")

    assert pid1 == pid2
    assert pid1 != pid3
    assert uuid.UUID(pid1)


def test_qdrant_store_not_configured():
    """QdrantKnowledgeStore returns is_available() False when URL/key is unset."""
    store = QdrantKnowledgeStore(url="", api_key="")
    assert store.is_available() is False
    assert store.search_text("alex-doe") == []
    assert store.list_documents() == []


def test_qdrant_store_mocked_success():
    """Verifies collection creation, embedding, upsert, and search with mocked QdrantClient."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])

    mock_hit = MagicMock()
    mock_hit.id = "test-uuid"
    mock_hit.score = 0.85
    mock_hit.payload = {
        "id": "alex_doe",
        "title": "Alex Doe - AI Specialist",
        "summary": "Specialist in machine learning",
        "content": "TYPE: PERSON\nName: Alex Doe",
        "is_active": True,
    }
    mock_client.query_points.return_value = MagicMock(points=[mock_hit])
    mock_client.scroll.return_value = ([], None)

    with patch("rag_knowledge.storage.qdrant_storage.QdrantClient", return_value=mock_client), \
         patch("rag_knowledge.storage.qdrant_storage.TextEmbedding") as mock_embed_cls:

        mock_model = MagicMock()
        mock_model.embed.return_value = [[0.1] * 384]
        mock_embed_cls.return_value = mock_model

        store = QdrantKnowledgeStore(url="https://fake.qdrant.io", api_key="fake-key")
        assert store.connect() is True
        assert store.is_available() is True

        # Test upsert
        count = store.upsert_documents([{"id": "alex_doe", "title": "Alex Doe"}])
        assert count == 1
        assert mock_client.upsert.called

        # Test search
        results = store.search_text("machine learning")
        assert len(results) == 1
        assert results[0]["id"] == "alex_doe"
        assert results[0]["score"] == 85.0


def test_upsert_vector_alignment_skips_invalid_ids():
    """Verifies Bug 1 fix: documents missing an ID are filtered before embedding to prevent vector misalignment."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])

    with patch("rag_knowledge.storage.qdrant_storage.QdrantClient", return_value=mock_client), \
         patch("rag_knowledge.storage.qdrant_storage.TextEmbedding") as mock_embed_cls:

        mock_model = MagicMock()
        # Embed receives 2 texts (doc1 and doc2, skipping doc_invalid)
        mock_model.embed.return_value = [
            [0.1] * 384,  # vector for doc1
            [0.2] * 384,  # vector for doc2
        ]
        mock_embed_cls.return_value = mock_model

        store = QdrantKnowledgeStore(url="https://fake.qdrant.io", api_key="fake-key")
        store.connect()

        docs_input = [
            {"id": "doc_1", "title": "Valid Doc 1", "content": "Text 1"},
            {"id": "", "title": "Invalid Empty ID", "content": "Text Invalid"},
            {"id": "doc_2", "title": "Valid Doc 2", "content": "Text 2"},
        ]
        count = store.upsert_documents(docs_input)
        assert count == 2
        assert mock_client.upsert.called
        points_upserted = mock_client.upsert.call_args[1]["points"]
        assert len(points_upserted) == 2
        assert points_upserted[0].payload["id"] == "doc_1"
        assert points_upserted[1].payload["id"] == "doc_2"


def test_search_and_list_filter_active_documents_only():
    """Verifies Bug 2 fix: inactive documents (is_active=False) are never returned in search or list."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_client.query_points.return_value = MagicMock(points=[])
    mock_client.scroll.return_value = ([], None)

    with patch("rag_knowledge.storage.qdrant_storage.QdrantClient", return_value=mock_client), \
         patch("rag_knowledge.storage.qdrant_storage.TextEmbedding") as mock_embed_cls:

        mock_model = MagicMock()
        mock_model.embed.return_value = [[0.1] * 384]
        mock_embed_cls.return_value = mock_model

        store = QdrantKnowledgeStore(url="https://fake.qdrant.io", api_key="fake-key")
        store.connect()

        # 1. Search text verifies query_filter contains is_active=True
        store.search_text("machine learning")
        qp_call = mock_client.query_points.call_args[1]
        q_filter = qp_call["query_filter"]
        assert q_filter is not None
        assert any(cond.key == "is_active" for cond in q_filter.must)

        # 2. list_documents verifies scroll_filter contains is_active=True
        store.list_documents(limit=50)
        scroll_call = mock_client.scroll.call_args[1]
        s_filter = scroll_call["scroll_filter"]
        assert s_filter is not None
        assert any(cond.key == "is_active" for cond in s_filter.must)


def test_identifier_detection_triggers_scroll_selectively():
    """Verifies Production Issue 3: only queries containing alphanumeric IDs with digits trigger exact scroll."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_client.query_points.return_value = MagicMock(points=[])
    mock_client.scroll.return_value = ([], None)

    with patch("rag_knowledge.storage.qdrant_storage.QdrantClient", return_value=mock_client), \
         patch("rag_knowledge.storage.qdrant_storage.TextEmbedding") as mock_embed_cls:

        mock_model = MagicMock()
        mock_model.embed.return_value = [[0.1] * 384]
        mock_embed_cls.return_value = mock_model

        store = QdrantKnowledgeStore(url="https://fake.qdrant.io", api_key="fake-key")
        store.connect()

        # Natural language query with words >= 4 chars without digits (no ID)
        mock_client.scroll.reset_mock()
        store.search_text("Tell me about Nikhil Kumar")
        assert not mock_client.scroll.called, "Scroll should NOT be triggered for queries without identifier tokens"

        # Query with roll number identifier (has digits, >= 6 chars)
        mock_client.scroll.reset_mock()
        store.search_text("Tell me about 2630BTECH2206")
        assert mock_client.scroll.called, "Scroll SHOULD be triggered when query contains identifier token"
        scroll_filter = mock_client.scroll.call_args[1]["scroll_filter"]
        assert scroll_filter is not None


def test_connection_cooldown_protection():
    """Verifies Production Issue 4: failed connection enters cooldown and prevents rapid retry loop."""
    mock_client_cls = MagicMock()
    mock_client_cls.side_effect = RuntimeError("Connection timed out")

    with patch("rag_knowledge.storage.qdrant_storage.QdrantClient", mock_client_cls):
        store = QdrantKnowledgeStore(url="https://fake.qdrant.io", api_key="fake-key")
        # First attempt fails
        assert store.is_available() is False
        assert mock_client_cls.call_count == 1

        # Second immediate attempt should be throttled by cooldown without calling QdrantClient again
        assert store.is_available() is False
        assert mock_client_cls.call_count == 1

        # Simulate cooldown expiration
        store._retry_after = time.monotonic() - 1.0
        assert store.is_available() is False
        assert mock_client_cls.call_count == 2


def test_collection_dimension_mismatch_raises_value_error():
    """Verifies dimension check: raises ValueError if existing collection dimension differs from model vector size."""
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.name = "riva_knowledge"
    mock_client.get_collections.return_value = MagicMock(collections=[mock_col])

    # Existing collection has 768 dimensions instead of expected 384
    mock_info = MagicMock()
    mock_info.config.params.vectors.size = 768
    mock_client.get_collection.return_value = mock_info

    with patch("rag_knowledge.storage.qdrant_storage.QdrantClient", return_value=mock_client), \
         patch("rag_knowledge.storage.qdrant_storage.TextEmbedding"):

        store = QdrantKnowledgeStore(url="https://fake.qdrant.io", api_key="fake-key")
        with pytest.raises(ValueError, match="vector dimension is 768, but model .* requires 384"):
            store.connect()


def test_private_metadata_stripped_in_search_and_list():
    """Verifies that private fields are stripped in search_text and list_documents."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])

    mock_hit = MagicMock()
    mock_hit.id = "test-uuid"
    mock_hit.score = 0.90
    mock_hit.payload = {
        "id": "student_123",
        "title": "Jane Doe",
        "summary": "Engineering student",
        "content": "Student profile",
        "metadata": {
            "name": "Jane Doe",
            "roll_number": "123456",
            "private": {
                "email": "jane@example.com",
                "phone": "555-1234",
            },
        },
        "is_active": True,
    }
    mock_client.query_points.return_value = MagicMock(points=[mock_hit])
    mock_client.scroll.return_value = ([mock_hit], None)

    with patch("rag_knowledge.storage.qdrant_storage.QdrantClient", return_value=mock_client), \
         patch("rag_knowledge.storage.qdrant_storage.TextEmbedding") as mock_embed_cls:

        mock_model = MagicMock()
        mock_model.embed.return_value = [[0.1] * 384]
        mock_embed_cls.return_value = mock_model

        store = QdrantKnowledgeStore(url="https://fake.qdrant.io", api_key="fake-key")
        store.connect()

        # 1. Search text strips 'private' from metadata
        search_res = store.search_text("Jane Doe")
        assert len(search_res) == 1
        assert "private" not in search_res[0]["metadata"]
        assert search_res[0]["metadata"]["name"] == "Jane Doe"

        # 2. list_documents strips 'private' from metadata
        list_res = store.list_documents()
        assert len(list_res) == 1
        assert "private" not in list_res[0]["metadata"]
        assert list_res[0]["metadata"]["name"] == "Jane Doe"


def test_get_document_include_private_toggle():
    """Verifies that get_document strips private fields by default, and includes them when requested."""
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])

    mock_point = MagicMock()
    mock_point.id = "point-uuid"
    mock_point.payload = {
        "id": "student_123",
        "title": "Jane Doe",
        "summary": "Engineering student",
        "content": "Student profile",
        "metadata": {
            "name": "Jane Doe",
            "private": {
                "email": "jane@example.com",
                "phone": "555-1234",
            },
        },
        "is_active": True,
    }
    mock_client.retrieve.return_value = [mock_point]

    with patch("rag_knowledge.storage.qdrant_storage.QdrantClient", return_value=mock_client):
        store = QdrantKnowledgeStore(url="https://fake.qdrant.io", api_key="fake-key")
        store.connect()

        # Default (include_private=False)
        doc_public = store.get_document("student_123")
        assert doc_public is not None
        assert "private" not in doc_public["metadata"]
        assert doc_public["metadata"]["name"] == "Jane Doe"

        # Admin mode (include_private=True)
        doc_admin = store.get_document("student_123", include_private=True)
        assert doc_admin is not None
        assert "private" in doc_admin["metadata"]
        assert doc_admin["metadata"]["private"]["email"] == "jane@example.com"
