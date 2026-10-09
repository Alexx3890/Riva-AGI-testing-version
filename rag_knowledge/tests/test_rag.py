"""Hermetic test suite for rag_knowledge."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from rag_knowledge.clients.gemini_client import GeminiRAGClient
from rag_knowledge.ingestion.ingest import (
    assert_no_private_in_embedded_fields,
    build_unified_student_documents,
)
from rag_knowledge.retrieval.retriever import KnowledgeRetriever
from rag_knowledge.service import RAGService
from rag_knowledge.storage.qdrant_storage import QdrantKnowledgeStore, string_to_point_id


def test_qdrant_storage_and_retriever():
    assert string_to_point_id("doc_1") == string_to_point_id("doc_1")

    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_hit = MagicMock(
        id="uuid_1",
        score=0.9,
        payload={"id": "doc_1", "title": "Test Title", "content": "Test Content", "metadata": {}},
    )
    mock_client.query_points.return_value = MagicMock(points=[mock_hit])
    mock_client.scroll.return_value = ([], None)

    with patch("rag_knowledge.storage.qdrant_storage.QdrantClient", return_value=mock_client), \
         patch("rag_knowledge.storage.qdrant_storage.TextEmbedding") as mock_embed_cls:
        mock_model = MagicMock()
        mock_model.embed.return_value = [[0.1] * 384]
        mock_embed_cls.return_value = mock_model

        store = QdrantKnowledgeStore(url="https://fake.qdrant.io", api_key="fake_key")
        assert store.connect()
        assert store.upsert_documents([{"id": "doc_1", "title": "Test"}]) == 1

        retriever = KnowledgeRetriever(store=store)
        results = retriever.retrieve("query")
        assert len(results) == 1
        assert results[0]["id"] == "doc_1"


@pytest.mark.anyio
async def test_rag_service_fallback_and_gemini():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"id": "doc_1", "title": "Doc 1", "summary": "Summary text", "content": "Content text", "score": 90.0}
    ]

    service_fallback = RAGService(retriever=mock_retriever, llm_client=GeminiRAGClient(api_key=""))
    res = await service_fallback.query("Test question")
    assert "Summary text" in res

    mock_llm = MagicMock(spec=GeminiRAGClient)
    mock_llm.is_configured = True
    mock_llm.generate_answer = AsyncMock(return_value="AI answer")
    service_llm = RAGService(retriever=mock_retriever, llm_client=mock_llm)
    assert await service_llm.query("Test question") == "AI answer"


def test_ingest_privacy_guards():
    nom = {"U1": {"uid": "U1", "name": "User One", "phone": "123", "email": "user@test.local"}}
    uid = {"U1": {"uid": "U1", "name": "User One"}}
    docs = build_unified_student_documents(nom, uid, [])
    assert len(docs) == 1
    assert_no_private_in_embedded_fields(docs)
    assert "@" not in docs[0]["content"]

    leaky_doc = [{"id": "bad", "content": "user@test.local", "summary": "", "title": ""}]
    with pytest.raises(ValueError):
        assert_no_private_in_embedded_fields(leaky_doc)


def test_multi_format_ingestion(tmp_path):
    from rag_knowledge.ingestion import (
        load_json_documents,
        load_csv_documents,
        load_text_or_markdown,
        load_source_documents,
    )

    json_file = tmp_path / "faq.json"
    json_file.write_text('[{"title": "FAQ 1", "answer": "Answer 1"}]', encoding="utf-8")
    assert len(load_json_documents(json_file)) == 1

    csv_file = tmp_path / "data.csv"
    csv_file.write_text("title,content\nItem 1,Detail 1", encoding="utf-8")
    assert len(load_csv_documents(csv_file)) == 1

    md_file = tmp_path / "doc.md"
    md_file.write_text("# Section 1\nContent 1\n# Section 2\nContent 2", encoding="utf-8")
    assert len(load_text_or_markdown(md_file)) == 2

    from rag_knowledge.ingestion import load_pdf_documents
    with patch("pypdf.PdfReader") as mock_pdf:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Header\nDescription"
        mock_pdf.return_value.pages = [mock_page]
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        assert len(load_pdf_documents(pdf_file)) == 1

    docs = load_source_documents(tmp_path)
    assert len(docs) >= 4


def test_separate_vision_and_text_api_keys(monkeypatch):
    from rag_knowledge.ingestion.ingest import get_vision_api_key, get_vision_model
    from rag_knowledge.clients.gemini_client import GeminiRAGClient

    # Set distinct keys for vision and text
    monkeypatch.setenv("GEMINI_API_KEY", "general_key")
    monkeypatch.setenv("GEMINI_TEXT_API_KEY", "dedicated_text_key")
    monkeypatch.setenv("GEMINI_VISION_API_KEY", "dedicated_vision_key")
    monkeypatch.setenv("GEMINI_VISION_MODEL", "gemini-vision-custom")

    # Text client uses text key
    client = GeminiRAGClient()
    assert client.api_key == "dedicated_text_key"

    # Vision extractor uses vision key and vision model
    assert get_vision_api_key() == "dedicated_vision_key"
    assert get_vision_model() == "gemini-vision-custom"

