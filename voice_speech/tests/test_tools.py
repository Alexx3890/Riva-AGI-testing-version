"""Unit tests for Gemini Live Function Calling Tools & Registry."""

from unittest.mock import MagicMock, patch
import pytest
from voice_speech.engine.gemini.tools import (
    dispatch_tool_call,
    fetch_news_summary,
    TOOL_REGISTRY,
    NEWS_TOOL_DECLARATION,
    KNOWLEDGE_TOOL_DECLARATION,
)


def test_tool_declaration():
    assert NEWS_TOOL_DECLARATION.name == "get_latest_news"
    assert "query" in NEWS_TOOL_DECLARATION.parameters.properties
    assert "query" in NEWS_TOOL_DECLARATION.parameters.required


def test_knowledge_tool_declaration():
    assert KNOWLEDGE_TOOL_DECLARATION.name == "query_knowledge_base"
    assert "query" in KNOWLEDGE_TOOL_DECLARATION.parameters.properties
    assert "query" in KNOWLEDGE_TOOL_DECLARATION.parameters.required


@pytest.mark.anyio
async def test_dispatch_registered_tool():
    with patch("voice_speech.engine.gemini.tools.fetch_news_summary") as mock_fetch:
        mock_fetch.return_value = "Mocked Headline 1 | Mocked Headline 2"
        result = await dispatch_tool_call("get_latest_news", {"query": "artificial intelligence"})
        assert result == "Mocked Headline 1 | Mocked Headline 2"
        mock_fetch.assert_called_once_with("artificial intelligence")


@pytest.mark.anyio
async def test_fetch_news_summary_rss_mock():
    sample_rss = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
        <item><title>AI Breakthrough Announced - TechNews</title></item>
        <item><title>Global Summit Begins - WorldNews</title></item>
    </channel></rss>"""

    mock_resp = MagicMock()
    mock_resp.read.return_value = sample_rss
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        summary = await fetch_news_summary("tech")
        assert "AI Breakthrough Announced" in summary
        assert "Global Summit Begins" in summary


@pytest.mark.anyio
async def test_fetch_news_summary_tavily_mock(monkeypatch):
    import json
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")

    fake_tavily = {
        "answer": "Quantum computing reached a new milestone today.",
        "results": [
            {"title": "Quantum Leap Announced", "content": "Scientists demonstrated 1000 qubit fault tolerance."}
        ]
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(fake_tavily).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        summary = await fetch_news_summary("quantum computing")
        assert "Quantum computing reached a new milestone" in summary
        assert "1000 qubit fault tolerance" in summary



@pytest.mark.anyio
async def test_dispatch_unsupported_tool():
    result = await dispatch_tool_call("non_existent_tool", {})
    assert "not supported" in result


@pytest.mark.anyio
async def test_custom_tool_registration():
    async def sample_handler(args):
        return f"Echo: {args.get('text', '')}"

    TOOL_REGISTRY["test_echo"] = sample_handler
    try:
        result = await dispatch_tool_call("test_echo", {"text": "hello"})
        assert result == "Echo: hello"
    finally:
        TOOL_REGISTRY.pop("test_echo", None)


@pytest.mark.anyio
async def test_orchestration_tool_dispatch():
    with patch("voice_speech.engine.gemini.tools.execute_orchestration_task") as mock_exec:
        mock_exec.return_value = "Task completed successfully by Coder Agent."
        result = await dispatch_tool_call("run_orchestration_task", {"task": "write hello world"})
        assert "Coder Agent" in result
        mock_exec.assert_called_once_with("write hello world")


@pytest.mark.anyio
async def test_open_browser_tool_dispatch():
    with patch("orchestration.tools.tool_registry.execute") as mock_tool:
        mock_tool.return_value = "Successfully opened 'https://example.com' in Google Chrome."
        result = await dispatch_tool_call("open_website_in_browser", {"url": "https://example.com", "browser": "chrome"})
        assert "Successfully opened" in result


@pytest.mark.anyio
async def test_knowledge_tool_dispatch():
    with patch("rag_knowledge.query_rag") as mock_query:
        mock_query.return_value = "Verified fact from KIET knowledge base."
        result = await dispatch_tool_call("query_knowledge_base", {"query": "NextGen club details"})
        assert "Verified fact" in result
        mock_query.assert_called_once_with("NextGen club details")


@pytest.mark.anyio
async def test_get_latest_news_person_query_fallback():
    with patch("rag_knowledge.query_rag") as mock_rag:
        mock_rag.return_value = "Ankit Kumar Singh is a B.Tech student in AI at KIET."
        result = await dispatch_tool_call("get_latest_news", {"query": "Who is Ankit kumar singh?"})
        assert "Ankit Kumar Singh is a B.Tech student" in result
        mock_rag.assert_called_once_with("Who is Ankit kumar singh?")

