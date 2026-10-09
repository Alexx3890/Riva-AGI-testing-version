"""Gemini Live Function Calling Tools & Registry.

Provides zero-key real-time news retrieval (Google News RSS + NewsAPI fallback)
and an extensible dispatcher registry for tool calls.
"""

import asyncio
import json
import logging
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Awaitable, Callable, Dict, List
from google.genai import types

logger = logging.getLogger("riva.tools")


async def fetch_news_summary(query: str) -> str:
    """Fetches real-time web news, facts, and live context using Tavily Search (if key provided),

    NewsAPI (fallback), or universal Google News RSS (zero-key fallback).
    """
    clean_query = query.strip()
    if not clean_query:
        clean_query = "top world news"

    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    news_api_key = os.getenv("NEWS_API_KEY", "").strip()
    loop = asyncio.get_running_loop()

    # 1. Primary: Tavily AI Search (rich context, snippets, and answers if key provided)
    if tavily_api_key:
        try:
            payload = {
                "api_key": tavily_api_key,
                "query": clean_query,
                "search_depth": "basic",
                "max_results": 4,
                "include_answer": True,
            }
            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "RivaVoice/1.0"},
                method="POST",
            )

            def _fetch_tavily():
                with urllib.request.urlopen(req, timeout=4.5) as resp:
                    return resp.read()

            raw_bytes = await loop.run_in_executor(None, _fetch_tavily)
            data = json.loads(raw_bytes.decode("utf-8"))

            snippets = []
            answer = data.get("answer")
            if answer:
                snippets.append(f"Direct Answer: {answer}")

            for item in data.get("results", [])[:4]:
                title = item.get("title", "").strip()
                content = item.get("content", "").strip()
                if title and content:
                    snippets.append(f"{title}: {content}")
                elif title:
                    snippets.append(title)

            if snippets:
                summary = " | ".join(snippets)[:1400]
                logger.info(f"Live Tavily response for '{clean_query}': {summary[:120]!r}")
                return summary
        except Exception as e:
            logger.warning(f"Tavily search error (falling back to NewsAPI / Google News): {e}")

    # 2. Secondary: NewsAPI.org query (if key provided)
    if news_api_key:
        try:
            encoded = urllib.parse.quote(clean_query)
            url = f"https://newsapi.org/v2/everything?q={encoded}&pageSize=4&sortBy=publishedAt&apiKey={news_api_key}"
            req = urllib.request.Request(url, headers={"User-Agent": "RivaVoice/1.0"})

            def _fetch_newsapi():
                with urllib.request.urlopen(req, timeout=3.5) as resp:
                    return resp.read()

            raw_json = await loop.run_in_executor(None, _fetch_newsapi)
            data = json.loads(raw_json)
            articles = data.get("articles", [])
            items = []
            for a in articles[:4]:
                title = a.get("title", "").strip()
                desc = a.get("description", "").strip()
                if title and desc:
                    items.append(f"{title} - {desc}")
                elif title:
                    items.append(title)
            if items:
                summary = " | ".join(items)[:1000]
                logger.info(f"Live NewsAPI response for '{clean_query}': {summary[:120]!r}")
                return summary
        except Exception as e:
            logger.warning(f"NewsAPI error (falling back to Google News RSS): {e}")

    # 3. Universal Zero-Key Fallback: Google News RSS
    try:
        encoded = urllib.parse.quote(clean_query)
        url = f"https://news.google.com/rss/search?q={encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

        def _fetch_rss():
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                return resp.read()

        xml_data = await loop.run_in_executor(None, _fetch_rss)
        root = ET.fromstring(xml_data)
        items = root.findall(".//item")

        headlines = []
        for item in items[:8]:
            title = item.find("title")
            if title is not None and title.text:
                # Strip trailing publisher tag while preserving internal hyphens in scores & stats
                clean_title = title.text.rsplit(" - ", 1)[0].strip() if " - " in title.text else title.text.strip()
                if clean_title and clean_title not in headlines:
                    headlines.append(clean_title)

        if headlines:
            summary = " | ".join(headlines)[:1000]
            logger.info(f"Live News RSS response for '{clean_query}': {summary[:120]!r}")
            return summary

        return f"No recent breaking news found for '{clean_query}'."
    except Exception as e:
        logger.warning(f"News RSS fetch error for '{clean_query}': {e}")
        return f"Could not retrieve recent news for '{clean_query}'."


# Tool Declarations
NEWS_TOOL_DECLARATION = types.FunctionDeclaration(
    name="get_latest_news",
    description=(
        "Search real-time web news, current events, recent developments, facts, or live updates on any topic. "
        "Call this tool whenever the user asks about current affairs, breaking news, recent events, "
        "people, organizations, technology, culture, weather, statistics, or any topic requiring fresh or up-to-date information."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={"query": types.Schema(type="STRING", description="Search query keywords or topic to look up")},
        required=["query"],
    ),
)

ORCHESTRATION_TOOL_DECLARATION = types.FunctionDeclaration(
    name="run_orchestration_task",
    description=(
        "Execute an autonomous multi-agent task via Riva-AGI's orchestration pipeline. "
        "Use this tool whenever the user asks to write code, debug software, inspect files or directories, "
        "create or edit files, execute system commands, or solve complex technical problems."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={"task": types.Schema(type="STRING", description="Detailed description of the autonomous task to perform")},
        required=["task"],
    ),
)

OPEN_BROWSER_TOOL_DECLARATION = types.FunctionDeclaration(
    name="open_website_in_browser",
    description="Open a specified website or web page in Google Chrome or the default web browser.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "url": types.Schema(type="STRING", description="The complete web address (URL) to open, e.g. https://www.google.com"),
            "browser": types.Schema(type="STRING", description="Browser to use: 'chrome' or 'default' (defaults to 'chrome')")
        },
        required=["url"],
    ),
)

DEFAULT_TOOLS: List[types.Tool] = [
    types.Tool(function_declarations=[
        NEWS_TOOL_DECLARATION,
        ORCHESTRATION_TOOL_DECLARATION,
        OPEN_BROWSER_TOOL_DECLARATION,
    ])
]


async def _handle_get_latest_news(args: Dict[str, Any]) -> str:
    query = str((args or {}).get("query", ""))
    return await fetch_news_summary(query)


async def execute_orchestration_task(task: str) -> str:
    """Dispatches a task to the RIVA LangGraph multi-agent orchestrator."""
    import uuid
    loop = asyncio.get_running_loop()

    def _run() -> str:
        try:
            from orchestration.orchestrator.main import create_orchestrator
            from orchestration import InputData, InputType

            app = create_orchestrator()
            task_id = f"voice-{uuid.uuid4().hex[:8]}"
            payload = InputData(input_type=InputType.TEXT, text_content=task, metadata={"source": "voice"})
            state = {
                "task_payload": payload,
                "agent": "fallback",
                "response_payload": None,
                "task_id": task_id,
                "session_id": f"voice-sess-{uuid.uuid4().hex[:6]}",
                "source": "voice",
                "complexity": "simple",
                "routing_decision": "fallback",
                "plan": [],
                "current_step": 0,
                "completed_steps": [],
                "feedback": "",
                "intent": "unknown",
                "confidence": 0.0,
            }
            final_res = None
            for step in app.stream(state):
                for node_name, node_update in step.items():
                    if isinstance(node_update, dict):
                        if "response_payload" in node_update and node_update["response_payload"]:
                            final_res = node_update["response_payload"]
            
            if final_res and hasattr(final_res, "content") and final_res.content:
                return str(final_res.content)
            return "Orchestration task completed successfully."
        except Exception as e:
            logger.error(f"Error in execute_orchestration_task: {e}", exc_info=True)
            return f"Error executing task: {str(e)}"

    res = await loop.run_in_executor(None, _run)
    return res[:1500]


async def _handle_run_orchestration_task(args: Dict[str, Any]) -> str:
    task = str((args or {}).get("task", "")).strip()
    if not task:
        return "Error: No task description provided."
    return await execute_orchestration_task(task)


async def _handle_open_website_in_browser(args: Dict[str, Any]) -> str:
    url = str((args or {}).get("url", "")).strip()
    browser = str((args or {}).get("browser", "chrome")).strip()
    if not url:
        return "Error: No URL provided."
    
    loop = asyncio.get_running_loop()
    try:
        from orchestration.tools import tool_registry
        return await loop.run_in_executor(
            None, 
            lambda: tool_registry.execute("open_browser", url=url, browser=browser)
        )
    except Exception as e:
        logger.error(f"Error opening browser for '{url}': {e}")
        return f"Error opening browser: {str(e)}"


# Extensible Tool Handler Registry
TOOL_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Awaitable[str]]] = {
    "get_latest_news": _handle_get_latest_news,
    "run_orchestration_task": _handle_run_orchestration_task,
    "open_website_in_browser": _handle_open_website_in_browser,
}


async def dispatch_tool_call(name: str, args: Dict[str, Any]) -> str:
    """Dispatches a function call to the registered handler.

    Args:
        name: Name of the function declared in tool schema.
        args: Parsed argument dictionary from the model.

    Returns:
        String result to return to the model in FunctionResponse.
    """
    handler = TOOL_REGISTRY.get(name)
    if not handler:
        logger.warning(f"No handler registered for tool call '{name}'")
        return f"Tool '{name}' is not supported."

    logger.info(f"Executing tool call '{name}' with args={args}")
    try:
        return await handler(args)
    except Exception as e:
        logger.error(f"Error executing tool '{name}': {e}", exc_info=True)
        return f"Error executing tool '{name}': {e}"
