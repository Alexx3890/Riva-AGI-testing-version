"""Google Gemini client for RAG knowledge synthesis."""

import asyncio
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional

from ..prompts import (
    DEFAULT_SYSTEM_INSTRUCTION,
    PROMPT_VERSION,
    format_rag_user_prompt,
)

logger = logging.getLogger("rag.gemini")

def get_default_gemini_model() -> str:
    """Retrieves primary Gemini model from environment without hardcoded fallbacks."""
    try:
        from rag_knowledge import load_env
        load_env()
    except ImportError:
        pass
    return (
        os.getenv("GEMINI_TEXT_MODEL", "").strip()
        or os.getenv("GEMINI_RAG_MODEL", "").strip()
        or os.getenv("GEMINI_MODEL", "").strip()
        or os.getenv("GEMINI_DEFAULT_MODEL", "").strip()
    )


def get_fallback_gemini_models() -> list[str]:
    """Retrieves fallback Gemini models from environment without hardcoded strings."""
    try:
        from rag_knowledge import load_env
        load_env()
    except ImportError:
        pass
    raw = os.getenv("GEMINI_FALLBACK_MODELS", "").strip()
    return [m.strip() for m in raw.split(",") if m.strip()]


DEFAULT_GEMINI_MODEL = get_default_gemini_model()
FALLBACK_GEMINI_MODELS = get_fallback_gemini_models()


class GeminiRAGClient:
    """Client for querying Google Gemini API with RAG context."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_models: Optional[list] = None,
        timeout: Optional[float] = None,
        system_instruction: Optional[str] = None,
    ):
        try:
            from rag_knowledge import load_env
            load_env()
        except ImportError:
            pass

        self.api_key = (
            api_key
            if api_key is not None
            else (
                os.getenv("GEMINI_TEXT_API_KEY", "").strip()
                or os.getenv("GEMINI_RAG_API_KEY", "").strip()
                or os.getenv("GEMINI_API_KEY", "").strip()
            )
        )
        if model is not None:
            self.model = model
        else:
            rag_model = os.getenv("GEMINI_TEXT_MODEL", "").strip() or os.getenv("GEMINI_RAG_MODEL", "").strip()
            gen_model = os.getenv("GEMINI_MODEL", "").strip()
            if gen_model and "live" in gen_model.lower():
                gen_model = ""
            self.model = rag_model or gen_model or get_default_gemini_model()
        self.fallback_models = fallback_models if fallback_models is not None else get_fallback_gemini_models()
        env_timeout = float(os.getenv("GEMINI_TIMEOUT", "20.0"))
        self.timeout = timeout if timeout is not None else env_timeout
        self.system_instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION

    @property
    def is_configured(self) -> bool:
        """Returns True if a Gemini API key is set."""
        return bool(self.api_key)

    async def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Synthesizes a voice-friendly answer using Google Gemini Flash API.

        Args:
            query: The user's original question.
            context: Retrieved facts/knowledge context from knowledge store.
            system_prompt: Optional system instruction override.

        Returns:
            Synthesized response text, or None if key is missing or call fails.
        """
        api_key = self.api_key
        if not api_key:
            logger.debug("GEMINI_API_KEY is not set. Using retrieved context directly.")
            return None

        effective_system_prompt = system_prompt or self.system_instruction
        user_content = format_rag_user_prompt(query, context)
        logger.debug("Synthesizing RAG answer [model=%s, prompt_version=%s]", self.model, PROMPT_VERSION)

        payload = {
            "systemInstruction": {
                "parts": [{"text": effective_system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1500")),
                "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
            }
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "RivaRAG/1.0",
            "x-goog-api-key": api_key,
        }

        models_to_try = [self.model]
        for fb in self.fallback_models:
            if fb and fb not in models_to_try:
                models_to_try.append(fb)

        deadline = time.time() + max(self.timeout * 1.5, 6.0)

        def _call_api_with_model(model_name: str) -> tuple[Optional[str], Optional[int]]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        raw = resp.read().decode("utf-8")
                        data = json.loads(raw)
                        candidates = data.get("candidates", [])
                        if candidates:
                            cand = candidates[0]
                            finish_reason = cand.get("finishReason", "STOP")
                            if finish_reason not in ("STOP", ""):
                                logger.debug("Gemini response (%s) finishReason: %s", model_name, finish_reason)
                            parts = cand.get("content", {}).get("parts", [])
                            text_parts = [p.get("text", "") for p in parts if "text" in p]
                            if text_parts:
                                return "".join(text_parts).strip(), None
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8", errors="ignore")
                logger.warning(f"Gemini API ({model_name}) HTTP Error {e.code}: {err_msg[:160]}")
                return None, e.code
            except Exception as e:
                logger.warning(f"Gemini API ({model_name}) call error: {e}")
                return None, None
            return None, None

        def _call_api() -> Optional[str]:
            for model_candidate in models_to_try:
                if time.time() >= deadline:
                    logger.warning("Gemini cascade overall deadline exceeded.")
                    break
                answer, err_code = _call_api_with_model(model_candidate)
                if answer:
                    return answer
                if err_code not in (503, 429, 404, 408, None):
                    break
            return None

        try:
            answer = await asyncio.to_thread(_call_api)
            if answer:
                logger.debug("Gemini generated response (%d chars)", len(answer))
                return answer
        except Exception as e:
            logger.error(f"Async worker error calling Gemini: {e}")

        return None
