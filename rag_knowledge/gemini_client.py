"""Google Gemini Client for RAG Knowledge Synthesis.

Uses Google Gemini's REST API (gemini-flash-latest / Gemini Flash models) to
synthesize conversational, voice-optimized responses based on retrieved context.
Zero external runtime dependencies (pure standard library urllib).
"""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger("rag.gemini")

DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
FALLBACK_GEMINI_MODELS = ["gemini-3-flash-preview", "gemini-flash-latest"]


class GeminiRAGClient:
    """Client for querying Google Gemini API with RAG context."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "").strip()
        if model is not None:
            self.model = model
        else:
            rag_model = os.getenv("GEMINI_RAG_MODEL", "").strip()
            gen_model = os.getenv("GEMINI_MODEL", "").strip()
            if gen_model and "live" in gen_model.lower():
                gen_model = ""
            self.model = rag_model or gen_model or DEFAULT_GEMINI_MODEL
        env_timeout = float(os.getenv("GEMINI_TIMEOUT", "10.0"))
        self.timeout = timeout if timeout is not None else env_timeout

    @property
    def is_configured(self) -> bool:
        """Returns True if a Gemini API key is set."""
        return bool(self.api_key)

    async def generate_answer(self, query: str, context: str) -> Optional[str]:
        """Synthesizes a voice-friendly answer using Google Gemini Flash API.

        Args:
            query: The user's original question.
            context: Retrieved facts/knowledge context from knowledge store.

        Returns:
            Synthesized response text, or None if key is missing or call fails.
        """
        api_key = self.api_key
        if not api_key:
            logger.debug("GEMINI_API_KEY is not set. Using retrieved context directly.")
            return None

        system_prompt = (
            "You are Riva's voice knowledge assistant. Answer the user's question directly, warmly, "
            "and concisely using ONLY the provided reference facts in <context>. "
            "If the context does not contain enough information to answer the question, state that you do not have that information. "
            "Do not fabricate facts. Keep the answer to 2-3 natural sentences suitable for spoken conversation."
        )

        safe_context = context.replace("</context>", "")
        safe_query = query.replace("</user_question>", "").replace("</context>", "")

        user_content = (
            f"<context>\n{safe_context}\n</context>\n\n"
            f"<user_question>\n{safe_query}\n</user_question>\n\n"
            f"Please provide a concise, spoken answer based strictly on the reference context."
        )

        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 350,
                "temperature": 0.2,
            }
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "RivaRAG/1.0",
        }

        loop = asyncio.get_running_loop()

        # Build list of models to try (configured model first, then fallback models)
        models_to_try = [self.model]
        for fb in FALLBACK_GEMINI_MODELS:
            if fb not in models_to_try:
                models_to_try.append(fb)

        def _call_api_with_model(model_name: str) -> tuple[Optional[str], Optional[int]]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
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
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and "text" in parts[0]:
                                return parts[0]["text"].strip(), None
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
                answer, err_code = _call_api_with_model(model_candidate)
                if answer:
                    return answer
                # Only retry on 503 (high demand) or 404 (model deprecated/unavailable)
                if err_code not in (503, 404):
                    break
            return None

        try:
            answer = await loop.run_in_executor(None, _call_api)
            if answer:
                logger.debug("Gemini generated response (%d chars)", len(answer))
                return answer
        except Exception as e:
            logger.error(f"Async executor error calling Gemini: {e}")

        return None
