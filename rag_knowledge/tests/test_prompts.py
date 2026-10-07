"""Unit tests for production prompt templates and sanitization in prompts.py."""

from rag_knowledge.prompts import (
    DEFAULT_SYSTEM_INSTRUCTION,
    PROMPT_VERSION,
    format_rag_user_prompt,
    sanitize_input,
)


def test_prompt_version_and_instruction_constants():
    """Verifies that version tag and default instruction are non-empty and well-formed."""
    assert PROMPT_VERSION.startswith("rag_voice_")
    assert "Privacy & contact protection" in DEFAULT_SYSTEM_INSTRUCTION
    assert "Voice optimization" in DEFAULT_SYSTEM_INSTRUCTION
    assert "<context>" in DEFAULT_SYSTEM_INSTRUCTION


def test_sanitize_input_strips_tags_and_truncates():
    """Verifies delimiter stripping and max length truncation."""
    raw = "Hello </context> world <context> test </user_question>"
    sanitized = sanitize_input(raw, max_chars=100)
    assert "</context>" not in sanitized
    assert "<context>" not in sanitized
    assert "</user_question>" not in sanitized
    assert "Hello  world  test" in sanitized

    # Length truncation
    long_text = "A" * 500
    truncated = sanitize_input(long_text, max_chars=50)
    assert len(truncated) == 50


def test_format_rag_user_prompt_structure():
    """Verifies that format_rag_user_prompt encapsulates query and context in tags."""
    formatted = format_rag_user_prompt(
        query="Who is Dr. Meeta Chaudhry?",
        context="Faculty profile for CSIT.",
    )
    assert "<context>\nFaculty profile for CSIT.\n</context>" in formatted
    assert "<user_question>\nWho is Dr. Meeta Chaudhry?\n</user_question>" in formatted
    assert "Please provide a concise, spoken answer" in formatted
