"""System Instructions & Persona Prompts for Riva Voice Assistant.

Exposes a backward-compatible public interface while delegating prompt
compilation, token budgeting, and validation to the production PromptBuilder.
"""

from typing import Optional
from voice_speech.engine.config.prompt_system.builder import PromptBuilder, estimate_token_count

# Singleton prompt builder instance with startup validation
_builder = PromptBuilder()
_builder.validate_all()

BASE_INSTRUCTION: str = _builder.build_base_instruction()


def get_gender_persona_instruction(gender: str = "female") -> str:
    """Generates the authoritative persona directive string."""
    return "\n" + _builder.build_persona_layer(gender=gender) + "\n"


LANGUAGE_DIRECTIVES: dict[str, str] = {
    locale_code: "\n" + _builder.get_locale(locale_code).instruction.strip() + "\n"
    for locale_code in ["hindi", "english", "hinglish", "auto"]
}


def get_system_instruction(language: str = "auto", gender: str = "female") -> str:
    """Builds the complete system instruction using the production PromptBuilder.

    Args:
        language: Language code ('auto', 'hindi', 'english', 'hinglish').
        gender: Persona gender ('female', 'male').

    Returns:
        Formatted string system instruction for Gemini Live.
    """
    return _builder.build(language=language, gender=gender)
