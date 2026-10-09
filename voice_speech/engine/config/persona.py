"""Authoritative Agent Persona and Voice-Gender Synchronization.

Provides a single source of truth for:
1. Prebuilt TTS voice models and their canonical gender mapping.
2. Immutable AgentPersona data model.
3. Authoritative persona resolution ensuring TTS voice, gender identity,
   and system instructions remain in lockstep.
"""

from dataclasses import dataclass
import logging
from typing import Literal, Optional

logger = logging.getLogger("riva.persona")

GenderType = Literal["female", "male"]

# Explicit canonical mapping of Gemini Live voice names to persona genders
VOICE_GENDER_REGISTRY: dict[str, GenderType] = {
    "aoede": "female",
    "kore": "female",
    "puck": "male",
    "charon": "male",
    "fenrir": "male",
}

# Canonical voice name casing
CANONICAL_VOICE_NAMES: dict[str, str] = {
    "aoede": "Aoede",
    "kore": "Kore",
    "puck": "Puck",
    "charon": "Charon",
    "fenrir": "Fenrir",
}

DEFAULT_VOICE = "Aoede"
DEFAULT_GENDER: GenderType = "female"
VALID_VOICES: set[str] = set(CANONICAL_VOICE_NAMES.values())


@dataclass(frozen=True)
class AgentPersona:
    """Authoritative, immutable container for active voice persona identity."""
    voice_id: str
    gender: GenderType

    def __post_init__(self):
        norm_gender = (self.gender or DEFAULT_GENDER).strip().lower()
        if norm_gender not in ("female", "male"):
            object.__setattr__(self, "gender", DEFAULT_GENDER)
        else:
            object.__setattr__(self, "gender", norm_gender)

        clean_voice = CANONICAL_VOICE_NAMES.get(
            (self.voice_id or DEFAULT_VOICE).strip().lower(),
            DEFAULT_VOICE,
        )
        object.__setattr__(self, "voice_id", clean_voice)


def resolve_persona(
    voice: Optional[str] = None,
    configured_gender: Optional[str] = None,
) -> AgentPersona:
    """Resolves and synchronizes the authoritative AgentPersona.

    Guarantees:
    1. A single normalized source of truth for voice and gender.
    2. Voice casing is canonicalized against supported Gemini Live voices.
    3. If configured_gender is passed, it is strictly validated ('female' | 'male').
    4. TTS voice and conversational persona gender are synchronized. If a conflict
       exists between voice registry and configured gender, the registry mapping
       is treated as authoritative to prevent voice-personality mismatch.
    5. Clean diagnostic logging is emitted:
       Selected Voice: <voiceId> | Configured Gender: <gender> | Active Persona: <gender>

    Args:
        voice: Requested voice name (e.g., 'Aoede', 'Puck', 'Fenrir').
        configured_gender: Optional explicit gender string ('female' | 'male').

    Returns:
        Immutable AgentPersona instance.
    """
    clean_voice_input = (voice or DEFAULT_VOICE).strip().lower()
    canonical_voice = CANONICAL_VOICE_NAMES.get(clean_voice_input, DEFAULT_VOICE)
    registry_gender = VOICE_GENDER_REGISTRY.get(canonical_voice.lower(), DEFAULT_GENDER)

    norm_configured = (configured_gender or "").strip().lower()
    if norm_configured in ("female", "male"):
        active_gender: GenderType = norm_configured
    else:
        active_gender = registry_gender

    # Safety check: Prevent voice-persona mismatch (e.g. Aoede voice with Male persona)
    if active_gender != registry_gender:
        logger.warning(
            f"Voice/Gender mismatch detected! Voice '{canonical_voice}' is mapped to '{registry_gender}', "
            f"but received configured gender '{active_gender}'. "
            f"Resynchronizing active persona to '{registry_gender}' to maintain lockstep."
        )
        active_gender = registry_gender

    persona = AgentPersona(voice_id=canonical_voice, gender=active_gender)

    # Standardized development logging requirement
    logger.info(
        f"Selected Voice: {persona.voice_id} | "
        f"Configured Gender: {persona.gender.upper()} | "
        f"Active Persona: {persona.gender.upper()}"
    )

    return persona
