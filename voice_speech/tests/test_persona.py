"""Unit tests for AgentPersona and authoritative Voice-Gender synchronization."""

import pytest
from voice_speech.engine.config.persona import (
    AgentPersona,
    resolve_persona,
    VOICE_GENDER_REGISTRY,
    CANONICAL_VOICE_NAMES,
    DEFAULT_VOICE,
    DEFAULT_GENDER,
)


def test_agent_persona_immutability():
    persona = AgentPersona(voice_id="Aoede", gender="female")
    with pytest.raises(Exception):
        persona.gender = "male"  # Frozen dataclass


def test_resolve_persona_female_voices():
    for voice in ["Aoede", "Kore", "aoede", "kore"]:
        persona = resolve_persona(voice=voice)
        assert persona.gender == "female"
        assert persona.voice_id in ("Aoede", "Kore")


def test_resolve_persona_male_voices():
    for voice in ["Puck", "Charon", "Fenrir", "puck", "charon", "fenrir"]:
        persona = resolve_persona(voice=voice)
        assert persona.gender == "male"
        assert persona.voice_id in ("Puck", "Charon", "Fenrir")


def test_resolve_persona_mismatch_resynchronization():
    # If caller specifies voice Aoede (female) but configured_gender="male",
    # the authoritative registry resynchronizes to female to avoid voice-identity mismatch.
    persona = resolve_persona(voice="Aoede", configured_gender="male")
    assert persona.gender == "female"
    assert persona.voice_id == "Aoede"

    # Similarly for male voice with female configured_gender
    persona2 = resolve_persona(voice="Puck", configured_gender="female")
    assert persona2.gender == "male"
    assert persona2.voice_id == "Puck"


def test_resolve_persona_unknown_voice_fallback():
    persona = resolve_persona(voice="UnknownNonExistentVoice")
    assert persona.voice_id == DEFAULT_VOICE
    assert persona.gender == DEFAULT_GENDER
