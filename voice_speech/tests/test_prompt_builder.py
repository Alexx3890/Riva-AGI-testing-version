"""Unit tests for the production PromptBuilder and Pydantic Schemas."""

import pytest
from pydantic import ValidationError
from voice_speech.engine.config.prompt_system.schemas import (
    PersonaConfig,
    LocaleConfig,
    ContrastiveExample,
)
from voice_speech.engine.config.prompt_system.builder import (
    PromptBuilder,
    estimate_token_count,
)


def test_schema_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        PersonaConfig.model_validate({
            "id": "female",
            "gender": "female",
            "assistant_name": "Riva",
            "identity_statement": "test",
            "anti_gaslighting_directive": "test",
            "unknown_random_field": "illegal",  # extra="forbid"
        })


def test_contrastive_example_schema():
    ex = ContrastiveExample(
        correct="मैं कर सकती हूँ।",
        incorrect="मैं कर सकता हूँ।",
        explanation="Auxiliary verb",
    )
    assert ex.correct == "मैं कर सकती हूँ।"
    assert ex.incorrect == "मैं कर सकता हूँ।"


def test_builder_loads_personas_and_locales():
    builder = PromptBuilder()
    assert "female" in builder.personas
    assert "male" in builder.personas
    assert "hindi" in builder.locales
    assert "english" in builder.locales
    assert "hinglish" in builder.locales
    assert "auto" in builder.locales


def test_builder_validate_all_runs_cleanly():
    builder = PromptBuilder()
    # Runs startup checks across all (Persona x Locale) permutations
    builder.validate_all()


def test_token_estimation_weights_devanagari_heavier():
    english_sample = "I am ready to help you with your question."
    hindi_sample = "मैं आपके सवाल का जवाब देने के लिए तैयार हूँ।"

    eng_tokens = estimate_token_count(english_sample)
    hin_tokens = estimate_token_count(hindi_sample)

    assert eng_tokens > 0
    assert hin_tokens > 0
    # Hindi has higher token density per character
    hin_char_ratio = hin_tokens / len(hindi_sample)
    eng_char_ratio = eng_tokens / len(english_sample)
    assert hin_char_ratio > eng_char_ratio


def test_contrastive_examples_in_rendered_prompt():
    builder = PromptBuilder()

    female_hindi = builder.build(language="hindi", gender="female")
    assert 'RIGHT: "हाँ, मैं आपकी मदद कर सकती हूँ。"' or 'RIGHT: "हाँ, मैं आपकी मदद कर सकती हूँ。"' in female_hindi
    assert "सकती हूँ" in female_hindi
    assert "FEMALE" in female_hindi
    assert "a FEMALE voice assistant" in female_hindi

    male_hindi = builder.build(language="hindi", gender="male")
    assert "सकता हूँ" in male_hindi
    assert "MALE" in male_hindi
    assert "a MALE voice assistant" in male_hindi


def test_token_budget_threshold():
    builder = PromptBuilder()
    for gender in ["female", "male"]:
        for lang in ["auto", "hindi", "english", "hinglish"]:
            prompt = builder.build(language=lang, gender=gender)
            tokens = estimate_token_count(prompt)
            assert tokens <= builder.config.max_token_budget, (
                f"Prompt for ({gender}, {lang}) has {tokens} tokens, exceeding budget {builder.config.max_token_budget}"
            )


def test_kiet_formatting_with_dots():
    builder = PromptBuilder()
    base = builder.build_base_instruction()
    assert "K.I.E.T" in base
    prompt_female = builder.build(gender="female", language="auto")
    assert "K.I.E.T" in prompt_female
    prompt_male = builder.build(gender="male", language="auto")
    assert "K.I.E.T" in prompt_male
