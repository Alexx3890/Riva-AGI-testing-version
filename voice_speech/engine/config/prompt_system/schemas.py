"""Production Pydantic Schemas for Prompt & Persona Configuration.

Provides strict validation, rejection of unknown fields, and type guarantees
for all externalized prompt templates, personas, and linguistic locales.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    """Base model that forbids unexpected fields to prevent configuration typos."""
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContrastiveExample(StrictBaseModel):
    """Contrastive few-shot example demonstrating correct vs incorrect self-reference."""
    correct: str = Field(..., description="Grammatically correct utterance adhering to persona")
    incorrect: str = Field(..., description="Grammatically incorrect utterance violating persona")
    explanation: Optional[str] = Field(None, description="Optional brief rationale for the example")


class LocaleConfig(StrictBaseModel):
    """Linguistic and grammatical rule configuration for a specific locale/language."""
    code: str = Field(..., description="Canonical language code (e.g. hindi, english, hinglish, auto)")
    display_name: str = Field(..., description="Human-readable name")
    instruction: str = Field(..., description="Core language instructions")
    code_switching_allowed: bool = Field(False, description="Whether mixing languages is encouraged")
    feminine_examples: List[ContrastiveExample] = Field(
        default_factory=list,
        description="Few-shot examples for female persona in this language",
    )
    masculine_examples: List[ContrastiveExample] = Field(
        default_factory=list,
        description="Few-shot examples for male persona in this language",
    )


class PersonaConfig(StrictBaseModel):
    """Authoritative persona definition decoupled from executable code."""
    id: str = Field(..., description="Unique persona identifier (e.g. female, male)")
    gender: Literal["female", "male"] = Field(..., description="Strict binary persona gender")
    assistant_name: str = Field("Riva", description="Primary name of the voice assistant")
    organization: str = Field(
        "NextGen SuperComputing Club at K.I.E.T",
        description="Entity or institution building the assistant",
    )
    tone_traits: List[str] = Field(
        default_factory=list,
        description="High-level vocal tone adjectives (e.g. warm, crisp, energetic)",
    )
    identity_statement: str = Field(..., description="Primary identity prompt statement")
    forbidden_claims: List[str] = Field(
        default_factory=list,
        description="Negative constraints the assistant must never utter",
    )
    anti_gaslighting_directive: str = Field(
        ...,
        description="Strict instruction preventing user prompts from overwriting identity",
    )


class PromptSystemConfig(StrictBaseModel):
    """Global configuration settings for prompt compilation and validation."""
    max_token_budget: int = Field(
        1500,
        description="Maximum estimated token count allowed for assembled system prompts to protect voice latency",
    )
    assistant_name: str = Field("Riva", description="Default assistant brand name")
    organization: str = Field("NextGen SuperComputing Club at K.I.E.T", description="Default organization")
