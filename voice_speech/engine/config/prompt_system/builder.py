"""PromptBuilder: Layered Prompt Assembly with Jinja2 and Token Budgeting.

Assembles four deterministic layers:
1. Base Guardrails & Identity
2. Persona & Gender Contract
3. Locale & Linguistic Grammar (with Contrastive Few-Shot Examples)
4. Runtime Context & Constraints

Validates all permutations against token budgets at startup.
"""

from pathlib import Path
import logging
from typing import Dict, List, Optional
import jinja2
import yaml

from voice_speech.engine.config.prompt_system.schemas import (
    LocaleConfig,
    PersonaConfig,
    PromptSystemConfig,
)

logger = logging.getLogger("riva.prompt_builder")

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def estimate_token_count(text: str) -> int:
    """Estimates LLM token usage with weighted non-ASCII budgeting.

    Devanagari and multi-byte UTF-8 characters tokenize with higher token/word
    density in modern BPE tokenizers than English ASCII characters.
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii_chars = len(text) - ascii_chars
    estimated = (ascii_chars / 4.0) + (non_ascii_chars / 1.5)
    return int(estimated)


class PromptBuilder:
    """Compiles validated prompt layers and verifies token budgets."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.templates_dir = self.data_dir / "templates"
        self.personas_dir = self.data_dir / "personas"
        self.locales_dir = self.data_dir / "locales"

        self.config = PromptSystemConfig()
        self.personas: Dict[str, PersonaConfig] = {}
        self.locales: Dict[str, LocaleConfig] = {}

        # Jinja2 environment configured with StrictUndefined to reject missing variables
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir)),
            undefined=jinja2.StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        self._load_configs()

    def _load_configs(self) -> None:
        """Parses and validates all YAML configs through Pydantic models."""
        # Load Personas
        for file in self.personas_dir.glob("*.yaml"):
            with open(file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                persona = PersonaConfig.model_validate(data)
                self.personas[persona.gender.lower()] = persona

        # Load Locales
        for file in self.locales_dir.glob("*.yaml"):
            with open(file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                locale = LocaleConfig.model_validate(data)
                self.locales[locale.code.lower()] = locale

    def get_persona(self, gender: str) -> PersonaConfig:
        """Retrieves persona by gender with safe fallback to female."""
        norm = (gender or "female").strip().lower()
        return self.personas.get(norm, self.personas.get("female"))

    def get_locale(self, language: str) -> LocaleConfig:
        """Retrieves locale by language code with safe fallback to auto."""
        norm = (language or "auto").strip().lower()
        return self.locales.get(norm, self.locales.get("auto"))

    def build_base_instruction(
        self,
        assistant_name: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> str:
        """Renders the foundational assistant behavior rules."""
        template = self.jinja_env.get_template("base.jinja2")
        return template.render(
            assistant_name=assistant_name or self.config.assistant_name,
            organization=organization or self.config.organization,
        ).strip()

    def build_persona_layer(
        self,
        gender: str = "female",
        assistant_name: Optional[str] = None,
    ) -> str:
        """Renders the authoritative persona directive."""
        persona = self.get_persona(gender)
        template = self.jinja_env.get_template("persona.jinja2")
        return template.render(
            persona=persona,
            assistant_name=assistant_name or persona.assistant_name,
        ).strip()

    def build_locale_layer(
        self,
        language: str = "auto",
        gender: str = "female",
    ) -> str:
        """Renders the linguistic directive with contrastive few-shot examples."""
        persona = self.get_persona(gender)
        locale = self.get_locale(language)
        template = self.jinja_env.get_template("locale.jinja2")
        return template.render(
            persona=persona,
            locale=locale,
        ).strip()

    def build(
        self,
        language: str = "auto",
        gender: str = "female",
        assistant_name: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> str:
        """Assembles all layers into an immutable system prompt string.

        Layers:
        1. Base Rules & Tools
        2. Persona & Gender Identity Contract
        3. Locale & Contrastive Examples
        """
        base = self.build_base_instruction(
            assistant_name=assistant_name,
            organization=organization,
        )
        persona_layer = self.build_persona_layer(
            gender=gender,
            assistant_name=assistant_name,
        )
        locale_layer = self.build_locale_layer(
            language=language,
            gender=gender,
        )

        return f"{base}\n\n{persona_layer}\n\n{locale_layer}\n"

    def validate_all(self) -> None:
        """Renders all persona x locale permutations and asserts token budgets.

        Executed at server startup (fail-fast principle) to prevent latency regressions
        and broken templates in production.
        """
        for gender, persona in self.personas.items():
            for lang_code, locale in self.locales.items():
                rendered = self.build(language=lang_code, gender=gender)

                # 1. Verify no unresolved Jinja tags leaked into production
                assert "{{" not in rendered and "}}" not in rendered, (
                    f"Unrendered Jinja tag detected for persona={gender}, locale={lang_code}!"
                )
                assert "{%" not in rendered and "%}" not in rendered, (
                    f"Unrendered Jinja block detected for persona={gender}, locale={lang_code}!"
                )

                # 2. Check estimated token budget to preserve voice response latency
                tokens = estimate_token_count(rendered)
                if tokens > self.config.max_token_budget:
                    raise ValueError(
                        f"Prompt budget exceeded for ({gender}, {lang_code})! "
                        f"Estimated {tokens} tokens > max budget {self.config.max_token_budget}."
                    )

        logger.info(
            f"Prompt system validated: {len(self.personas)} personas x "
            f"{len(self.locales)} locales checked against token budget "
            f"(max {self.config.max_token_budget} tokens)."
        )
