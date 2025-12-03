"""
Configuration loader for extraction pipeline.

Loads extractor settings from TOML config and builds provider-specific
response formats for structured output.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).parent / "config.toml"


@dataclass
class PricingConfig:
    """Pricing per 1M tokens in USD."""
    INPUT: float = 0.0
    OUTPUT: float = 0.0


@dataclass
class ProviderConfig:
    """Provider-specific routing configuration."""
    ORDER: list[str] = field(default_factory=list)
    ALLOW_FALLBACKS: bool = True


@dataclass
class ExtractorConfig:
    """Configuration for a single extractor."""
    MODEL: str
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 2048
    TIMEOUT: int = 60
    MAX_CONCURRENT: int = 10
    PRICING: PricingConfig = field(default_factory=PricingConfig)
    PROVIDER: ProviderConfig | None = None

    @property
    def is_openrouter(self) -> bool:
        """Check if this extractor uses OpenRouter."""
        return self.MODEL.startswith("openrouter/")

    @property
    def is_gemini(self) -> bool:
        """Check if this extractor uses Gemini."""
        return self.MODEL.startswith("gemini/")

    @property
    def needs_provider_routing(self) -> bool:
        """Check if provider routing is configured."""
        return self.PROVIDER is not None and len(self.PROVIDER.ORDER) > 0


@dataclass
class SchemaConfig:
    """Schema configuration for structured output."""
    NAME: str = "relationship_extraction"
    STRICT: bool = True
    RELATIONSHIP_TYPES: list[str] = field(default_factory=list)

    def build_json_schema(self) -> dict[str, Any]:
        """Build JSON schema for OpenAI-compatible providers."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.NAME,
                "strict": self.STRICT,
                "schema": {
                    "type": "object",
                    "properties": {
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity1": {"type": "string"},
                                    "entity2": {"type": "string"},
                                    "relationship": {
                                        "type": "string",
                                        "enum": self.RELATIONSHIP_TYPES,
                                    },
                                    "confidence": {"type": "number"},
                                    "evidence_text": {"type": "string"},
                                },
                                "required": [
                                    "entity1",
                                    "entity2",
                                    "relationship",
                                    "confidence",
                                    "evidence_text",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["relationships"],
                    "additionalProperties": False,
                },
            },
        }

    def build_gemini_response_schema(self) -> dict[str, Any]:
        """Build response schema for Gemini providers."""
        return {
            "type": "json_object",
            "response_schema": {
                "type": "object",
                "properties": {
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity1": {"type": "string"},
                                "entity2": {"type": "string"},
                                "relationship": {
                                    "type": "string",
                                    "enum": self.RELATIONSHIP_TYPES,
                                },
                                "confidence": {"type": "number"},
                                "evidence_text": {"type": "string"},
                            },
                            "required": [
                                "entity1",
                                "entity2",
                                "relationship",
                                "confidence",
                                "evidence_text",
                            ],
                        },
                    },
                },
                "required": ["relationships"],
            },
        }

    def build_batched_json_schema(self) -> dict[str, Any]:
        """Build JSON schema for batched mining with sentence indices."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.NAME,
                "strict": self.STRICT,
                "schema": {
                    "type": "object",
                    "properties": {
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity1": {"type": "string"},
                                    "entity2": {"type": "string"},
                                    "relationship": {
                                        "type": "string",
                                        "enum": self.RELATIONSHIP_TYPES,
                                    },
                                    "confidence": {"type": "number"},
                                    "evidence_sentence_indices": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
                                    "pmid": {"type": "string"},
                                },
                                "required": [
                                    "entity1",
                                    "entity2",
                                    "relationship",
                                    "confidence",
                                    "evidence_sentence_indices",
                                    "pmid",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["relationships"],
                    "additionalProperties": False,
                },
            },
        }

    def build_batched_gemini_schema(self) -> dict[str, Any]:
        """Build Gemini response schema for batched mining with sentence indices."""
        return {
            "type": "json_object",
            "response_schema": {
                "type": "object",
                "properties": {
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity1": {"type": "string"},
                                "entity2": {"type": "string"},
                                "relationship": {
                                    "type": "string",
                                    "enum": self.RELATIONSHIP_TYPES,
                                },
                                "confidence": {"type": "number"},
                                "evidence_sentence_indices": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                "pmid": {"type": "string"},
                            },
                            "required": [
                                "entity1",
                                "entity2",
                                "relationship",
                                "confidence",
                                "evidence_sentence_indices",
                                "pmid",
                            ],
                        },
                    },
                },
                "required": ["relationships"],
            },
        }


@dataclass
class PromptConfig:
    """Prompt templates for extraction."""
    SYSTEM: str = ""
    USER_TEMPLATE: str = ""


@dataclass
class BatchConfig:
    """Batch processing settings for pair-based extraction."""
    DEFAULT_CONCURRENT: int = 10
    RETRY_ATTEMPTS: int = 2
    RETRY_DELAY_MS: int = 1000


@dataclass
class BatchedConfig:
    """Configuration for batched mining (chunked abstracts)."""
    TARGET_TOKENS_PER_CHUNK: int = 5000
    MIN_CHUNKS: int = 3
    MAX_CONCURRENT: int = 5
    MAX_RETRIES: int = 3
    RETRY_DELAY_MS: int = 1000
    MIN_CONFIDENCE: float = 0.5
    MAX_TOKENS: int = 8192  # Higher for multi-abstract output


@dataclass
class BatchedPromptConfig:
    """Prompt templates for batched mining."""
    SYSTEM: str = ""
    USER_TEMPLATE: str = ""


@dataclass
class ExtractionConfig:
    """Complete extraction pipeline configuration."""
    DEFAULT_EXTRACTOR: str = "cerebras"
    EXTRACTORS: dict[str, ExtractorConfig] = field(default_factory=dict)
    SCHEMA: SchemaConfig = field(default_factory=SchemaConfig)
    PROMPT: PromptConfig = field(default_factory=PromptConfig)
    BATCH: BatchConfig = field(default_factory=BatchConfig)
    BATCHED: BatchedConfig = field(default_factory=BatchedConfig)
    BATCHED_PROMPT: BatchedPromptConfig = field(default_factory=BatchedPromptConfig)

    def get_extractor(self, name: str | None = None) -> ExtractorConfig:
        """Get extractor config by name, or default."""
        key = name or self.DEFAULT_EXTRACTOR
        if key not in self.EXTRACTORS:
            raise ValueError(
                f"Unknown extractor '{key}'. "
                f"Available: {list(self.EXTRACTORS.keys())}"
            )
        return self.EXTRACTORS[key]

    def get_response_format(self, extractor_name: str | None = None) -> dict[str, Any]:
        """Get provider-appropriate response format for structured output."""
        extractor = self.get_extractor(extractor_name)

        if extractor.is_gemini:
            return self.SCHEMA.build_gemini_response_schema()
        else:
            # OpenAI-compatible (OpenRouter, etc.)
            return self.SCHEMA.build_json_schema()

    def get_extra_params(self, extractor_name: str | None = None) -> dict[str, Any]:
        """Get extra parameters for LiteLLM call (e.g., provider routing)."""
        extractor = self.get_extractor(extractor_name)
        params: dict[str, Any] = {}

        if extractor.needs_provider_routing:
            params["extra_body"] = {
                "provider": {
                    "order": extractor.PROVIDER.ORDER,
                    "allow_fallbacks": extractor.PROVIDER.ALLOW_FALLBACKS,
                }
            }

        return params

    def get_batched_response_format(self, extractor_name: str | None = None) -> dict[str, Any]:
        """Get provider-appropriate response format for batched mining (includes PMID)."""
        extractor = self.get_extractor(extractor_name)

        if extractor.is_gemini:
            return self.SCHEMA.build_batched_gemini_schema()
        else:
            # OpenAI-compatible (OpenRouter, etc.)
            return self.SCHEMA.build_batched_json_schema()


def load_config(config_path: Path | None = None) -> ExtractionConfig:
    """Load extraction config from TOML file."""
    path = config_path or CONFIG_PATH

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Parse extractors
    extractors: dict[str, ExtractorConfig] = {}
    for name, ext_data in raw.get("EXTRACTORS", {}).items():
        pricing_data = ext_data.pop("PRICING", {})
        provider_data = ext_data.pop("PROVIDER", None)

        pricing = PricingConfig(
            INPUT=pricing_data.get("INPUT", 0.0),
            OUTPUT=pricing_data.get("OUTPUT", 0.0),
        )

        provider = None
        if provider_data:
            provider = ProviderConfig(
                ORDER=provider_data.get("ORDER", []),
                ALLOW_FALLBACKS=provider_data.get("ALLOW_FALLBACKS", True),
            )

        extractors[name] = ExtractorConfig(
            MODEL=ext_data.get("MODEL", ""),
            TEMPERATURE=ext_data.get("TEMPERATURE", 0.1),
            MAX_TOKENS=ext_data.get("MAX_TOKENS", 2048),
            TIMEOUT=ext_data.get("TIMEOUT", 60),
            MAX_CONCURRENT=ext_data.get("MAX_CONCURRENT", 10),
            PRICING=pricing,
            PROVIDER=provider,
        )

    # Parse schema
    schema_data = raw.get("SCHEMA", {})
    schema = SchemaConfig(
        NAME=schema_data.get("NAME", "relationship_extraction"),
        STRICT=schema_data.get("STRICT", True),
        RELATIONSHIP_TYPES=schema_data.get("RELATIONSHIPS", {}).get("TYPES", []),
    )

    # Parse prompt
    prompt_data = raw.get("PROMPT", {})
    prompt = PromptConfig(
        SYSTEM=prompt_data.get("SYSTEM", ""),
        USER_TEMPLATE=prompt_data.get("USER_TEMPLATE", ""),
    )

    # Parse batch settings (pair-based extraction)
    batch_data = raw.get("BATCH", {})
    batch = BatchConfig(
        DEFAULT_CONCURRENT=batch_data.get("DEFAULT_CONCURRENT", 10),
        RETRY_ATTEMPTS=batch_data.get("RETRY_ATTEMPTS", 2),
        RETRY_DELAY_MS=batch_data.get("RETRY_DELAY_MS", 1000),
    )

    # Parse batched mining settings
    batched_data = raw.get("BATCHED", {})
    batched = BatchedConfig(
        TARGET_TOKENS_PER_CHUNK=batched_data.get("TARGET_TOKENS_PER_CHUNK", 5000),
        MIN_CHUNKS=batched_data.get("MIN_CHUNKS", 3),
        MAX_CONCURRENT=batched_data.get("MAX_CONCURRENT", 5),
        MAX_RETRIES=batched_data.get("MAX_RETRIES", 3),
        RETRY_DELAY_MS=batched_data.get("RETRY_DELAY_MS", 1000),
        MIN_CONFIDENCE=batched_data.get("MIN_CONFIDENCE", 0.5),
        MAX_TOKENS=batched_data.get("MAX_TOKENS", 8192),
    )

    # Parse batched mining prompts
    batched_prompt_data = raw.get("BATCHED_PROMPT", {})
    batched_prompt = BatchedPromptConfig(
        SYSTEM=batched_prompt_data.get("SYSTEM", ""),
        USER_TEMPLATE=batched_prompt_data.get("USER_TEMPLATE", ""),
    )

    return ExtractionConfig(
        DEFAULT_EXTRACTOR=raw.get("DEFAULT", {}).get("EXTRACTOR", "cerebras"),
        EXTRACTORS=extractors,
        SCHEMA=schema,
        PROMPT=prompt,
        BATCH=batch,
        BATCHED=batched,
        BATCHED_PROMPT=batched_prompt,
    )


# Singleton config instance
_config: ExtractionConfig | None = None


def get_config() -> ExtractionConfig:
    """Get or load the extraction config singleton."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: Path | None = None) -> ExtractionConfig:
    """Force reload of config from file."""
    global _config
    _config = load_config(config_path)
    return _config
