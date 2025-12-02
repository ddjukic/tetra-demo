"""
Pipeline configuration for the scientific knowledge graph multi-agent system.

This module provides configuration management for all pipeline phases including
STRING network expansion, PubMed search, relationship mining, ML prediction,
and observability settings.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineConfig:
    """
    Configuration settings for the knowledge graph pipeline.

    Controls behavior of all pipeline phases including STRING network expansion,
    PubMed search, LLM relationship mining, ML link prediction, and observability.

    Attributes:
        string_min_score: Minimum STRING confidence score (0-1000). Default 700 for high confidence.
        string_max_partners: Maximum interaction partners per protein to fetch from STRING.
        pubmed_max_results: Maximum number of PubMed articles to retrieve.
        pubmed_date_filter: Optional PubMed date filter (e.g., "2020:2024[pdat]").
        pubmed_species_filter: Optional MeSH species filter (e.g., "humans[MeSH Terms]").
        mining_max_concurrent: Maximum concurrent LLM API calls for relationship mining.
        mining_max_retries: Maximum retry attempts for failed LLM calls.
        mining_base_delay: Base delay in seconds for exponential backoff.
        mining_retry_on_codes: HTTP status codes that trigger retry.
        mining_model: LLM model identifier for relationship mining.
        ml_min_score: Minimum ML prediction score threshold (0-1).
        ml_max_predictions: Maximum number of novel predictions to return.
        langfuse_enabled: Whether to enable Langfuse observability.
        langfuse_session_id: Optional session ID for Langfuse tracing.
    """

    # STRING network expansion settings
    string_min_score: int = 700  # High confidence threshold (0-1000)
    string_max_partners: int = 30  # Max interaction partners per protein

    # PubMed search settings
    pubmed_max_results: int = 50
    pubmed_date_filter: str | None = None  # e.g., "2020:2024[pdat]"
    pubmed_species_filter: str | None = "humans[MeSH Terms]"

    # Relationship mining settings (LLM-based extraction)
    mining_max_concurrent: int = 5
    mining_max_retries: int = 3
    mining_base_delay: float = 1.0
    mining_retry_on_codes: tuple[int, ...] = field(
        default_factory=lambda: (429, 503, 500, 502)
    )
    mining_model: str = "gemini-2.0-flash-exp"

    # ML Link Prediction settings
    ml_min_score: float = 0.7
    ml_max_predictions: int = 20

    # Observability settings
    langfuse_enabled: bool = True
    langfuse_session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "string_min_score": self.string_min_score,
            "string_max_partners": self.string_max_partners,
            "pubmed_max_results": self.pubmed_max_results,
            "pubmed_date_filter": self.pubmed_date_filter,
            "pubmed_species_filter": self.pubmed_species_filter,
            "mining_max_concurrent": self.mining_max_concurrent,
            "mining_max_retries": self.mining_max_retries,
            "mining_base_delay": self.mining_base_delay,
            "mining_retry_on_codes": list(self.mining_retry_on_codes),
            "mining_model": self.mining_model,
            "ml_min_score": self.ml_min_score,
            "ml_max_predictions": self.ml_max_predictions,
            "langfuse_enabled": self.langfuse_enabled,
            "langfuse_session_id": self.langfuse_session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """
        Create a PipelineConfig from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            New PipelineConfig instance with values from the dictionary.
        """
        # Handle tuple conversion for retry codes
        if "mining_retry_on_codes" in data:
            codes = data["mining_retry_on_codes"]
            if isinstance(codes, list):
                data = {**data, "mining_retry_on_codes": tuple(codes)}

        # Filter to only known fields
        known_fields = {
            "string_min_score",
            "string_max_partners",
            "pubmed_max_results",
            "pubmed_date_filter",
            "pubmed_species_filter",
            "mining_max_concurrent",
            "mining_max_retries",
            "mining_base_delay",
            "mining_retry_on_codes",
            "mining_model",
            "ml_min_score",
            "ml_max_predictions",
            "langfuse_enabled",
            "langfuse_session_id",
        }
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """
        Create a PipelineConfig from environment variables.

        Environment variables (all optional with defaults):
            TETRA_STRING_MIN_SCORE: Minimum STRING confidence score
            TETRA_STRING_MAX_PARTNERS: Maximum interaction partners
            TETRA_PUBMED_MAX_RESULTS: Maximum PubMed results
            TETRA_PUBMED_DATE_FILTER: PubMed date filter string
            TETRA_PUBMED_SPECIES_FILTER: PubMed species filter
            TETRA_MINING_MAX_CONCURRENT: Max concurrent LLM calls
            TETRA_MINING_MAX_RETRIES: Max retry attempts
            TETRA_MINING_BASE_DELAY: Base delay for backoff
            TETRA_MINING_MODEL: LLM model identifier
            TETRA_ML_MIN_SCORE: Minimum ML prediction score
            TETRA_ML_MAX_PREDICTIONS: Maximum predictions
            TETRA_LANGFUSE_ENABLED: Enable Langfuse (true/false)
            TETRA_LANGFUSE_SESSION_ID: Langfuse session ID

        Returns:
            New PipelineConfig instance with values from environment.
        """

        def get_int(key: str, default: int) -> int:
            value = os.environ.get(key)
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                return default

        def get_float(key: str, default: float) -> float:
            value = os.environ.get(key)
            if value is None:
                return default
            try:
                return float(value)
            except ValueError:
                return default

        def get_bool(key: str, default: bool) -> bool:
            value = os.environ.get(key)
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes")

        def get_str_or_none(key: str, default: str | None) -> str | None:
            return os.environ.get(key, default) or default

        return cls(
            string_min_score=get_int("TETRA_STRING_MIN_SCORE", 700),
            string_max_partners=get_int("TETRA_STRING_MAX_PARTNERS", 30),
            pubmed_max_results=get_int("TETRA_PUBMED_MAX_RESULTS", 50),
            pubmed_date_filter=get_str_or_none("TETRA_PUBMED_DATE_FILTER", None),
            pubmed_species_filter=get_str_or_none(
                "TETRA_PUBMED_SPECIES_FILTER", "humans[MeSH Terms]"
            ),
            mining_max_concurrent=get_int("TETRA_MINING_MAX_CONCURRENT", 5),
            mining_max_retries=get_int("TETRA_MINING_MAX_RETRIES", 3),
            mining_base_delay=get_float("TETRA_MINING_BASE_DELAY", 1.0),
            mining_model=os.environ.get("TETRA_MINING_MODEL", "gemini-2.0-flash-exp"),
            ml_min_score=get_float("TETRA_ML_MIN_SCORE", 0.7),
            ml_max_predictions=get_int("TETRA_ML_MAX_PREDICTIONS", 20),
            langfuse_enabled=get_bool("TETRA_LANGFUSE_ENABLED", True),
            langfuse_session_id=get_str_or_none("TETRA_LANGFUSE_SESSION_ID", None),
        )

    def with_session_id(self, session_id: str | None = None) -> PipelineConfig:
        """
        Create a copy of this config with a new session ID.

        If no session_id is provided, generates a new UUID.

        Args:
            session_id: Optional session ID. Generated if not provided.

        Returns:
            New PipelineConfig instance with updated session ID.
        """
        return PipelineConfig(
            string_min_score=self.string_min_score,
            string_max_partners=self.string_max_partners,
            pubmed_max_results=self.pubmed_max_results,
            pubmed_date_filter=self.pubmed_date_filter,
            pubmed_species_filter=self.pubmed_species_filter,
            mining_max_concurrent=self.mining_max_concurrent,
            mining_max_retries=self.mining_max_retries,
            mining_base_delay=self.mining_base_delay,
            mining_retry_on_codes=self.mining_retry_on_codes,
            mining_model=self.mining_model,
            ml_min_score=self.ml_min_score,
            ml_max_predictions=self.ml_max_predictions,
            langfuse_enabled=self.langfuse_enabled,
            langfuse_session_id=session_id or str(uuid.uuid4()),
        )

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return (
            f"PipelineConfig(\n"
            f"  STRING: min_score={self.string_min_score}, max_partners={self.string_max_partners}\n"
            f"  PubMed: max_results={self.pubmed_max_results}, "
            f"date_filter={self.pubmed_date_filter}, species_filter={self.pubmed_species_filter}\n"
            f"  Mining: concurrent={self.mining_max_concurrent}, retries={self.mining_max_retries}, "
            f"model={self.mining_model}\n"
            f"  ML: min_score={self.ml_min_score}, max_predictions={self.ml_max_predictions}\n"
            f"  Langfuse: enabled={self.langfuse_enabled}, session={self.langfuse_session_id}\n"
            f")"
        )
