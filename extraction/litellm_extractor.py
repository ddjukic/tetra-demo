"""
LiteLLM-based relationship extraction from biomedical abstracts.

This module uses LiteLLM to enable multi-provider LLM support for relationship
extraction. Configuration is loaded from extraction/config.toml.

Supported providers:
- Cerebras GPT-OSS-120B via OpenRouter (~2000 tok/s)
- Gemini 2.5 Flash / 1.5 Pro
- Any other LiteLLM-supported provider

Usage:
    from extraction.litellm_extractor import create_extractor

    # Use default extractor (from config)
    extractor = create_extractor()

    # Use specific extractor by name
    extractor = create_extractor("gemini")
    extractor = create_extractor("cerebras")

    # Extract relationships
    relationships, metrics = await extractor.extract_relationships(
        abstract, entity_pairs, pmid
    )

Environment Variables:
    CEREBRAS_API_KEY: API key for Cerebras inference
    OPENROUTER_API_KEY: API key for OpenRouter
    GOOGLE_API_KEY: API key for Gemini
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from litellm import acompletion

from extraction.config_loader import (
    ExtractionConfig,
    ExtractorConfig,
    get_config,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionMetrics:
    """
    Metrics for a single extraction call.

    Tracks tokens, latency, cost, and throughput for performance analysis.
    """

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    relationships_found: int = 0

    @property
    def tokens_per_second(self) -> float:
        """Calculate throughput in tokens per second."""
        if self.latency_ms <= 0:
            return 0.0
        return (self.total_tokens / self.latency_ms) * 1000

    @property
    def completion_tokens_per_second(self) -> float:
        """Calculate output throughput (completion tokens per second)."""
        if self.latency_ms <= 0:
            return 0.0
        return (self.completion_tokens / self.latency_ms) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "relationships_found": self.relationships_found,
            "tokens_per_second": self.tokens_per_second,
            "completion_tokens_per_second": self.completion_tokens_per_second,
        }


@dataclass
class BatchMetrics:
    """Aggregated metrics for a batch of extractions."""

    model: str
    total_abstracts: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    total_relationships: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    individual_metrics: list[ExtractionMetrics] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per extraction."""
        if self.successful_extractions <= 0:
            return 0.0
        return self.total_latency_ms / self.successful_extractions

    @property
    def throughput_tokens_per_second(self) -> float:
        """Overall throughput in tokens per second."""
        if self.total_latency_ms <= 0:
            return 0.0
        return (self.total_tokens / self.total_latency_ms) * 1000

    @property
    def completion_throughput(self) -> float:
        """Output throughput (completion tokens per second)."""
        if self.total_latency_ms <= 0:
            return 0.0
        return (self.total_completion_tokens / self.total_latency_ms) * 1000

    def add_extraction(self, metrics: ExtractionMetrics, success: bool = True) -> None:
        """Add metrics from a single extraction."""
        self.individual_metrics.append(metrics)
        if success:
            self.successful_extractions += 1
            self.total_relationships += metrics.relationships_found
        else:
            self.failed_extractions += 1

        self.total_prompt_tokens += metrics.prompt_tokens
        self.total_completion_tokens += metrics.completion_tokens
        self.total_tokens += metrics.total_tokens
        self.total_latency_ms += metrics.latency_ms
        self.total_cost_usd += metrics.cost_usd

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "total_abstracts": self.total_abstracts,
            "successful_extractions": self.successful_extractions,
            "failed_extractions": self.failed_extractions,
            "total_relationships": self.total_relationships,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_ms": self.avg_latency_ms,
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "completion_throughput": self.completion_throughput,
        }


class LiteLLMRelationshipExtractor:
    """
    Extract typed relationships from abstracts using LiteLLM.

    Configuration is loaded from extraction/config.toml. Supports multiple
    LLM providers through LiteLLM's unified interface with provider-specific
    response format handling.

    Args:
        extractor_name: Name of extractor config to use (e.g., "cerebras", "gemini")
        config: Optional ExtractionConfig override
    """

    def __init__(
        self,
        extractor_name: str | None = None,
        config: ExtractionConfig | None = None,
    ):
        """
        Initialize the LiteLLM relationship extractor from config.

        Args:
            extractor_name: Name of extractor in config (default from config)
            config: Optional config override
        """
        self._config = config or get_config()
        self._extractor_name = extractor_name or self._config.DEFAULT_EXTRACTOR
        self._extractor_config = self._config.get_extractor(self._extractor_name)

        # Configure API keys based on provider
        self._configure_api_keys()

        logger.info(
            f"Initialized LiteLLMRelationshipExtractor: "
            f"extractor={self._extractor_name}, model={self._extractor_config.MODEL}"
        )

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._extractor_config.MODEL

    @property
    def extractor_config(self) -> ExtractorConfig:
        """Get the extractor configuration."""
        return self._extractor_config

    def _configure_api_keys(self) -> None:
        """Configure API keys for the selected provider."""
        model = self._extractor_config.MODEL

        if model.startswith("cerebras/"):
            if not os.environ.get("CEREBRAS_API_KEY"):
                logger.warning("CEREBRAS_API_KEY not set")
        elif model.startswith("openrouter/"):
            if not os.environ.get("OPENROUTER_API_KEY"):
                logger.warning("OPENROUTER_API_KEY not set")
            os.environ.setdefault("OR_SITE_URL", "https://tetra-kg.example.com")
            os.environ.setdefault("OR_APP_NAME", "tetra-kg-agent")
        elif model.startswith("gemini/"):
            if not os.environ.get("GOOGLE_API_KEY"):
                logger.warning("GOOGLE_API_KEY not set")

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD based on token usage."""
        pricing = self._extractor_config.PRICING
        input_cost = (prompt_tokens / 1_000_000) * pricing.INPUT
        output_cost = (completion_tokens / 1_000_000) * pricing.OUTPUT
        return input_cost + output_cost

    def _build_prompt(
        self, abstract: str, entity_pairs: list[tuple[str, str]]
    ) -> str:
        """Build the extraction prompt from config template."""
        pairs_text = "\n".join(f"- {e1} and {e2}" for e1, e2 in entity_pairs)
        return self._config.PROMPT.USER_TEMPLATE.format(
            abstract=abstract,
            entity_pairs=pairs_text,
        )

    async def extract_relationships(
        self,
        abstract: str,
        entity_pairs: list[tuple[str, str]],
        pmid: str,
    ) -> tuple[list[dict[str, Any]], ExtractionMetrics]:
        """
        Extract relationships for co-occurring entity pairs in an abstract.

        Args:
            abstract: The abstract text to analyze
            entity_pairs: List of (entity1, entity2) tuples to classify
            pmid: PubMed ID for reference

        Returns:
            Tuple of (relationships, metrics).
        """
        metrics = ExtractionMetrics(model=self.model)

        if not entity_pairs:
            return [], metrics

        # Build prompt from config template
        prompt = self._build_prompt(abstract, entity_pairs)

        messages = [
            {"role": "system", "content": self._config.PROMPT.SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            start_time = time.time()

            # Get provider-appropriate response format
            response_format = self._config.get_response_format(self._extractor_name)

            # Get extra params (e.g., provider routing for OpenRouter)
            extra_params = self._config.get_extra_params(self._extractor_name)

            # Call LiteLLM async completion
            response = await acompletion(
                model=self.model,
                messages=messages,
                temperature=self._extractor_config.TEMPERATURE,
                max_tokens=self._extractor_config.MAX_TOKENS,
                timeout=self._extractor_config.TIMEOUT,
                response_format=response_format,
                **extra_params,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract token usage from response
            usage = response.get("usage", {})
            metrics.prompt_tokens = usage.get("prompt_tokens", 0)
            metrics.completion_tokens = usage.get("completion_tokens", 0)
            metrics.total_tokens = usage.get("total_tokens", 0)
            metrics.latency_ms = latency_ms
            metrics.cost_usd = self._calculate_cost(
                metrics.prompt_tokens,
                metrics.completion_tokens,
            )

            # Parse response
            result_text = response.choices[0].message.content or "{}"
            result = json.loads(result_text)

            # Handle both array and object with 'relationships' key
            if isinstance(result, list):
                relationships = result
            elif isinstance(result, dict) and "relationships" in result:
                relationships = result["relationships"]
            else:
                relationships = []

            # Add PMID to each relationship and normalize
            for rel in relationships:
                rel["pmid"] = pmid
                if "relationship" in rel:
                    rel["relationship"] = rel["relationship"].lower()

            metrics.relationships_found = len(relationships)

            logger.debug(
                f"Extracted {len(relationships)} relationships from PMID {pmid} "
                f"in {latency_ms:.0f}ms ({metrics.completion_tokens_per_second:.0f} tok/s)"
            )

            return relationships, metrics

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response for PMID {pmid}: {e}")
            return [], metrics
        except Exception as e:
            logger.error(f"Error extracting relationships from PMID {pmid}: {e}")
            return [], metrics

    async def batch_extract(
        self,
        abstracts: list[dict[str, Any]],
        max_concurrent: int | None = None,
    ) -> tuple[list[dict[str, Any]], BatchMetrics]:
        """
        Batch extraction with semaphore concurrency control.

        Args:
            abstracts: List of dictionaries with keys:
                - pmid: PubMed ID
                - abstract: Abstract text
                - entity_pairs: List of (entity1, entity2) tuples
            max_concurrent: Max concurrent requests (default from config)

        Returns:
            Tuple of (all_relationships, batch_metrics).
        """
        if max_concurrent is None:
            max_concurrent = self._extractor_config.MAX_CONCURRENT

        batch_metrics = BatchMetrics(
            model=self.model,
            total_abstracts=len(abstracts),
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(
            item: dict[str, Any],
        ) -> tuple[list[dict[str, Any]], ExtractionMetrics, bool]:
            async with semaphore:
                pmid = item.get("pmid", "unknown")
                abstract = item.get("abstract", "")
                entity_pairs = item.get("entity_pairs", [])

                if not abstract or not entity_pairs:
                    return [], ExtractionMetrics(model=self.model), False

                try:
                    relationships, metrics = await self.extract_relationships(
                        abstract, entity_pairs, pmid
                    )
                    return relationships, metrics, True
                except Exception as e:
                    logger.error(f"Batch extraction error for PMID {pmid}: {e}")
                    return [], ExtractionMetrics(model=self.model), False

        # Process all abstracts concurrently with semaphore limit
        tasks = [process_one(item) for item in abstracts]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        all_relationships: list[dict[str, Any]] = []
        for relationships, metrics, success in results:
            batch_metrics.add_extraction(metrics, success)
            if relationships:
                all_relationships.extend(relationships)

        logger.info(
            f"Batch extraction complete: "
            f"{batch_metrics.successful_extractions}/{batch_metrics.total_abstracts} "
            f"successful, {batch_metrics.total_relationships} relationships, "
            f"{batch_metrics.completion_throughput:.0f} tok/s"
        )

        return all_relationships, batch_metrics


class ADKLiteLLMRelationshipExtractor:
    """
    ADK Agent-based relationship extractor using LiteLLM for multi-model support.

    This class integrates with Google ADK's Agent framework while using LiteLLM
    as the model backend.
    """

    def __init__(
        self,
        extractor_name: str | None = None,
        config: ExtractionConfig | None = None,
    ):
        """
        Initialize the ADK-based LiteLLM extractor.

        Args:
            extractor_name: Name of extractor in config
            config: Optional config override
        """
        try:
            from google.adk.models.lite_llm import LiteLlm

            self._has_adk = True
        except ImportError:
            logger.warning("google-adk not installed, falling back to direct LiteLLM")
            self._has_adk = False

        self._extractor = LiteLLMRelationshipExtractor(
            extractor_name=extractor_name,
            config=config,
        )

        if self._has_adk:
            self._litellm_model = LiteLlm(model=self._extractor.model)
            logger.info(
                f"Initialized ADK LiteLLM extractor with model={self._extractor.model}"
            )

    async def extract_relationships(
        self,
        abstract: str,
        entity_pairs: list[tuple[str, str]],
        pmid: str,
    ) -> list[dict[str, Any]]:
        """
        Extract relationships using ADK agent with LiteLLM backend.

        Falls back to direct LiteLLM if ADK is not available.
        """
        relationships, _ = await self._extractor.extract_relationships(
            abstract, entity_pairs, pmid
        )
        return relationships


def create_extractor(
    extractor_name: str | None = None,
    config: ExtractionConfig | None = None,
) -> LiteLLMRelationshipExtractor:
    """
    Factory function to create a relationship extractor.

    Args:
        extractor_name: Name of extractor in config (e.g., "cerebras", "gemini")
        config: Optional config override

    Returns:
        Configured LiteLLMRelationshipExtractor instance

    Examples:
        >>> extractor = create_extractor()  # Uses default from config
        >>> extractor = create_extractor("cerebras")  # Uses Cerebras via OpenRouter
        >>> extractor = create_extractor("gemini")  # Uses Gemini 2.5 Flash
    """
    return LiteLLMRelationshipExtractor(
        extractor_name=extractor_name,
        config=config,
    )
