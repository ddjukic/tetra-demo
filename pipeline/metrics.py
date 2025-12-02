"""
Metrics tracking and reporting for the knowledge graph pipeline.

This module provides comprehensive metrics collection for:
- Token usage tracking per LLM call
- Phase-level metrics aggregation
- Pipeline-wide reporting with timing, costs, and key findings
- Cost estimation based on model pricing
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# =============================================================================
# Model Pricing Configuration (USD per 1M tokens)
# =============================================================================

MODEL_PRICING: dict[str, dict[str, float]] = {
    # Gemini 2.0 Flash
    "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    # Gemini 2.5 Flash (for budgeting - higher tier)
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-preview-05-20": {"input": 0.15, "output": 0.60},
    # Gemini 1.5 Flash (legacy)
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-latest": {"input": 0.075, "output": 0.30},
    # Gemini 1.5 Pro
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-pro-latest": {"input": 1.25, "output": 5.00},
    # Default fallback (conservative estimate)
    "default": {"input": 0.15, "output": 0.60},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate cost in USD based on model pricing.

    Uses known pricing for supported models, falls back to conservative
    estimates for unknown models.

    Args:
        model: Model identifier (e.g., "gemini-2.0-flash-exp").
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.

    Returns:
        Estimated cost in USD.

    Examples:
        >>> estimate_cost("gemini-2.0-flash-exp", 1000, 500)
        0.000225
        >>> estimate_cost("gemini-2.5-flash", 10000, 2000)
        0.0027
    """
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


# =============================================================================
# Token Usage Tracking
# =============================================================================


@dataclass
class TokenUsage:
    """
    Token usage tracking for a single LLM API call.

    Captures detailed metrics about token consumption, latency,
    model used, and estimated cost for individual LLM requests.

    Attributes:
        phase: Pipeline phase name (e.g., "query_construction", "relationship_mining").
        step: Specific step within the phase (e.g., "paper_1", "abstract_15").
        prompt_tokens: Number of input/prompt tokens used.
        completion_tokens: Number of output/completion tokens generated.
        total_tokens: Total tokens (prompt + completion).
        latency_ms: Request latency in milliseconds.
        model: Model identifier used for the request.
        cost_usd: Estimated cost in USD.
    """

    phase: str
    step: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str
    cost_usd: float

    @classmethod
    def from_response(
        cls,
        phase: str,
        step: str,
        response: Any,
        latency_ms: float,
        model: str | None = None,
    ) -> TokenUsage:
        """
        Create TokenUsage from an LLM response object.

        Supports Google Generative AI response format with usage_metadata.
        Falls back to zero values if token counts are not available.

        Args:
            phase: Pipeline phase name.
            step: Step identifier within the phase.
            response: LLM response object (must have usage_metadata attribute).
            latency_ms: Request latency in milliseconds.
            model: Optional model override. If not provided, attempts to extract from response.

        Returns:
            TokenUsage instance with extracted metrics.
        """
        # Extract token counts from Google GenAI response
        prompt_tokens = 0
        completion_tokens = 0

        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

        total_tokens = prompt_tokens + completion_tokens

        # Try to extract model from response if not provided
        if model is None:
            # Fallback to default
            model = "gemini-2.0-flash-exp"

        cost_usd = estimate_cost(model, prompt_tokens, completion_tokens)

        return cls(
            phase=phase,
            step=step,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            model=model,
            cost_usd=cost_usd,
        )

    @classmethod
    def create(
        cls,
        phase: str,
        step: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        model: str,
    ) -> TokenUsage:
        """
        Create TokenUsage with explicit token counts.

        Convenience factory method when token counts are known directly.

        Args:
            phase: Pipeline phase name.
            step: Step identifier within the phase.
            prompt_tokens: Number of input/prompt tokens.
            completion_tokens: Number of output/completion tokens.
            latency_ms: Request latency in milliseconds.
            model: Model identifier.

        Returns:
            TokenUsage instance with calculated cost.
        """
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = estimate_cost(model, prompt_tokens, completion_tokens)

        return cls(
            phase=phase,
            step=step,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            model=model,
            cost_usd=cost_usd,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase": self.phase,
            "step": self.step,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "cost_usd": self.cost_usd,
        }


# =============================================================================
# Phase Metrics
# =============================================================================


@dataclass
class PhaseMetrics:
    """
    Metrics aggregation for a single pipeline phase.

    Tracks timing, token usage, API calls, and errors for one phase
    of the pipeline (e.g., STRING expansion, relationship mining).

    Attributes:
        phase_name: Name of the pipeline phase.
        start_time: Unix timestamp when phase started.
        end_time: Unix timestamp when phase ended (None if still running).
        token_usages: List of TokenUsage records for all LLM calls in this phase.
        api_calls: Count of API calls made (including non-LLM calls).
        items_processed: Count of items processed (papers, entities, etc.).
        errors: List of error messages encountered.
    """

    phase_name: str
    start_time: float
    end_time: float | None = None
    token_usages: list[TokenUsage] = field(default_factory=list)
    api_calls: int = 0
    items_processed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        """
        Calculate phase duration in seconds.

        Returns:
            Duration in seconds. Returns 0 if phase hasn't ended.
        """
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def total_tokens(self) -> int:
        """
        Calculate total tokens used across all LLM calls.

        Returns:
            Sum of all token counts in this phase.
        """
        return sum(usage.total_tokens for usage in self.token_usages)

    @property
    def total_prompt_tokens(self) -> int:
        """Calculate total prompt tokens used."""
        return sum(usage.prompt_tokens for usage in self.token_usages)

    @property
    def total_completion_tokens(self) -> int:
        """Calculate total completion tokens used."""
        return sum(usage.completion_tokens for usage in self.token_usages)

    @property
    def total_cost_usd(self) -> float:
        """
        Calculate total cost in USD for this phase.

        Returns:
            Sum of all costs from LLM calls in this phase.
        """
        return sum(usage.cost_usd for usage in self.token_usages)

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency across all LLM calls."""
        if not self.token_usages:
            return 0.0
        return sum(u.latency_ms for u in self.token_usages) / len(self.token_usages)

    def add_token_usage(self, usage: TokenUsage) -> None:
        """Add a token usage record to this phase."""
        self.token_usages.append(usage)

    def add_error(self, error: str) -> None:
        """Record an error that occurred during this phase."""
        self.errors.append(error)

    def increment_api_calls(self, count: int = 1) -> None:
        """Increment the API call counter."""
        self.api_calls += count

    def increment_items_processed(self, count: int = 1) -> None:
        """Increment the items processed counter."""
        self.items_processed += count

    def complete(self) -> None:
        """Mark this phase as complete with current timestamp."""
        self.end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase_name": self.phase_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.duration_s,
            "token_usages": [u.to_dict() for u in self.token_usages],
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": self.total_cost_usd,
            "api_calls": self.api_calls,
            "items_processed": self.items_processed,
            "errors": self.errors,
        }


# =============================================================================
# Pipeline Report
# =============================================================================


@dataclass
class PipelineReport:
    """
    Comprehensive report for a complete pipeline run.

    Aggregates metrics from all phases, tracks key findings,
    and provides summary statistics for the entire pipeline execution.

    Attributes:
        started_at: Datetime when pipeline started.
        completed_at: Datetime when pipeline completed (None if still running).
        total_duration_s: Total pipeline duration in seconds.
        phase_metrics: Dictionary mapping phase names to PhaseMetrics.
        nodes_created: Count of nodes created in the knowledge graph.
        edges_created: Count of edges created in the knowledge graph.
        papers_processed: Count of papers processed from PubMed.
        relationships_extracted: Count of relationships extracted by LLM.
        predictions_made: Count of ML link predictions.
        top_entities: List of (entity_id, centrality_score) tuples for key entities.
        novel_predictions: List of novel ML predictions (no literature support).
        communities: List of community member lists from community detection.
    """

    # Timing
    started_at: datetime
    completed_at: datetime | None = None
    total_duration_s: float = 0.0
    phase_metrics: dict[str, PhaseMetrics] = field(default_factory=dict)

    # Results summary
    nodes_created: int = 0
    edges_created: int = 0
    papers_processed: int = 0
    relationships_extracted: int = 0
    predictions_made: int = 0

    # Key findings (populated at end)
    top_entities: list[tuple[str, float]] = field(default_factory=list)
    novel_predictions: list[dict[str, Any]] = field(default_factory=list)
    communities: list[list[str]] = field(default_factory=list)

    @classmethod
    def create(cls) -> PipelineReport:
        """
        Create a new PipelineReport with current timestamp.

        Returns:
            New PipelineReport instance ready for metrics collection.
        """
        return cls(started_at=datetime.now())

    def start_phase(self, phase_name: str) -> PhaseMetrics:
        """
        Start tracking a new pipeline phase.

        Creates and registers a PhaseMetrics instance for the given phase.

        Args:
            phase_name: Name of the phase to start.

        Returns:
            The newly created PhaseMetrics instance for this phase.
        """
        metrics = PhaseMetrics(
            phase_name=phase_name,
            start_time=time.time(),
        )
        self.phase_metrics[phase_name] = metrics
        return metrics

    def end_phase(self, phase_name: str) -> None:
        """
        Mark a pipeline phase as complete.

        Records the end timestamp for the specified phase.

        Args:
            phase_name: Name of the phase to complete.
        """
        if phase_name in self.phase_metrics:
            self.phase_metrics[phase_name].complete()

    def get_phase(self, phase_name: str) -> PhaseMetrics | None:
        """
        Get metrics for a specific phase.

        Args:
            phase_name: Name of the phase.

        Returns:
            PhaseMetrics for the phase, or None if not found.
        """
        return self.phase_metrics.get(phase_name)

    def add_token_usage(self, phase_name: str, usage: TokenUsage) -> None:
        """
        Add a token usage record to a specific phase.

        If the phase doesn't exist, it will be created.

        Args:
            phase_name: Name of the phase.
            usage: TokenUsage record to add.
        """
        if phase_name not in self.phase_metrics:
            self.start_phase(phase_name)
        self.phase_metrics[phase_name].add_token_usage(usage)

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used across all phases."""
        return sum(phase.total_tokens for phase in self.phase_metrics.values())

    @property
    def total_prompt_tokens(self) -> int:
        """Calculate total prompt tokens across all phases."""
        return sum(phase.total_prompt_tokens for phase in self.phase_metrics.values())

    @property
    def total_completion_tokens(self) -> int:
        """Calculate total completion tokens across all phases."""
        return sum(
            phase.total_completion_tokens for phase in self.phase_metrics.values()
        )

    @property
    def total_cost_usd(self) -> float:
        """Calculate total cost in USD across all phases."""
        return sum(phase.total_cost_usd for phase in self.phase_metrics.values())

    @property
    def total_api_calls(self) -> int:
        """Calculate total API calls across all phases."""
        return sum(phase.api_calls for phase in self.phase_metrics.values())

    @property
    def total_errors(self) -> list[str]:
        """Collect all errors from all phases."""
        errors = []
        for phase in self.phase_metrics.values():
            for error in phase.errors:
                errors.append(f"[{phase.phase_name}] {error}")
        return errors

    def finalize(self) -> None:
        """
        Finalize the pipeline report.

        Marks the pipeline as complete and calculates total duration.
        Should be called when all phases have finished.
        """
        self.completed_at = datetime.now()
        self.total_duration_s = (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert report to dictionary for JSON serialization.

        Returns:
            Complete dictionary representation of the report.
        """
        return {
            "timing": {
                "started_at": self.started_at.isoformat(),
                "completed_at": (
                    self.completed_at.isoformat() if self.completed_at else None
                ),
                "total_duration_s": self.total_duration_s,
            },
            "phases": {
                name: metrics.to_dict()
                for name, metrics in self.phase_metrics.items()
            },
            "results": {
                "nodes_created": self.nodes_created,
                "edges_created": self.edges_created,
                "papers_processed": self.papers_processed,
                "relationships_extracted": self.relationships_extracted,
                "predictions_made": self.predictions_made,
            },
            "tokens": {
                "total": self.total_tokens,
                "prompt": self.total_prompt_tokens,
                "completion": self.total_completion_tokens,
            },
            "cost_usd": self.total_cost_usd,
            "api_calls": self.total_api_calls,
            "errors": self.total_errors,
            "findings": {
                "top_entities": self.top_entities,
                "novel_predictions": self.novel_predictions,
                "communities_count": len(self.communities),
            },
        }

    def summary_text(self) -> str:
        """
        Generate a human-readable summary of the pipeline run.

        Returns:
            Formatted string with key metrics and findings.
        """
        lines = [
            "=" * 60,
            "PIPELINE EXECUTION REPORT",
            "=" * 60,
            "",
            "TIMING",
            "-" * 40,
            f"  Started:  {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if self.completed_at:
            lines.append(
                f"  Completed: {self.completed_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            lines.append(f"  Duration: {self.total_duration_s:.2f}s")
        else:
            lines.append("  Status: In Progress")

        lines.extend(
            [
                "",
                "RESULTS",
                "-" * 40,
                f"  Nodes Created: {self.nodes_created}",
                f"  Edges Created: {self.edges_created}",
                f"  Papers Processed: {self.papers_processed}",
                f"  Relationships Extracted: {self.relationships_extracted}",
                f"  ML Predictions: {self.predictions_made}",
                "",
                "TOKEN USAGE",
                "-" * 40,
                f"  Prompt Tokens: {self.total_prompt_tokens:,}",
                f"  Completion Tokens: {self.total_completion_tokens:,}",
                f"  Total Tokens: {self.total_tokens:,}",
                f"  Estimated Cost: ${self.total_cost_usd:.4f}",
                "",
                "PHASE BREAKDOWN",
                "-" * 40,
            ]
        )

        for name, metrics in self.phase_metrics.items():
            lines.append(
                f"  {name}:"
                f" {metrics.duration_s:.2f}s, "
                f"{metrics.total_tokens:,} tokens, "
                f"${metrics.total_cost_usd:.4f}"
            )
            if metrics.errors:
                lines.append(f"    Errors: {len(metrics.errors)}")

        if self.top_entities:
            lines.extend(
                [
                    "",
                    "TOP ENTITIES (by centrality)",
                    "-" * 40,
                ]
            )
            for entity, score in self.top_entities[:5]:
                lines.append(f"  {entity}: {score:.4f}")

        if self.novel_predictions:
            lines.extend(
                [
                    "",
                    f"NOVEL PREDICTIONS ({len(self.novel_predictions)} found)",
                    "-" * 40,
                ]
            )
            for pred in self.novel_predictions[:3]:
                source = pred.get("source", "?")
                target = pred.get("target", "?")
                score = pred.get("ml_score", 0)
                lines.append(f"  {source} -> {target} (score: {score:.2f})")

        if self.total_errors:
            lines.extend(
                [
                    "",
                    f"ERRORS ({len(self.total_errors)})",
                    "-" * 40,
                ]
            )
            for error in self.total_errors[:5]:
                lines.append(f"  {error}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        status = "completed" if self.completed_at else "in_progress"
        return (
            f"PipelineReport({status}, "
            f"duration={self.total_duration_s:.1f}s, "
            f"nodes={self.nodes_created}, "
            f"edges={self.edges_created}, "
            f"cost=${self.total_cost_usd:.4f})"
        )
