"""
Langfuse tracing module for Google ADK agents via OpenTelemetry.

This module provides instrumentation for tracing LLM agent operations,
exporting traces to Langfuse for observability, debugging, and cost tracking.

The implementation uses OpenTelemetry with the OTLP HTTP exporter to send
traces to Langfuse's OpenTelemetry endpoint. It also integrates with
OpenInference's Google ADK instrumentor for automatic agent tracing.

Gemini 2.0 Flash Pricing (as of 2025):
    - Input: $0.075 per 1M tokens
    - Output: $0.30 per 1M tokens
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator, Optional

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

logger = logging.getLogger(__name__)

# Module-level state for idempotent initialization
_initialized: bool = False
_init_lock: threading.Lock = threading.Lock()
_tracer_provider: Any = None

# Gemini 2.0 Flash pricing per 1M tokens (USD)
GEMINI_2_0_FLASH_INPUT_COST_PER_MILLION: float = 0.075
GEMINI_2_0_FLASH_OUTPUT_COST_PER_MILLION: float = 0.30

# Model pricing lookup table (cost per 1M tokens)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash": {
        "input": GEMINI_2_0_FLASH_INPUT_COST_PER_MILLION,
        "output": GEMINI_2_0_FLASH_OUTPUT_COST_PER_MILLION,
    },
    "gemini-2.0-flash-exp": {
        "input": GEMINI_2_0_FLASH_INPUT_COST_PER_MILLION,
        "output": GEMINI_2_0_FLASH_OUTPUT_COST_PER_MILLION,
    },
    "gemini-2.5-flash": {
        "input": 0.15,  # Estimate for 2.5 flash
        "output": 0.60,
    },
    "gemini-1.5-flash": {
        "input": 0.075,
        "output": 0.30,
    },
    "gemini-1.5-pro": {
        "input": 1.25,
        "output": 5.00,
    },
}

# Default pricing for unknown models (use flash pricing as fallback)
DEFAULT_PRICING: dict[str, float] = {
    "input": GEMINI_2_0_FLASH_INPUT_COST_PER_MILLION,
    "output": GEMINI_2_0_FLASH_OUTPUT_COST_PER_MILLION,
}


def is_tracing_enabled() -> bool:
    """
    Check if Langfuse tracing has been initialized.

    Returns:
        True if tracing is enabled, False otherwise.
    """
    return _initialized


def setup_langfuse_tracing() -> bool:
    """
    Configure Langfuse tracing for ADK agents via OpenTelemetry.

    This function is idempotent - calling it multiple times is safe and will
    only initialize tracing once. Subsequent calls return the initialization
    status without re-initializing.

    Environment variables required:
        LANGFUSE_SECRET_KEY: Your Langfuse secret key
        LANGFUSE_PUBLIC_KEY: Your Langfuse public key
        LANGFUSE_BASE_URL: Langfuse API URL (default: https://cloud.langfuse.com)

    Returns:
        True if tracing was successfully initialized, False otherwise.

    Example:
        >>> if setup_langfuse_tracing():
        ...     print("Tracing enabled")
        ... else:
        ...     print("Tracing disabled - check credentials")
    """
    global _initialized, _tracer_provider

    # Fast path: already initialized
    if _initialized:
        logger.debug("Langfuse tracing already initialized")
        return True

    # Thread-safe initialization
    with _init_lock:
        # Double-check after acquiring lock
        if _initialized:
            return True

        langfuse_secret = os.environ.get("LANGFUSE_SECRET_KEY")
        langfuse_public = os.environ.get("LANGFUSE_PUBLIC_KEY")
        langfuse_url = os.environ.get(
            "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
        )

        if not langfuse_secret or not langfuse_public:
            logger.info(
                "Langfuse credentials not found in environment variables. "
                "Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY to enable tracing."
            )
            return False

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            # Build OTLP endpoint for Langfuse
            otlp_endpoint = f"{langfuse_url}/api/public/otel/v1/traces"

            # Build Basic Auth header
            auth_string = f"{langfuse_public}:{langfuse_secret}"
            auth_bytes = base64.b64encode(auth_string.encode()).decode()

            # Create OTLP exporter with Langfuse authentication
            exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                headers={"Authorization": f"Basic {auth_bytes}"},
            )

            # Configure tracer provider
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            _tracer_provider = provider

            # Instrument Google ADK if available
            _instrument_google_adk()

            _initialized = True
            logger.info(
                f"Langfuse tracing initialized successfully. "
                f"Endpoint: {langfuse_url}"
            )
            return True

        except ImportError as e:
            logger.warning(
                f"OpenTelemetry dependencies not installed: {e}. "
                "Install with: pip install opentelemetry-sdk "
                "opentelemetry-exporter-otlp-proto-http"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse tracing: {e}")
            return False


def _instrument_google_adk() -> None:
    """
    Instrument Google ADK for automatic tracing.

    This enables automatic capture of agent executions, tool calls,
    and LLM interactions without manual instrumentation.
    """
    try:
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor

        GoogleADKInstrumentor().instrument()
        logger.debug("Google ADK instrumentation enabled")
    except ImportError:
        logger.debug(
            "openinference-instrumentation-google-adk not installed. "
            "ADK auto-instrumentation disabled. Install with: "
            "pip install openinference-instrumentation-google-adk"
        )
    except Exception as e:
        logger.warning(f"Failed to instrument Google ADK: {e}")


def get_tracer(name: str = __name__) -> "Tracer":
    """
    Get an OpenTelemetry tracer for creating custom spans.

    If tracing is not initialized, returns a no-op tracer that creates
    no-op spans (operations are safely ignored).

    Args:
        name: Name for the tracer, typically the module name.
              Defaults to the current module.

    Returns:
        An OpenTelemetry Tracer instance.

    Example:
        >>> tracer = get_tracer("my_module")
        >>> with tracer.start_as_current_span("my_operation") as span:
        ...     span.set_attribute("custom_key", "value")
        ...     # Your code here
    """
    from opentelemetry import trace

    return trace.get_tracer(name)


@contextmanager
def TracingContext(
    name: str,
    attributes: Optional[dict[str, Any]] = None,
    tracer_name: str = __name__,
) -> Generator["Span", None, None]:
    """
    Context manager for wrapping operations with tracing.

    Creates a span for the operation and automatically handles
    start/end timing, error recording, and attribute setting.

    Args:
        name: Name of the operation being traced.
        attributes: Optional dictionary of attributes to set on the span.
        tracer_name: Name for the tracer. Defaults to current module.

    Yields:
        The active Span object for adding additional attributes or events.

    Example:
        >>> with TracingContext("process_document", {"doc_id": "123"}) as span:
        ...     result = process(document)
        ...     span.set_attribute("result_length", len(result))
        ...     add_token_usage(span, prompt_tokens=100, completion_tokens=50)
    """
    tracer = get_tracer(tracer_name)

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            # Record exception details on the span
            span.set_attribute("error", True)
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            span.record_exception(e)
            raise


def add_token_usage(
    span: "Span",
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gemini-2.0-flash",
) -> None:
    """
    Add token usage metrics to a span.

    This adds standard GenAI semantic convention attributes for token
    usage tracking, which Langfuse uses for cost analysis and monitoring.

    Args:
        span: The span to add attributes to.
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.
        model: The model name for cost calculation.

    Example:
        >>> with TracingContext("llm_call") as span:
        ...     response = call_llm(prompt)
        ...     add_token_usage(span,
        ...         prompt_tokens=response.usage.prompt_tokens,
        ...         completion_tokens=response.usage.completion_tokens,
        ...         model="gemini-2.0-flash"
        ...     )
    """
    total_tokens = prompt_tokens + completion_tokens

    # Set standard GenAI semantic convention attributes
    span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
    span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
    span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
    span.set_attribute("gen_ai.request.model", model)

    # Also set Langfuse-specific attributes for compatibility
    span.set_attribute("llm.token_count.prompt", prompt_tokens)
    span.set_attribute("llm.token_count.completion", completion_tokens)
    span.set_attribute("llm.token_count.total", total_tokens)
    span.set_attribute("llm.model_name", model)

    # Calculate and add cost estimate
    add_cost_estimate(
        span,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
    )


def add_cost_estimate(
    span: "Span",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model: str = "gemini-2.0-flash",
) -> float:
    """
    Estimate and add cost attribute to a span based on token usage.

    Calculates cost based on model-specific pricing. If the model is
    not in the pricing table, falls back to Gemini 2.0 Flash pricing.

    Args:
        span: The span to add the cost attribute to.
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.
        model: The model name for pricing lookup.

    Returns:
        The estimated cost in USD.

    Example:
        >>> with TracingContext("llm_call") as span:
        ...     cost = add_cost_estimate(span,
        ...         prompt_tokens=1000,
        ...         completion_tokens=500,
        ...         model="gemini-2.0-flash"
        ...     )
        ...     print(f"Estimated cost: ${cost:.6f}")
    """
    # Get pricing for the model
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_cost_per_million = pricing.get("input", DEFAULT_PRICING["input"])
    output_cost_per_million = pricing.get("output", DEFAULT_PRICING["output"])

    # Calculate cost
    input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million
    output_cost = (completion_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost

    # Set cost attributes
    span.set_attribute("gen_ai.usage.cost", total_cost)
    span.set_attribute("gen_ai.usage.cost_currency", "USD")
    span.set_attribute("llm.cost", total_cost)

    # Set detailed cost breakdown
    span.set_attribute("cost.input_tokens", input_cost)
    span.set_attribute("cost.output_tokens", output_cost)
    span.set_attribute("cost.total", total_cost)
    span.set_attribute("cost.model", model)

    return total_cost


def get_model_pricing(model: str) -> dict[str, float]:
    """
    Get the pricing for a specific model.

    Args:
        model: The model name.

    Returns:
        Dictionary with 'input' and 'output' costs per 1M tokens.

    Example:
        >>> pricing = get_model_pricing("gemini-2.0-flash")
        >>> print(f"Input: ${pricing['input']}/1M tokens")
    """
    return MODEL_PRICING.get(model, DEFAULT_PRICING).copy()


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gemini-2.0-flash",
) -> float:
    """
    Estimate the cost for a given token usage without a span.

    Utility function for cost estimation without requiring an active span.

    Args:
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.
        model: The model name for pricing lookup.

    Returns:
        The estimated cost in USD.

    Example:
        >>> cost = estimate_cost(1000, 500, "gemini-2.0-flash")
        >>> print(f"Estimated cost: ${cost:.6f}")
    """
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
