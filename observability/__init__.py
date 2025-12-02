"""
Observability module for Langfuse tracing integration with Google ADK agents.

This module provides instrumentation for tracing LLM agent operations via
OpenTelemetry, exporting traces to Langfuse for observability and cost tracking.

Example usage:
    from observability import setup_langfuse_tracing, get_tracer, TracingContext

    # Initialize tracing (idempotent - safe to call multiple times)
    if setup_langfuse_tracing():
        print("Langfuse tracing enabled")

    # Get a tracer for custom spans
    tracer = get_tracer("my_module")

    # Use context manager for traced operations
    with TracingContext("my_operation") as span:
        # Your code here
        add_token_usage(span, prompt_tokens=100, completion_tokens=50, model="gemini-2.0-flash")
"""

from observability.tracing import (
    setup_langfuse_tracing,
    get_tracer,
    TracingContext,
    add_token_usage,
    add_cost_estimate,
    is_tracing_enabled,
)

__all__ = [
    "setup_langfuse_tracing",
    "get_tracer",
    "TracingContext",
    "add_token_usage",
    "add_cost_estimate",
    "is_tracing_enabled",
]
