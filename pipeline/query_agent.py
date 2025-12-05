"""
LLM-powered query construction agent for PubMed searches.

This module uses a language model to construct optimal PubMed search queries
from a list of proteins identified in the STRING network expansion phase.

Phase 2 in the pipeline flow:
    STRING Expansion -> Query Construction -> PubMed Search

The agent applies biomedical domain expertise to:
- Use appropriate MeSH terms for accurate filtering
- Group related proteins with boolean operators
- Add date and species filters
- Optimize query length for API compatibility

Example:
    >>> from pipeline import PipelineConfig, PipelineReport
    >>> from pipeline.query_agent import construct_pubmed_query
    >>>
    >>> config = PipelineConfig()
    >>> report = PipelineReport.create()
    >>> result = await construct_pubmed_query(
    ...     proteins=["BRCA1", "BRCA2", "TP53", "ATM"],
    ...     research_focus="DNA damage response and cell cycle regulation",
    ...     config=config,
    ...     report=report,
    ... )
    >>> print(result.query)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from pipeline.config import PipelineConfig
from pipeline.metrics import PhaseMetrics, PipelineReport, TokenUsage

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================


QUERY_CONSTRUCTION_PROMPT = '''You are a biomedical literature search expert specializing in PubMed queries.

Construct an optimal PubMed query based on the user's research focus. Parse the input carefully to identify:
- **Proteins/genes**: Use gene symbols directly (e.g., BRCA1, TP53)
- **Diseases**: Convert to proper MeSH terms (e.g., "alzheimers" → "Alzheimer Disease"[MeSH Terms])
- **Organisms**: Detect species mentions and use correct MeSH (e.g., "mice" → "Mice"[MeSH Terms], NOT "humans")
- **Date ranges**: Parse year mentions (e.g., "2024" → "2024[pdat]", "recent" → last 2 years)

## Critical Rules:
1. **Parse organism from research focus** - if user says "mice", use "Mice"[MeSH Terms], NOT humans
2. **Parse dates from research focus** - if user says "2024", use "2024[pdat]"
3. Use proper MeSH terms for diseases (e.g., "Alzheimer Disease"[MeSH Terms], "Breast Neoplasms"[MeSH Terms])
4. Group related terms with OR, combine groups with AND
5. Keep query under 500 characters
6. If proteins are provided, include the most relevant ones (max 8)
7. Add context terms that enhance specificity

## Common MeSH Mappings:
- alzheimer/alzheimers → "Alzheimer Disease"[MeSH Terms]
- parkinson → "Parkinson Disease"[MeSH Terms]
- cancer → use specific type or "Neoplasms"[MeSH Terms]
- diabetes → "Diabetes Mellitus"[MeSH Terms]
- mice/mouse → "Mice"[MeSH Terms]
- rat/rats → "Rats"[MeSH Terms]
- human/humans → "Humans"[MeSH Terms]
- zebrafish → "Zebrafish"[MeSH Terms]

## Input:
Proteins from STRING network: {proteins}
User's research focus: {focus}
Default date filter (use if no date in focus): {date_filter}
Default species filter (override if species in focus): {species_filter}

## Output:
Return ONLY the PubMed query string. No explanation, no markdown, just the raw query.

## Examples:
Input: "alzheimers mice 2024", proteins: []
Output: "Alzheimer Disease"[MeSH Terms] AND "Mice"[MeSH Terms] AND 2024[pdat]

Input: "BRCA1 breast cancer", proteins: [BRCA1, BRCA2, TP53]
Output: (BRCA1 OR BRCA2 OR TP53) AND "Breast Neoplasms"[MeSH Terms] AND "Humans"[MeSH Terms]

Input: "orexin signaling pathway", proteins: [HCRTR1, HCRTR2, HCRT]
Output: (HCRTR1 OR HCRTR2 OR HCRT) AND (orexin OR hypocretin) AND "Humans"[MeSH Terms] AND 2020:2025[pdat]
'''


QUERY_REFINEMENT_PROMPT = '''The following PubMed query is too long ({length} characters, max 500).
Please shorten it while preserving the most important search terms.

Current query: {query}

Rules:
1. Remove less important proteins (keep top 5-8)
2. Simplify boolean expressions
3. Keep essential MeSH terms and filters
4. Target around 400 characters

Return ONLY the shortened query, no explanation.'''


# =============================================================================
# Result Data Structures
# =============================================================================


@dataclass
class QueryConstructionResult:
    """
    Result of LLM-powered query construction.

    Attributes:
        query: The constructed PubMed query string.
        strategy_explanation: Brief explanation of query strategy (if available).
        token_usage: Token usage metrics from the LLM call.
        was_refined: Whether the query was refined due to length.
        original_length: Original query length before refinement.
    """

    query: str
    strategy_explanation: str
    token_usage: TokenUsage
    was_refined: bool = False
    original_length: int = 0

    @property
    def query_length(self) -> int:
        """Get the length of the final query."""
        return len(self.query)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "query_length": self.query_length,
            "strategy_explanation": self.strategy_explanation,
            "was_refined": self.was_refined,
            "original_length": self.original_length,
            "token_usage": self.token_usage.to_dict(),
        }


# =============================================================================
# Core Query Construction
# =============================================================================


async def construct_pubmed_query(
    proteins: list[str],
    research_focus: str,
    config: PipelineConfig,
    report: PipelineReport,
) -> QueryConstructionResult:
    """
    Use LLM to construct an optimal PubMed query from STRING proteins.

    This is Phase 2 of the knowledge graph pipeline. It takes the expanded
    protein list from STRING and constructs a targeted PubMed search query
    using LLM expertise in biomedical literature search.

    Args:
        proteins: List of protein gene symbols from STRING expansion.
        research_focus: Description of the research focus/question.
        config: Pipeline configuration with LLM settings.
        report: Pipeline report for metrics tracking.

    Returns:
        QueryConstructionResult containing:
            - query: The constructed PubMed query string
            - strategy_explanation: Brief explanation of the strategy
            - token_usage: Metrics from the LLM call

    Raises:
        RuntimeError: If GOOGLE_API_KEY is not set.

    Example:
        >>> result = await construct_pubmed_query(
        ...     proteins=["BRCA1", "TP53", "ATM", "CHEK2"],
        ...     research_focus="breast cancer susceptibility",
        ...     config=PipelineConfig(),
        ...     report=PipelineReport.create(),
        ... )
        >>> print(f"Query ({len(result.query)} chars): {result.query}")
    """
    # Note: proteins can be empty - the LLM will construct query from research_focus alone

    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please set it to use the query construction agent."
        )

    # Start phase metrics
    phase_name = "query_construction"
    metrics = report.start_phase(phase_name)

    logger.info(
        "Constructing PubMed query for %d proteins with focus: %s",
        len(proteins) if proteins else 0,
        research_focus[:50] + "..." if len(research_focus) > 50 else research_focus,
    )

    # Import Google Generative AI
    try:
        import google.generativeai as genai
    except ImportError as e:
        metrics.add_error("google-generativeai package not installed")
        metrics.complete()
        raise RuntimeError(
            "google-generativeai package is required. "
            "Install with: pip install google-generativeai"
        ) from e

    # Configure the API
    genai.configure(api_key=api_key)

    # Build the prompt
    date_filter_text = (
        f"Date filter: {config.pubmed_date_filter}"
        if config.pubmed_date_filter
        else "Date filter: Last 5 years recommended"
    )
    species_filter_text = (
        f"Species filter: {config.pubmed_species_filter}"
        if config.pubmed_species_filter
        else "Species filter: humans[MeSH Terms]"
    )

    # Format proteins list (show "None" if empty so LLM knows to parse from focus)
    proteins_text = ", ".join(proteins) if proteins else "None (construct query from research focus only)"

    prompt = QUERY_CONSTRUCTION_PROMPT.format(
        proteins=proteins_text,
        focus=research_focus,
        date_filter=date_filter_text,
        species_filter=species_filter_text,
    )

    # Make the LLM call
    model = genai.GenerativeModel(config.mining_model)

    start_time = time.time()
    try:
        response = await model.generate_content_async(prompt)
        latency_ms = (time.time() - start_time) * 1000
        metrics.increment_api_calls()

        # Extract the query from response
        query = response.text.strip()

        # Clean up the query (remove markdown if present)
        query = _clean_query(query)

        logger.debug("Initial query (%d chars): %s", len(query), query[:100])

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        error_msg = f"LLM call failed: {e}"
        logger.error(error_msg)
        metrics.add_error(error_msg)
        metrics.complete()

        # Return a fallback query
        fallback_query = _build_fallback_query(proteins, config)
        fallback_usage = TokenUsage.create(
            phase=phase_name,
            step="fallback",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency_ms,
            model=config.mining_model,
        )
        report.add_token_usage(phase_name, fallback_usage)

        return QueryConstructionResult(
            query=fallback_query,
            strategy_explanation="Fallback query due to LLM error",
            token_usage=fallback_usage,
            was_refined=False,
        )

    # Track token usage
    token_usage = TokenUsage.from_response(
        phase=phase_name,
        step="initial_query",
        response=response,
        latency_ms=latency_ms,
        model=config.mining_model,
    )
    report.add_token_usage(phase_name, token_usage)

    # Check if query needs refinement (too long)
    was_refined = False
    original_length = len(query)

    if len(query) > 500:
        logger.info("Query too long (%d chars), requesting refinement", len(query))

        refine_prompt = QUERY_REFINEMENT_PROMPT.format(
            length=len(query),
            query=query,
        )

        start_time = time.time()
        try:
            refine_response = await model.generate_content_async(refine_prompt)
            refine_latency_ms = (time.time() - start_time) * 1000
            metrics.increment_api_calls()

            refined_query = refine_response.text.strip()
            refined_query = _clean_query(refined_query)

            if len(refined_query) <= 500:
                query = refined_query
                was_refined = True
                logger.info("Query refined to %d chars", len(query))

                # Track refinement token usage
                refine_usage = TokenUsage.from_response(
                    phase=phase_name,
                    step="query_refinement",
                    response=refine_response,
                    latency_ms=refine_latency_ms,
                    model=config.mining_model,
                )
                report.add_token_usage(phase_name, refine_usage)

                # Update total token usage
                token_usage = TokenUsage.create(
                    phase=phase_name,
                    step="total",
                    prompt_tokens=token_usage.prompt_tokens + refine_usage.prompt_tokens,
                    completion_tokens=token_usage.completion_tokens + refine_usage.completion_tokens,
                    latency_ms=token_usage.latency_ms + refine_usage.latency_ms,
                    model=config.mining_model,
                )

        except Exception as e:
            logger.warning("Query refinement failed: %s. Using truncated query.", e)
            # Truncate as fallback
            query = _truncate_query(query, 500)
            was_refined = True

    # Final validation - ensure query is usable
    if len(query) > 500:
        query = _truncate_query(query, 500)
        was_refined = True

    # Build strategy explanation
    strategy = _build_strategy_explanation(proteins, research_focus, query)

    metrics.increment_items_processed(1)
    metrics.complete()
    report.end_phase(phase_name)

    result = QueryConstructionResult(
        query=query,
        strategy_explanation=strategy,
        token_usage=token_usage,
        was_refined=was_refined,
        original_length=original_length if was_refined else len(query),
    )

    logger.info(
        "Query construction complete: %d chars, %d tokens used",
        len(query),
        token_usage.total_tokens,
    )

    return result


# =============================================================================
# Helper Functions
# =============================================================================


def _clean_query(query: str) -> str:
    """
    Clean up LLM-generated query by removing markdown and extra formatting.

    Args:
        query: Raw query string from LLM.

    Returns:
        Cleaned query string.
    """
    # Remove markdown code blocks
    if query.startswith("```"):
        lines = query.split("\n")
        # Remove first and last lines if they're code block markers
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        query = "\n".join(lines)

    # Remove backticks
    query = query.strip("`")

    # Remove leading/trailing whitespace and newlines
    query = " ".join(query.split())

    # Remove any "Query:" prefix
    if query.lower().startswith("query:"):
        query = query[6:].strip()

    return query.strip()


def _build_fallback_query(
    proteins: list[str],
    config: PipelineConfig,
) -> str:
    """
    Build a simple fallback query when LLM fails.

    Args:
        proteins: List of protein names.
        config: Pipeline configuration.

    Returns:
        Simple PubMed query string.
    """
    # Take top 8 proteins
    top_proteins = proteins[:8]
    protein_clause = " OR ".join(top_proteins)

    parts = [f"({protein_clause})"]

    if config.pubmed_species_filter:
        parts.append(config.pubmed_species_filter)

    if config.pubmed_date_filter:
        parts.append(config.pubmed_date_filter)

    query = " AND ".join(parts)

    # Truncate if needed
    if len(query) > 500:
        query = _truncate_query(query, 500)

    return query


def _truncate_query(query: str, max_length: int) -> str:
    """
    Truncate a query to fit within length limit.

    Attempts to truncate at a logical boundary (AND operator).

    Args:
        query: Query string to truncate.
        max_length: Maximum allowed length.

    Returns:
        Truncated query string.
    """
    if len(query) <= max_length:
        return query

    # Try to find a good truncation point
    truncated = query[:max_length]

    # Look for last AND to truncate cleanly
    last_and = truncated.rfind(" AND ")
    if last_and > max_length // 2:
        truncated = truncated[:last_and]
    else:
        # Just truncate at max length
        truncated = truncated[:max_length]

    # Ensure balanced parentheses
    open_parens = truncated.count("(")
    close_parens = truncated.count(")")
    if open_parens > close_parens:
        truncated += ")" * (open_parens - close_parens)

    return truncated


def _build_strategy_explanation(
    proteins: list[str],
    research_focus: str,
    query: str,
) -> str:
    """
    Build a brief explanation of the query strategy.

    Args:
        proteins: Original protein list.
        research_focus: Research focus description.
        query: Generated query.

    Returns:
        Strategy explanation string.
    """
    # Count components
    protein_count = sum(1 for p in proteins if p.upper() in query.upper())
    has_mesh = "[mesh" in query.lower()
    has_date = "[pdat]" in query.lower()

    explanation_parts = [
        f"Query targets {protein_count}/{len(proteins)} proteins",
        f"for research focus: {research_focus[:30]}...",
    ]

    if has_mesh:
        explanation_parts.append("Uses MeSH terms for precise filtering")
    if has_date:
        explanation_parts.append("Includes date filter")

    return ". ".join(explanation_parts) + "."


# =============================================================================
# Query Validation
# =============================================================================


def validate_pubmed_query(query: str) -> tuple[bool, list[str]]:
    """
    Validate a PubMed query for common issues.

    Args:
        query: PubMed query string to validate.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues = []

    # Check length
    if len(query) > 500:
        issues.append(f"Query too long: {len(query)} chars (max 500)")

    # Check for empty query
    if not query.strip():
        issues.append("Query is empty")
        return False, issues

    # Check for balanced parentheses
    if query.count("(") != query.count(")"):
        issues.append("Unbalanced parentheses")

    # Check for balanced brackets
    if query.count("[") != query.count("]"):
        issues.append("Unbalanced brackets")

    # Check for common typos in field tags
    # Note: Some field tags like [MeSH Terms] or [MeSH Major Topic] have additional text
    simple_fields = ["pdat", "tiab", "auth", "jour"]  # Fields that are just [field]
    for field in simple_fields:
        # Check for typo like "[pdta]" instead of "[pdat]"
        if f"[{field}" in query.lower() and f"[{field}]" not in query.lower():
            issues.append(f"Possible malformed field tag: [{field}]")

    # Check for mesh tags specifically (they can be [MeSH], [MeSH Terms], etc.)
    if "[mesh" in query.lower():
        # Ensure bracket is closed somewhere after [mesh
        mesh_start = query.lower().find("[mesh")
        rest_of_query = query[mesh_start:]
        if "]" not in rest_of_query:
            issues.append("Possible malformed MeSH field tag: missing closing bracket")

    # Check for overly complex queries (too many operators)
    and_count = query.lower().count(" and ")
    or_count = query.lower().count(" or ")
    if and_count + or_count > 20:
        issues.append(f"Query may be too complex: {and_count} ANDs, {or_count} ORs")

    return len(issues) == 0, issues
