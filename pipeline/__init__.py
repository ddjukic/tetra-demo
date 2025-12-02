"""
Pipeline infrastructure for the scientific knowledge graph multi-agent system.

This package provides configuration management, metrics tracking, reporting,
and orchestration for the knowledge graph construction pipeline phases.

Modules:
    config: Pipeline configuration with PipelineConfig dataclass
    metrics: Token usage tracking, phase metrics, and pipeline reporting
    string_expansion: STRING database network expansion (Phase 1)
    query_agent: LLM-powered PubMed query construction (Phase 2)
    parallel_extraction: Parallel co-occurrence and LLM relationship extraction

Example:
    >>> from pipeline import PipelineConfig, PipelineReport, TokenUsage
    >>>
    >>> # Create configuration from environment
    >>> config = PipelineConfig.from_env()
    >>>
    >>> # Start a pipeline report
    >>> report = PipelineReport.create()
    >>> metrics = report.start_phase("string_expansion")
    >>>
    >>> # Track token usage
    >>> usage = TokenUsage.create(
    ...     phase="relationship_mining",
    ...     step="paper_1",
    ...     prompt_tokens=1000,
    ...     completion_tokens=250,
    ...     latency_ms=450.5,
    ...     model="gemini-2.0-flash-exp"
    ... )
    >>> report.add_token_usage("relationship_mining", usage)
    >>>
    >>> # Finalize and get summary
    >>> report.finalize()
    >>> print(report.summary_text())

Pipeline Flow Example:
    >>> from pipeline import PipelineConfig, PipelineReport
    >>> from pipeline.string_expansion import expand_string_network
    >>> from pipeline.query_agent import construct_pubmed_query
    >>>
    >>> config = PipelineConfig()
    >>> report = PipelineReport.create()
    >>>
    >>> # Phase 1: Expand protein network via STRING
    >>> expansion = await expand_string_network(
    ...     seed_proteins=["BRCA1", "TP53"],
    ...     config=config,
    ...     report=report,
    ... )
    >>>
    >>> # Phase 2: Construct optimized PubMed query
    >>> query_result = await construct_pubmed_query(
    ...     proteins=expansion.expanded_proteins,
    ...     research_focus="DNA damage response",
    ...     config=config,
    ...     report=report,
    ... )

Parallel Extraction Example:
    >>> from pipeline import (
    ...     run_parallel_extraction,
    ...     CoOccurrenceEdge,
    ...     ExtractionResult,
    ...     annotations_list_to_dict,
    ... )
    >>>
    >>> # Run parallel extraction (co-occurrence + LLM)
    >>> result = await run_parallel_extraction(
    ...     articles=articles,
    ...     annotations=annotations_dict,
    ...     config=config,
    ...     report=report,
    ... )
    >>> print(f"Found {len(result.cooccurrence_edges)} co-occurrence edges")
    >>> print(f"Found {len(result.llm_relationships)} LLM relationships")
"""

from pipeline.config import PipelineConfig
from pipeline.metrics import (
    MODEL_PRICING,
    PhaseMetrics,
    PipelineReport,
    TokenUsage,
    estimate_cost,
)
from pipeline.parallel_extraction import (
    CoOccurrenceEdge,
    ExtractionResult,
    annotations_list_to_dict,
    extract_cooccurrences,
    extract_relationships_parallel,
    merge_cooccurrence_and_llm_edges,
    run_parallel_extraction,
)
from pipeline.query_agent import (
    QueryConstructionResult,
    construct_pubmed_query,
    validate_pubmed_query,
)
from pipeline.string_expansion import (
    STRINGExpansionResult,
    expand_string_network,
    extract_high_confidence_partners,
    get_interaction_summary,
)
from pipeline.merge import (
    MergeResult,
    merge_to_knowledge_graph,
    calculate_edge_confidence,
    deduplicate_edges,
    get_merge_summary,
    filter_high_confidence_edges,
)

__all__ = [
    # Configuration
    "PipelineConfig",
    # Metrics
    "TokenUsage",
    "PhaseMetrics",
    "PipelineReport",
    # Utilities
    "estimate_cost",
    "MODEL_PRICING",
    # Phase 1: STRING Expansion
    "STRINGExpansionResult",
    "expand_string_network",
    "extract_high_confidence_partners",
    "get_interaction_summary",
    # Phase 2: Query Construction
    "QueryConstructionResult",
    "construct_pubmed_query",
    "validate_pubmed_query",
    # Parallel Extraction
    "CoOccurrenceEdge",
    "ExtractionResult",
    "run_parallel_extraction",
    "extract_cooccurrences",
    "extract_relationships_parallel",
    "merge_cooccurrence_and_llm_edges",
    "annotations_list_to_dict",
    # Phase 5: Graph Merge
    "MergeResult",
    "merge_to_knowledge_graph",
    "calculate_edge_confidence",
    "deduplicate_edges",
    "get_merge_summary",
    "filter_high_confidence_edges",
]
