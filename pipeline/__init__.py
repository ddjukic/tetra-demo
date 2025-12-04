"""
Pipeline infrastructure for the scientific knowledge graph multi-agent system.

This package provides configuration management, metrics tracking, reporting,
and orchestration for the knowledge graph construction pipeline phases.

Hybrid Architecture:
    The pipeline now uses a hybrid architecture that separates:
    - Data Fetching (LLM-driven): DataFetchAgent constructs queries intelligently
    - Data Processing (Deterministic): KGPipeline runs extraction and graph building

    The KGOrchestrator coordinates both components.

Modules:
    config: Pipeline configuration with PipelineConfig dataclass
    metrics: Token usage tracking, phase metrics, and pipeline reporting
    models: Data models for pipeline input/output (PipelineInput, PipelineResult)
    kg_pipeline: Deterministic KG processing pipeline
    orchestrator: Coordination of data fetching and processing
    string_expansion: STRING database network expansion (Phase 1)
    query_agent: LLM-powered PubMed query construction (Phase 2)
    parallel_extraction: Parallel co-occurrence and LLM relationship extraction

Example - Hybrid Architecture:
    >>> from pipeline import KGOrchestrator, PipelineInput
    >>>
    >>> # Create orchestrator
    >>> orchestrator = KGOrchestrator(extractor_name="cerebras")
    >>>
    >>> # Build a knowledge graph
    >>> graph = await orchestrator.build("Build a KG for the orexin pathway")
    >>> print(f"Built graph with {graph.to_summary()['node_count']} nodes")
    >>>
    >>> # Expand the graph
    >>> await orchestrator.expand("Add narcolepsy disease associations")

Example - Direct Pipeline Usage:
    >>> from pipeline import KGPipeline, PipelineInput
    >>>
    >>> # Create input with pre-fetched data
    >>> input_data = PipelineInput(
    ...     articles=articles,
    ...     annotations=annotations_by_pmid,
    ...     string_interactions=string_data,
    ... )
    >>>
    >>> # Run pipeline
    >>> pipeline = KGPipeline(extractor_name="cerebras")
    >>> result = await pipeline.run(input_data)
    >>> print(f"Extracted {result.relationships_valid} relationships")

Legacy Example - Phase-based Pipeline:
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
"""

from pipeline.config import PipelineConfig
from pipeline.metrics import (
    MODEL_PRICING,
    PhaseMetrics,
    PipelineReport,
    TokenUsage,
    estimate_cost,
)
from pipeline.models import (
    PipelineInput,
    PipelineResult,
)
from pipeline.kg_pipeline import (
    KGPipeline,
    run_kg_pipeline,
)
from pipeline.orchestrator import (
    KGOrchestrator,
    create_orchestrator,
    get_orchestrator,
    set_orchestrator,
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

# Import batched miner from extraction for convenience
from extraction.batched_litellm_miner import (
    BatchedLiteLLMMiner,
    run_batched_mining,
    create_batched_miner,
    MiningStatistics,
    ExtractedRelationship,
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
    # Hybrid Architecture - Data Models
    "PipelineInput",
    "PipelineResult",
    # Hybrid Architecture - Pipeline
    "KGPipeline",
    "run_kg_pipeline",
    # Hybrid Architecture - Orchestrator
    "KGOrchestrator",
    "create_orchestrator",
    "get_orchestrator",
    "set_orchestrator",
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
    # Batched Mining (from extraction)
    "BatchedLiteLLMMiner",
    "run_batched_mining",
    "create_batched_miner",
    "MiningStatistics",
    "ExtractedRelationship",
]
