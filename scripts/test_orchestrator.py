#!/usr/bin/env python3
"""
Test the KG Orchestrator with pruning and PubMed filters.

Usage:
    uv run python scripts/test_orchestrator.py [--articles N] [--query "..."]

Examples:
    uv run python scripts/test_orchestrator.py --articles 20
    uv run python scripts/test_orchestrator.py --articles 50 --query "EGFR signaling in cancer"
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pipeline.orchestrator import KGOrchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_test(query: str, max_articles: int) -> dict:
    """Run the orchestrator test and return results."""
    print(f"\n{'='*60}")
    print(f"KG Orchestrator Test")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"Max articles: {max_articles}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Create orchestrator
    orchestrator = KGOrchestrator(extractor_name="cerebras")

    # Build the graph
    print("[1] Building knowledge graph...")
    graph = await orchestrator.build(
        user_query=query,
        max_articles=max_articles,
    )

    elapsed = time.time() - start_time

    # Get the pipeline result from orchestrator's last run
    # We need to access the result from the pipeline
    result = orchestrator._pipeline._miner  # Access for stats

    # Get graph summary
    summary = graph.to_summary()

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    # Display pruning stats if available
    # These come from the pipeline result, but we need to re-run to get them
    # For now, display graph summary
    print(f"\nGraph Summary (after pruning):")
    print(f"  Nodes: {summary.get('node_count', 0)}")
    print(f"  Edges: {summary.get('edge_count', 0)}")

    print(f"\nRelationship Types:")
    rel_types = summary.get("relationship_types", {})
    for rel_type, count in sorted(rel_types.items(), key=lambda x: -x[1]):
        print(f"  {rel_type}: {count}")

    print(f"\nEntity Types:")
    entity_types = summary.get("entity_types", {})
    for entity_type, count in sorted(entity_types.items(), key=lambda x: -x[1]):
        print(f"  {entity_type}: {count}")

    print(f"\n{'='*60}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return {
        "query": query,
        "max_articles": max_articles,
        "elapsed_seconds": elapsed,
        "graph_summary": summary,
    }


async def run_direct_pipeline_test(query: str, max_articles: int) -> dict:
    """
    Run a direct pipeline test to show before/after pruning stats.
    """
    from agent.data_fetch_agent import DataFetchAgent
    from pipeline.kg_pipeline import KGPipeline

    print(f"\n{'='*60}")
    print(f"Direct Pipeline Test (with pruning stats)")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"Max articles: {max_articles}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Step 1: Fetch data
    print("[1] Fetching data with DataFetchAgent...")
    fetch_agent = DataFetchAgent()
    pipeline_input = await fetch_agent.fetch(
        user_query=query,
        max_articles=max_articles,
    )

    print(f"    PubMed query: {pipeline_input.pubmed_query}")
    print(f"    Articles fetched: {pipeline_input.article_count}")
    print(f"    STRING interactions: {pipeline_input.interaction_count}")
    print(f"    NER annotations: {pipeline_input.annotation_count}")

    # Step 2: Run pipeline
    print("\n[2] Running KGPipeline...")
    pipeline = KGPipeline(extractor_name="cerebras")
    result = await pipeline.run(pipeline_input)

    elapsed = time.time() - start_time

    # Display results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    print(f"\nBefore Pruning:")
    before = result.stats_before_pruning
    print(f"  Nodes: {before.node_count}")
    print(f"  Edges: {before.edge_count}")
    print(f"  Components: {before.component_count}")
    if before.relationship_types:
        print(f"  Relationship types:")
        for rt, count in sorted(before.relationship_types.items(), key=lambda x: -x[1]):
            print(f"    {rt}: {count}")

    print(f"\nAfter Pruning:")
    after = result.stats_after_pruning
    print(f"  Nodes: {after.node_count}")
    print(f"  Edges: {after.edge_count}")
    print(f"  Components: {after.component_count}")
    if after.relationship_types:
        print(f"  Relationship types:")
        for rt, count in sorted(after.relationship_types.items(), key=lambda x: -x[1]):
            print(f"    {rt}: {count}")

    print(f"\nPruning Summary:")
    print(f"  Nodes pruned: {result.nodes_pruned} ({result.nodes_pruned / max(before.node_count, 1) * 100:.1f}%)")
    print(f"  Edges pruned: {result.edges_pruned} ({result.edges_pruned / max(before.edge_count, 1) * 100:.1f}%)")

    print(f"\nExtraction Stats:")
    print(f"  Relationships extracted: {result.relationships_extracted}")
    print(f"  Valid relationships: {result.relationships_valid}")
    print(f"  Processing time: {result.processing_time_ms:.0f}ms")

    if result.errors:
        print(f"\nErrors: {result.errors}")

    print(f"\n{'='*60}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return {
        "query": query,
        "max_articles": max_articles,
        "pubmed_query": pipeline_input.pubmed_query,
        "articles_fetched": pipeline_input.article_count,
        "elapsed_seconds": elapsed,
        "before_pruning": before.to_dict(),
        "after_pruning": after.to_dict(),
        "nodes_pruned": result.nodes_pruned,
        "edges_pruned": result.edges_pruned,
        "relationships_extracted": result.relationships_extracted,
        "relationships_valid": result.relationships_valid,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test KG Orchestrator with pruning and PubMed filters"
    )
    parser.add_argument(
        "--articles", "-a",
        type=int,
        default=20,
        help="Maximum number of articles to fetch (default: 20)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="Build a knowledge graph for orexin signaling pathway with HCRTR1, HCRTR2, HCRT",
        help="Query to build the knowledge graph"
    )
    parser.add_argument(
        "--direct", "-d",
        action="store_true",
        help="Run direct pipeline test (shows before/after pruning stats)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    # Run the appropriate test
    if args.direct:
        result = asyncio.run(run_direct_pipeline_test(args.query, args.articles))
    else:
        result = asyncio.run(run_test(args.query, args.articles))

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
