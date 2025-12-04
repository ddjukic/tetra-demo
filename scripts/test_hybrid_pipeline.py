#!/usr/bin/env python3
"""
Test the hybrid KG pipeline architecture.

This script tests the new hybrid architecture:
1. DataFetchAgent - LLM-driven data collection
2. KGPipeline - Deterministic processing
3. KGOrchestrator - Coordination of both

Run with: uv run python scripts/test_hybrid_pipeline.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def test_data_fetch_agent():
    """Test the DataFetchAgent independently."""
    print("\n" + "=" * 60)
    print("TEST 1: DataFetchAgent")
    print("=" * 60)

    from agent.data_fetch_agent import DataFetchAgent

    agent = DataFetchAgent()

    # Test with orexin pathway
    input_data = await agent.fetch(
        user_query="Build a KG for the orexin signaling pathway",
        max_articles=10,
    )

    print(f"\nResults:")
    print(f"  - Seed proteins: {input_data.seed_proteins}")
    print(f"  - STRING interactions: {input_data.interaction_count}")
    print(f"  - Unique proteins from STRING: {len(input_data.unique_proteins)}")
    print(f"  - Articles fetched: {input_data.article_count}")
    print(f"  - Annotations collected: {input_data.annotation_count}")
    print(f"  - PubMed query: {input_data.pubmed_query}")

    return input_data


async def test_kg_pipeline(input_data):
    """Test the KGPipeline with pre-fetched data."""
    print("\n" + "=" * 60)
    print("TEST 2: KGPipeline")
    print("=" * 60)

    from pipeline.kg_pipeline import KGPipeline

    pipeline = KGPipeline(extractor_name="cerebras")

    result = await pipeline.run(input_data)

    print(f"\nResults:")
    print(f"  - Success: {result.success}")
    print(f"  - Relationships extracted: {result.relationships_extracted}")
    print(f"  - Relationships valid: {result.relationships_valid}")
    print(f"  - Entities in graph: {result.entities_found}")
    print(f"  - Processing time: {result.processing_time_ms:.0f}ms")

    if result.graph:
        summary = result.graph.to_summary()
        print(f"\nGraph Summary:")
        print(f"  - Nodes: {summary['node_count']}")
        print(f"  - Edges: {summary['edge_count']}")
        print(f"  - Entity types: {summary.get('entity_types', {})}")
        print(f"  - Relationship types: {summary.get('relationship_types', {})}")

    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    return result


async def test_orchestrator():
    """Test the full KGOrchestrator."""
    print("\n" + "=" * 60)
    print("TEST 3: KGOrchestrator (Full Pipeline)")
    print("=" * 60)

    from pipeline.orchestrator import KGOrchestrator

    orchestrator = KGOrchestrator(extractor_name="cerebras")

    # Build initial graph
    print("\n[Building initial graph...]")
    graph = await orchestrator.build(
        user_query="Build a KG for the orexin signaling pathway",
        max_articles=10,
    )

    summary = graph.to_summary()
    print(f"\nInitial Graph:")
    print(f"  - Nodes: {summary['node_count']}")
    print(f"  - Edges: {summary['edge_count']}")
    print(f"  - Entity types: {summary.get('entity_types', {})}")

    # Test expansion
    print("\n[Expanding graph with disease associations...]")
    try:
        expanded_graph = await orchestrator.expand(
            expansion_query="Add narcolepsy disease associations",
            max_articles=5,
        )

        expanded_summary = expanded_graph.to_summary()
        print(f"\nExpanded Graph:")
        print(f"  - Nodes: {expanded_summary['node_count']} (was {summary['node_count']})")
        print(f"  - Edges: {expanded_summary['edge_count']} (was {summary['edge_count']})")
    except Exception as e:
        print(f"  Expansion failed: {e}")

    return graph


async def main():
    """Run all tests."""
    print("=" * 60)
    print("HYBRID KG PIPELINE TEST")
    print("=" * 60)
    print("\nThis test validates the new hybrid architecture:")
    print("1. DataFetchAgent - LLM-driven data collection")
    print("2. KGPipeline - Deterministic processing")
    print("3. KGOrchestrator - Full coordination")

    try:
        # Test 1: Data Fetch Agent
        input_data = await test_data_fetch_agent()

        # Test 2: KG Pipeline (skip if no data)
        if not input_data.is_empty:
            await test_kg_pipeline(input_data)
        else:
            print("\nSkipping pipeline test - no data fetched")

        # Test 3: Full Orchestrator
        await test_orchestrator()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        logger.exception("Test failed")
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
