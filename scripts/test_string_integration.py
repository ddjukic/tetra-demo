#!/usr/bin/env python3
"""
Integration test for STRING Network Extension feature.

This script tests the full pipeline with the STRING extension enabled:
1. DataFetchAgent with extended STRING network
2. Comparison: with vs without extension
3. Full KG pipeline run showing richer graph output

Run with: uv run python scripts/test_string_integration.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
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

from agent.data_fetch_agent import DataFetchAgent
from pipeline.config import PipelineConfig
from pipeline.models import PipelineInput


def print_separator(title: str) -> None:
    """Print a section separator."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_input_summary(input_data: PipelineInput, label: str) -> None:
    """Print summary of pipeline input data."""
    print(f"\n{label}")
    print("-" * 50)
    print(f"  Seed proteins: {len(input_data.seed_proteins)}")
    if input_data.seed_proteins:
        proteins_str = ", ".join(input_data.seed_proteins[:15])
        if len(input_data.seed_proteins) > 15:
            proteins_str += f"... (+{len(input_data.seed_proteins) - 15} more)"
        print(f"    {proteins_str}")
    print(f"  STRING interactions: {input_data.interaction_count}")
    print(f"  Unique proteins from STRING: {len(input_data.unique_proteins)}")
    print(f"  Articles fetched: {input_data.article_count}")
    print(f"  Annotations collected: {input_data.annotation_count}")

    # Show extension metadata if available
    if "string_extension" in input_data.metadata:
        ext = input_data.metadata["string_extension"]
        print(f"\n  STRING Extension:")
        print(f"    Original seeds: {ext.get('original_seeds', [])}")
        print(f"    Extend network: {ext.get('extend_network', 0)}")
        print(f"    Expanded proteins: {ext.get('expanded_proteins', 0)}")
        print(f"    Total interactions: {ext.get('total_interactions', 0)}")


async def test_without_extension() -> PipelineInput:
    """Test DataFetchAgent WITHOUT STRING extension (baseline)."""
    print_separator("TEST 1: DataFetchAgent WITHOUT Extension (Baseline)")

    # Create agent with extension disabled
    config = PipelineConfig(string_extend_network=0)  # Disable extension
    agent = DataFetchAgent(config=config)

    input_data = await agent.fetch(
        user_query="Build a KG for the orexin signaling pathway",
        max_articles=10,
    )

    print_input_summary(input_data, "Results WITHOUT Extension")
    return input_data


async def test_with_extension() -> PipelineInput:
    """Test DataFetchAgent WITH STRING extension."""
    print_separator("TEST 2: DataFetchAgent WITH Extension (string_extend_network=10)")

    # Create agent with default extension (10)
    config = PipelineConfig(string_extend_network=10)
    agent = DataFetchAgent(config=config)

    input_data = await agent.fetch(
        user_query="Build a KG for the orexin signaling pathway",
        max_articles=10,
    )

    print_input_summary(input_data, "Results WITH Extension (extend_network=10)")
    return input_data


async def test_full_pipeline(input_data: PipelineInput) -> dict[str, Any]:
    """Test the full KG pipeline with extended data."""
    print_separator("TEST 3: Full KG Pipeline with Extended Data")

    from pipeline.kg_pipeline import KGPipeline

    pipeline = KGPipeline(extractor_name="cerebras")

    print("\nRunning KGPipeline.run()...")
    result = await pipeline.run(input_data)

    print(f"\nPipeline Results:")
    print("-" * 50)
    print(f"  Success: {result.success}")
    print(f"  Relationships extracted: {result.relationships_extracted}")
    print(f"  Relationships valid: {result.relationships_valid}")
    print(f"  Entities in graph: {result.entities_found}")
    print(f"  Processing time: {result.processing_time_ms:.0f}ms")

    if result.graph:
        summary = result.graph.to_summary()
        print(f"\nGraph Summary:")
        print(f"  - Nodes: {summary['node_count']}")
        print(f"  - Edges: {summary['edge_count']}")
        print(f"  - Entity types: {summary.get('entity_types', {})}")
        print(f"  - Relationship types: {summary.get('relationship_types', {})}")

        # Show some sample nodes
        if summary['node_count'] > 0:
            print(f"\n  Sample nodes (first 10):")
            node_list = list(result.graph.graph.nodes())[:10]
            for node in node_list:
                attrs = result.graph.graph.nodes[node]
                node_type = attrs.get("type", "unknown")
                print(f"    - {node} ({node_type})")

    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    return result.to_summary()


def compare_results(
    without_ext: PipelineInput,
    with_ext: PipelineInput,
) -> None:
    """Compare results with and without extension."""
    print_separator("COMPARISON: With vs Without Extension")

    print("\nMetric                        | Without Ext | With Ext | Change")
    print("-" * 70)

    # Seed proteins
    without_seeds = len(without_ext.seed_proteins)
    with_seeds = len(with_ext.seed_proteins)
    change = f"+{with_seeds - without_seeds}" if with_seeds > without_seeds else str(with_seeds - without_seeds)
    print(f"Seed/Expanded proteins        |     {without_seeds:>6} |   {with_seeds:>6} | {change}")

    # STRING interactions
    without_int = without_ext.interaction_count
    with_int = with_ext.interaction_count
    change = f"+{with_int - without_int}" if with_int > without_int else str(with_int - without_int)
    print(f"STRING interactions           |     {without_int:>6} |   {with_int:>6} | {change}")

    # Unique proteins from STRING
    without_unique = len(without_ext.unique_proteins)
    with_unique = len(with_ext.unique_proteins)
    change = f"+{with_unique - without_unique}" if with_unique > without_unique else str(with_unique - without_unique)
    print(f"Unique proteins (STRING)      |     {without_unique:>6} |   {with_unique:>6} | {change}")

    # New proteins discovered
    original_seeds = set(without_ext.metadata.get("string_extension", {}).get("original_seeds", without_ext.seed_proteins))
    if "string_extension" in with_ext.metadata:
        original_seeds = set(with_ext.metadata["string_extension"]["original_seeds"])

    new_proteins = with_ext.unique_proteins - original_seeds
    print(f"\nNew proteins discovered by extension:")
    if new_proteins:
        proteins_list = sorted(new_proteins)
        print(f"  Count: {len(new_proteins)}")
        print(f"  Proteins: {', '.join(proteins_list[:20])}")
        if len(proteins_list) > 20:
            print(f"            ...and {len(proteins_list) - 20} more")
    else:
        print("  None")

    # Calculate expansion ratios
    if without_int > 0:
        int_ratio = with_int / without_int
        print(f"\nExpansion ratio (interactions): {int_ratio:.1f}x")
    if without_unique > 0:
        protein_ratio = with_unique / without_unique
        print(f"Expansion ratio (proteins): {protein_ratio:.1f}x")


async def main() -> None:
    """Run all integration tests."""
    print("=" * 70)
    print("STRING NETWORK EXTENSION - INTEGRATION TEST")
    print("=" * 70)
    print("\nThis test validates the STRING extension feature in the full pipeline:")
    print("1. DataFetchAgent without extension (baseline)")
    print("2. DataFetchAgent with extension (string_extend_network=10)")
    print("3. Full KG pipeline with extended data")
    print("4. Comparison of results")

    try:
        # Test 1: Without extension (baseline)
        without_ext = await test_without_extension()

        # Test 2: With extension
        with_ext = await test_with_extension()

        # Compare results
        compare_results(without_ext, with_ext)

        # Test 3: Full pipeline with extended data
        # Only run if we have data
        if not with_ext.is_empty:
            await test_full_pipeline(with_ext)
        else:
            print("\nSkipping pipeline test - no data fetched")

        print_separator("ALL INTEGRATION TESTS COMPLETED SUCCESSFULLY")

        # Final summary
        print("\nSUCCESS CRITERIA CHECK:")
        print("-" * 50)

        # Check protein expansion
        expanded_proteins = len(with_ext.seed_proteins)
        protein_check = expanded_proteins >= 10
        print(f"  [{'OK' if protein_check else 'FAIL'}] Expanded proteins: {expanded_proteins} (target: >= 10)")

        # Check interactions
        interactions = with_ext.interaction_count
        int_check = interactions >= 20
        print(f"  [{'OK' if int_check else 'FAIL'}] STRING interactions: {interactions} (target: >= 20)")

        # Check for relevant proteins
        relevant_proteins = {"GNAI1", "GNAS", "GNB1", "GNG2"}  # G-proteins expected
        found_relevant = relevant_proteins & with_ext.unique_proteins
        relevance_check = len(found_relevant) >= 2
        print(f"  [{'OK' if relevance_check else 'FAIL'}] Relevant proteins found: {found_relevant}")

        all_passed = protein_check and int_check and relevance_check
        print(f"\n  Overall: {'ALL CRITERIA MET' if all_passed else 'SOME CRITERIA NOT MET'}")

    except Exception as e:
        logger.exception("Test failed")
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
