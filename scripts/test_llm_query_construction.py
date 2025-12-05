#!/usr/bin/env python3
"""
Test script to verify LLM-based PubMed query construction.

This script tests:
1. LLM query construction with valid API key
2. Fallback behavior when LLM fails
3. Query structure and content

Usage:
    uv run python scripts/test_llm_query_construction.py
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

from agent.data_fetch_agent import DataFetchAgent
from pipeline.config import PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_llm_query_construction():
    """Test LLM-based query construction."""
    print("=" * 60)
    print("Testing LLM-based PubMed Query Construction")
    print("=" * 60)

    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\nWARNING: GOOGLE_API_KEY not set. LLM will fail and fallback will be used.")
    else:
        print(f"\nGOOGLE_API_KEY is set (length: {len(api_key)})")

    # Create config with explicit settings
    config = PipelineConfig(
        pubmed_date_filter="2020:2025[pdat]",
        pubmed_species_filter="humans[MeSH Terms]",
        string_extend_network=10,
    )

    print(f"\nConfig:")
    print(f"  - pubmed_date_filter: {config.pubmed_date_filter}")
    print(f"  - pubmed_species_filter: {config.pubmed_species_filter}")
    print(f"  - string_extend_network: {config.string_extend_network}")

    # Create agent
    agent = DataFetchAgent(config=config)

    # Test query
    test_query = "orexin signaling pathway"
    print(f"\n{'=' * 60}")
    print(f"Test Query: '{test_query}'")
    print("=" * 60)

    # Run fetch (this will exercise the full pipeline including LLM query construction)
    # We'll use a small max_articles to keep it fast
    try:
        result = await agent.fetch(
            user_query=test_query,
            max_articles=5,  # Small number for testing
        )

        print(f"\nResults:")
        print(f"  - Seed proteins: {result.seed_proteins[:10]}...")
        print(f"  - STRING interactions: {result.interaction_count}")
        print(f"  - Articles fetched: {result.article_count}")
        print(f"  - PubMed query: {result.pubmed_query}")

        # Check query construction metadata
        query_meta = result.metadata.get("query_construction", {})
        print(f"\nQuery Construction:")
        print(f"  - Method: {query_meta.get('method', 'unknown')}")
        print(f"  - Query length: {query_meta.get('query_length', 'N/A')}")

        if query_meta.get("method") == "llm":
            print(f"  - Was refined: {query_meta.get('was_refined', False)}")
            token_usage = query_meta.get("token_usage", {})
            print(f"  - Tokens used: {token_usage.get('total_tokens', 'N/A')}")
            print(f"  - Cost: ${token_usage.get('cost_usd', 0):.6f}")
            print("\n*** LLM query construction SUCCESS ***")
        else:
            print("\n*** Fallback query construction was used ***")

        # Validate query structure
        print(f"\nQuery Validation:")
        query = result.pubmed_query
        has_proteins = any(p in query for p in result.seed_proteins[:5])
        has_mesh = "[mesh" in query.lower() or "[MeSH" in query
        has_date = "[pdat]" in query.lower()

        print(f"  - Contains proteins: {has_proteins}")
        print(f"  - Contains MeSH terms: {has_mesh}")
        print(f"  - Contains date filter: {has_date}")

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def test_query_only():
    """Test just the query construction without fetching literature."""
    print("\n" + "=" * 60)
    print("Testing Query Construction Only (No PubMed Fetch)")
    print("=" * 60)

    from pipeline.query_agent import construct_pubmed_query
    from pipeline.metrics import PipelineReport

    config = PipelineConfig(
        pubmed_date_filter="2020:2025[pdat]",
        pubmed_species_filter="humans[MeSH Terms]",
    )
    report = PipelineReport.create()

    # Test with BRCA proteins
    proteins = ["BRCA1", "BRCA2", "TP53", "ATM", "CHEK2", "RAD51", "PALB2", "BARD1"]
    research_focus = "DNA damage response and breast cancer susceptibility"

    print(f"\nInput:")
    print(f"  - Proteins: {proteins}")
    print(f"  - Research focus: {research_focus}")

    try:
        result = await construct_pubmed_query(
            proteins=proteins,
            research_focus=research_focus,
            config=config,
            report=report,
        )

        print(f"\nResult:")
        print(f"  - Query: {result.query}")
        print(f"  - Length: {result.query_length} chars")
        print(f"  - Was refined: {result.was_refined}")
        print(f"  - Strategy: {result.strategy_explanation}")
        print(f"\nToken Usage:")
        print(f"  - Prompt tokens: {result.token_usage.prompt_tokens}")
        print(f"  - Completion tokens: {result.token_usage.completion_tokens}")
        print(f"  - Total tokens: {result.token_usage.total_tokens}")
        print(f"  - Cost: ${result.token_usage.cost_usd:.6f}")

        print("\n*** LLM query construction SUCCESS ***")
        return True

    except Exception as e:
        logger.error(f"Query construction failed: {e}", exc_info=True)
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LLM Query Construction Test Suite")
    print("=" * 60)

    # Test 1: Query construction only
    success1 = await test_query_only()

    # Test 2: Full pipeline with query construction
    success2 = await test_llm_query_construction()

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"  Query Only Test: {'PASS' if success1 else 'FAIL'}")
    print(f"  Full Pipeline Test: {'PASS' if success2 else 'FAIL'}")

    return 0 if (success1 and success2) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
