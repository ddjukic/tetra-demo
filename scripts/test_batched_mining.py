#!/usr/bin/env python3
"""
Test batched mining at different scales.

Fetches real abstracts from PubMed and measures performance of the
BatchedMiningOrchestrator at various scales (20, 50, 100 abstracts).

Run with: uv run python scripts/test_batched_mining.py

Usage:
    # Run full test suite
    uv run python scripts/test_batched_mining.py

    # Run with specific sizes
    uv run python scripts/test_batched_mining.py --sizes 20 50

    # Run with custom query
    uv run python scripts/test_batched_mining.py --query "BRCA1 breast cancer"
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from clients.pubmed_client import PubMedClient
from pipeline.batched_mining import (
    BatchedMiningConfig,
    BatchedMiningOrchestrator,
)


async def fetch_abstracts(query: str, max_results: int) -> list[dict]:
    """
    Fetch abstracts from PubMed.

    Args:
        query: PubMed search query.
        max_results: Maximum number of results to fetch.

    Returns:
        List of article dicts with pmid, title, abstract, year.
    """
    async with PubMedClient() as client:
        print(f"  Searching PubMed: '{query}' (max {max_results})...")
        pmids = await client.search(query, max_results=max_results)
        print(f"  Found {len(pmids)} PMIDs")

        if not pmids:
            return []

        print(f"  Fetching abstracts...")
        articles = await client.fetch_abstracts(pmids)
        print(f"  Retrieved {len(articles)} articles with abstracts")

        # Filter out articles without abstracts
        articles_with_abstracts = [a for a in articles if a.get("abstract")]
        print(f"  {len(articles_with_abstracts)} articles have abstracts")

        return articles_with_abstracts


async def get_pubtator_annotations(pmids: list[str]) -> dict[str, list[dict]]:
    """
    Get PubTator annotations for PMIDs.

    Args:
        pmids: List of PubMed IDs.

    Returns:
        Dictionary mapping pmid -> list of annotation dicts.
    """
    async with PubMedClient() as client:
        print(f"  Fetching PubTator annotations for {len(pmids)} articles...")
        annotations_list = await client.get_pubtator_annotations(
            pmids,
            entity_types=["Gene", "Disease", "Chemical"],
        )
        print(f"  Retrieved {len(annotations_list)} annotations")

        # Convert to dict keyed by PMID
        annotations_dict: dict[str, list[dict]] = {}
        for annot in annotations_list:
            pmid = annot.get("pmid", "")
            if pmid:
                if pmid not in annotations_dict:
                    annotations_dict[pmid] = []
                annotations_dict[pmid].append(annot)

        print(f"  Annotations cover {len(annotations_dict)} articles")
        return annotations_dict


async def test_batched_mining_scale(
    articles: list[dict],
    annotations: dict[str, list[dict]],
    n: int,
) -> dict:
    """
    Test batched mining at a specific scale.

    Args:
        articles: All fetched articles.
        annotations: All annotations.
        n: Number of abstracts to test with.

    Returns:
        Test results dictionary.
    """
    # Take subset
    test_articles = articles[:n]
    test_pmids = [a.get("pmid", "") for a in test_articles]
    test_annotations = {pmid: annotations.get(pmid, []) for pmid in test_pmids}

    print(f"\n{'='*60}")
    print(f"Testing with {n} abstracts")
    print(f"{'='*60}")

    # Create orchestrator with default config
    config = BatchedMiningConfig(
        target_tokens_per_chunk=5000,
        min_chunks=3,
        max_concurrent=5,
        max_retries=3,
    )
    orchestrator = BatchedMiningOrchestrator(config)

    # Analyze
    analysis = orchestrator.analyze_abstracts(test_articles)
    print(f"\nAnalysis:")
    print(f"  Total tokens: {analysis['total_tokens']}")
    print(f"  Mean tokens/abstract: {analysis['mean_tokens']}")
    print(f"  Min/Max tokens: {analysis['min_tokens']}/{analysis['max_tokens']}")
    print(f"  Recommended chunks: {analysis['recommended_chunks']}")

    # Run mining
    print(f"\nRunning batched mining...")
    start = time.time()
    result = await orchestrator.run(test_articles, test_annotations)
    duration = time.time() - start

    # Print results
    stats = result["statistics"]
    print(f"\nResults:")
    print(f"  Abstracts processed: {stats['total_abstracts']}")
    print(f"  Chunks created: {stats['chunks_created']}")
    print(f"  Chunks processed: {stats['chunks_processed']}")
    print(f"  Relationships (raw): {stats['relationships_raw']}")
    print(f"  Relationships (deduped): {stats['relationships_extracted']}")
    print(f"  Tokens used: {stats['tokens_used']}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Throughput: {stats['throughput_abstracts_per_sec']:.1f} abstracts/sec")

    if result["errors"]:
        print(f"\nErrors ({len(result['errors'])}):")
        for err in result["errors"][:3]:
            print(f"  - {err[:100]}...")

    # Show sample relationships
    relationships = result["relationships"]
    if relationships:
        print(f"\nSample relationships ({min(5, len(relationships))} of {len(relationships)}):")
        for rel in relationships[:5]:
            print(f"  {rel.source_entity} --[{rel.relation_type}]--> {rel.target_entity}")
            evidence = rel.evidence_sentence
            if len(evidence) > 100:
                evidence = evidence[:100] + "..."
            print(f"    Evidence: {evidence}")
            print(f"    PMID: {rel.pmid}, Confidence: {rel.confidence:.2f}")

    return {
        "n": n,
        "analysis": analysis,
        "statistics": stats,
        "relationships_count": len(relationships),
        "duration": duration,
        "errors_count": len(result["errors"]),
    }


async def run_scale_tests(
    query: str,
    sizes: list[int],
) -> list[dict]:
    """
    Run batched mining tests at multiple scales.

    Args:
        query: PubMed search query.
        sizes: List of test sizes (e.g., [20, 50, 100]).

    Returns:
        List of test results.
    """
    # Fetch maximum needed abstracts
    max_needed = max(sizes)
    print(f"\nFetching up to {max_needed} abstracts...")
    articles = await fetch_abstracts(query, max_needed + 20)  # Fetch extra for filtering

    if len(articles) < min(sizes):
        print(f"ERROR: Only found {len(articles)} articles, need at least {min(sizes)}")
        return []

    # Get annotations
    pmids = [a.get("pmid", "") for a in articles]
    annotations = await get_pubtator_annotations(pmids)

    # Run tests at each scale
    results = []
    for n in sizes:
        if n <= len(articles):
            try:
                result = await test_batched_mining_scale(articles, annotations, n)
                results.append(result)
            except Exception as e:
                print(f"\nERROR testing with {n} abstracts: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\nSkipping {n} abstracts (only {len(articles)} available)")

    return results


def print_summary(results: list[dict]) -> None:
    """Print summary table of all test results."""
    if not results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Header
    print(f"{'N':>6} | {'Chunks':>6} | {'Tokens':>8} | {'Rels':>6} | {'Duration':>10} | {'Throughput':>12}")
    print(f"{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*12}")

    for r in results:
        stats = r["statistics"]
        print(
            f"{r['n']:>6} | "
            f"{stats['chunks_created']:>6} | "
            f"{stats['tokens_used']:>8} | "
            f"{r['relationships_count']:>6} | "
            f"{r['duration']:>8.2f}s | "
            f"{stats['throughput_abstracts_per_sec']:>9.1f}/s"
        )

    print(f"\nConclusion:")
    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        speedup = last["statistics"]["throughput_abstracts_per_sec"] / first["statistics"]["throughput_abstracts_per_sec"] if first["statistics"]["throughput_abstracts_per_sec"] > 0 else 1.0
        print(f"  Throughput at {last['n']} abstracts: {last['statistics']['throughput_abstracts_per_sec']:.1f} abstracts/sec")
        print(f"  Scaling factor: {speedup:.2f}x from {first['n']} to {last['n']} abstracts")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test batched mining at different scales"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="orexin sleep regulation",
        help="PubMed search query (default: 'orexin sleep regulation')",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="Test sizes (default: 20 50 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Batched Mining Scale Test")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Test sizes: {args.sizes}")

    try:
        results = await run_scale_tests(args.query, args.sizes)

        print_summary(results)

        # Save results if output specified
        if args.output and results:
            output_path = Path(args.output)
            # Convert to serializable format
            serializable_results = []
            for r in results:
                sr = {**r}
                sr["relationships_count"] = r["relationships_count"]
                serializable_results.append(sr)

            with open(output_path, "w") as f:
                json.dump(serializable_results, f, indent=2)
            print(f"\nResults saved to: {output_path}")

        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
