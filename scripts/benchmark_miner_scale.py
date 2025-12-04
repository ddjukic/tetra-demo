#!/usr/bin/env python3
"""
Test batched mining at different scales using the new BatchedLiteLLMMiner.

Fetches real abstracts from PubMed and measures performance of the
BatchedLiteLLMMiner with multi-provider LLM support (Cerebras, Gemini, etc.)
at various scales (20, 50, 100 abstracts).

Run with: uv run python scripts/test_batched_mining.py

Usage:
    # Run full test suite with default extractor (cerebras)
    uv run python scripts/test_batched_mining.py

    # Run with specific extractor
    uv run python scripts/test_batched_mining.py --extractor gemini

    # Run with specific sizes
    uv run python scripts/test_batched_mining.py --sizes 20 50

    # Run with custom query
    uv run python scripts/test_batched_mining.py --query "BRCA1 breast cancer"

    # Quick test with 5 abstracts
    uv run python scripts/test_batched_mining.py --abstracts 5
"""

import argparse
import asyncio
import json
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
from extraction import (
    BatchedLiteLLMMiner,
    create_batched_miner,
    get_config,
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
    extractor_name: str,
) -> dict:
    """
    Test batched mining at a specific scale.

    Args:
        articles: All fetched articles.
        annotations: All annotations.
        n: Number of abstracts to test with.
        extractor_name: Name of extractor to use (cerebras, gemini, etc.)

    Returns:
        Test results dictionary.
    """
    # Take subset
    test_articles = articles[:n]
    test_pmids = [a.get("pmid", "") for a in test_articles]
    test_annotations = {pmid: annotations.get(pmid, []) for pmid in test_pmids}

    print(f"\n{'='*60}")
    print(f"Testing with {n} abstracts using {extractor_name}")
    print(f"{'='*60}")

    # Create miner using factory function
    miner = create_batched_miner(
        extractor_name=extractor_name,
        evidence_threshold=0.65,
    )

    # Get config for display
    config = get_config()
    batched_config = config.BATCHED
    print(f"\nConfiguration:")
    print(f"  Model: {miner.model}")
    print(f"  Target tokens/chunk: {batched_config.TARGET_TOKENS_PER_CHUNK}")
    print(f"  Min chunks: {batched_config.MIN_CHUNKS}")
    print(f"  Max concurrent: {batched_config.MAX_CONCURRENT}")

    # Run mining
    print(f"\nRunning batched mining...")
    start = time.time()
    result = await miner.run(test_articles, test_annotations)
    duration = time.time() - start

    # Print results
    stats = result["statistics"]
    print(f"\nResults:")
    print(f"  Abstracts processed: {stats['total_abstracts']}")
    print(f"  Chunks created: {stats['total_chunks']}")
    print(f"  Chunks processed: {stats['chunks_processed']}")
    print(f"  Relationships (total): {stats['total_relationships']}")
    print(f"  Relationships (valid): {stats['valid_relationships']}")
    print(f"  Validation rate: {stats['validation_rate']:.1%}")
    print(f"  PMID failures: {stats['pmid_failures']}")
    print(f"  Evidence failures: {stats['evidence_failures']}")
    print(f"  Tokens used: {stats['total_prompt_tokens'] + stats['total_completion_tokens']}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Throughput: {stats['throughput_tok_per_sec']:.0f} tokens/sec")

    if result["errors"]:
        print(f"\nErrors ({len(result['errors'])}):")
        for err in result["errors"][:3]:
            print(f"  - {err[:100]}...")

    # Show sample valid relationships
    valid_rels = result["valid_relationships"]
    if valid_rels:
        print(f"\nSample valid relationships ({min(5, len(valid_rels))} of {len(valid_rels)}):")
        for rel in valid_rels[:5]:
            print(f"  {rel['entity1']} --[{rel['relationship']}]--> {rel['entity2']}")
            evidence = rel.get("evidence_text", "")
            if len(evidence) > 100:
                evidence = evidence[:100] + "..."
            print(f"    Evidence: {evidence}")
            print(f"    PMID: {rel['pmid']}, Confidence: {rel['confidence']:.2f}")

    return {
        "n": n,
        "extractor": extractor_name,
        "model": miner.model,
        "statistics": stats,
        "valid_count": len(valid_rels),
        "total_count": stats["total_relationships"],
        "duration": duration,
        "errors_count": len(result["errors"]),
    }


async def run_scale_tests(
    query: str,
    sizes: list[int],
    extractor_name: str,
) -> list[dict]:
    """
    Run batched mining tests at multiple scales.

    Args:
        query: PubMed search query.
        sizes: List of test sizes (e.g., [20, 50, 100]).
        extractor_name: Name of extractor to use.

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
                result = await test_batched_mining_scale(
                    articles, annotations, n, extractor_name
                )
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

    if results:
        print(f"Extractor: {results[0].get('extractor', 'unknown')}")
        print(f"Model: {results[0].get('model', 'unknown')}")
        print()

    # Header
    print(f"{'N':>6} | {'Chunks':>6} | {'Tokens':>8} | {'Valid':>6} | {'Rate':>8} | {'Duration':>10} | {'Tok/s':>10}")
    print(f"{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

    for r in results:
        stats = r["statistics"]
        total_tokens = stats['total_prompt_tokens'] + stats['total_completion_tokens']
        print(
            f"{r['n']:>6} | "
            f"{stats['total_chunks']:>6} | "
            f"{total_tokens:>8} | "
            f"{r['valid_count']:>6} | "
            f"{stats['validation_rate']:>7.1%} | "
            f"{r['duration']:>8.2f}s | "
            f"{stats['throughput_tok_per_sec']:>8.0f}"
        )

    print(f"\nConclusion:")
    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        first_throughput = first["statistics"]["throughput_tok_per_sec"]
        last_throughput = last["statistics"]["throughput_tok_per_sec"]
        if first_throughput > 0:
            speedup = last_throughput / first_throughput
            print(f"  Throughput at {last['n']} abstracts: {last_throughput:.0f} tokens/sec")
            print(f"  Scaling factor: {speedup:.2f}x from {first['n']} to {last['n']} abstracts")
        print(f"  Overall validation rate: {last['statistics']['validation_rate']:.1%}")


async def run_single_test(
    query: str,
    n: int,
    extractor_name: str,
) -> dict | None:
    """
    Run a single batched mining test.

    Args:
        query: PubMed search query.
        n: Number of abstracts to test with.
        extractor_name: Name of extractor to use.

    Returns:
        Test result dictionary or None on failure.
    """
    print(f"\nFetching {n} abstracts...")
    articles = await fetch_abstracts(query, n + 5)  # Fetch extra for filtering

    if len(articles) < n:
        print(f"WARNING: Only found {len(articles)} articles, using all of them")
        n = len(articles)

    if not articles:
        print("ERROR: No articles found")
        return None

    # Get annotations
    pmids = [a.get("pmid", "") for a in articles]
    annotations = await get_pubtator_annotations(pmids)

    try:
        result = await test_batched_mining_scale(articles, annotations, n, extractor_name)
        return result
    except Exception as e:
        print(f"\nERROR testing: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test batched mining at different scales with LiteLLM multi-provider support"
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
        help="Test sizes for scale tests (default: 20 50 100)",
    )
    parser.add_argument(
        "--abstracts",
        type=int,
        help="Run single test with specific number of abstracts (overrides --sizes)",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="cerebras",
        choices=["cerebras", "gemini", "openrouter"],
        help="LLM extractor to use (default: cerebras)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Batched Mining Test (LiteLLM Multi-Provider)")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Extractor: {args.extractor}")

    if args.abstracts:
        print(f"Mode: Single test with {args.abstracts} abstracts")
    else:
        print(f"Mode: Scale test with sizes {args.sizes}")

    try:
        if args.abstracts:
            # Run single test
            result = await run_single_test(args.query, args.abstracts, args.extractor)
            results = [result] if result else []
        else:
            # Run scale tests
            results = await run_scale_tests(args.query, args.sizes, args.extractor)

        print_summary(results)

        # Save results if output specified
        if args.output and results:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
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
