#!/usr/bin/env python3
"""
Test script for BatchedLiteLLMMiner with provenance validation.

Fetches test data from PubMed, runs batched extraction with both
Cerebras and Gemini, and validates provenance for all relationships.

Usage:
    # Run with default extractor (cerebras)
    uv run python scripts/test_batched_miner.py

    # Run with specific extractor
    uv run python scripts/test_batched_miner.py --extractor gemini

    # Run with custom number of abstracts
    uv run python scripts/test_batched_miner.py --abstracts 20

    # Run both extractors
    uv run python scripts/test_batched_miner.py --extractor all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from clients.pubmed_client import PubMedClient
from extraction.batched_litellm_miner import BatchedLiteLLMMiner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def fetch_test_data(
    num_abstracts: int = 20,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Fetch test data from PubMed and PubTator.

    Returns:
        Tuple of (articles, annotations_by_pmid)
    """
    logger.info(f"Fetching {num_abstracts} abstracts from PubMed...")

    async with PubMedClient() as client:
        query = "orexin receptor AND 2018:2024[pdat]"
        pmids = await client.search(query, max_results=num_abstracts)

        if len(pmids) < num_abstracts:
            query2 = "(HCRTR1 OR HCRTR2) AND 2015:2024[pdat]"
            pmids2 = await client.search(query2, max_results=num_abstracts - len(pmids))
            pmids = list(set(pmids + pmids2))[:num_abstracts]

        logger.info(f"Found {len(pmids)} PMIDs")

        articles = await client.fetch_abstracts(pmids)
        logger.info(f"Fetched {len(articles)} articles with abstracts")

        annotations_list = await client.get_pubtator_annotations(pmids)
        annotations_by_pmid: dict[str, list[dict]] = {}
        for annot in annotations_list:
            pmid = annot.get("pmid", "")
            if pmid:
                if pmid not in annotations_by_pmid:
                    annotations_by_pmid[pmid] = []
                annotations_by_pmid[pmid].append(annot)

        logger.info(f"Got annotations for {len(annotations_by_pmid)} articles")

    return articles, annotations_by_pmid


def calculate_evidence_stats(results: dict) -> dict:
    """
    Calculate evidence sentence statistics per abstract.

    Returns dict with:
    - relationships_per_pmid: mean, std
    - unique_evidence_per_pmid: mean, std
    - pmids_with_relationships: count
    """
    # Group relationships by PMID
    rels_by_pmid: dict[str, list[dict]] = defaultdict(list)
    evidence_by_pmid: dict[str, set[str]] = defaultdict(set)

    for rel in results.get("relationships", []):
        pmid = rel.get("pmid", "")
        if pmid:
            rels_by_pmid[pmid].append(rel)
            evidence_by_pmid[pmid].add(rel.get("evidence_text", ""))

    if not rels_by_pmid:
        return {
            "pmids_with_relationships": 0,
            "rels_per_pmid_mean": 0.0,
            "rels_per_pmid_std": 0.0,
            "evidence_per_pmid_mean": 0.0,
            "evidence_per_pmid_std": 0.0,
        }

    rel_counts = [len(rels) for rels in rels_by_pmid.values()]
    evidence_counts = [len(evs) for evs in evidence_by_pmid.values()]

    return {
        "pmids_with_relationships": len(rels_by_pmid),
        "rels_per_pmid_mean": mean(rel_counts),
        "rels_per_pmid_std": stdev(rel_counts) if len(rel_counts) > 1 else 0.0,
        "evidence_per_pmid_mean": mean(evidence_counts),
        "evidence_per_pmid_std": stdev(evidence_counts) if len(evidence_counts) > 1 else 0.0,
    }


def print_validation_report(results: dict, extractor_name: str) -> None:
    """Print a detailed validation report."""
    stats = results["statistics"]
    validation_results = results["validation_results"]

    print("\n" + "=" * 80)
    print(f"VALIDATION REPORT: {extractor_name.upper()}")
    print("=" * 80)

    print(f"\nExtraction Statistics:")
    print(f"  Total abstracts:     {stats['total_abstracts']}")
    print(f"  Chunks processed:    {stats['chunks_processed']}/{stats['total_chunks']}")
    print(f"  Total relationships: {stats['total_relationships']}")
    print(f"  Valid relationships: {stats['valid_relationships']}")
    print(f"  Invalid:             {stats['invalid_relationships']}")
    print(f"  Validation rate:     {stats['validation_rate']:.1%}")

    print(f"\nValidation Failures:")
    print(f"  PMID failures:       {stats['pmid_failures']}")
    print(f"  Evidence failures:   {stats['evidence_failures']}")

    print(f"\nPerformance:")
    print(f"  Wall clock time:     {stats['wall_clock_ms']:.0f}ms")
    print(f"  Total latency:       {stats['total_latency_ms']:.0f}ms")
    print(f"  Throughput:          {stats['throughput_tok_per_sec']:.0f} tok/s")
    print(f"  Prompt tokens:       {stats['total_prompt_tokens']}")
    print(f"  Completion tokens:   {stats['total_completion_tokens']}")

    # Evidence statistics per abstract
    ev_stats = calculate_evidence_stats(results)
    print(f"\nEvidence Statistics Per Abstract:")
    print(f"  PMIDs with relations: {ev_stats['pmids_with_relationships']}")
    print(f"  Rels per PMID:        {ev_stats['rels_per_pmid_mean']:.2f} ± {ev_stats['rels_per_pmid_std']:.2f}")
    print(f"  Evidence per PMID:    {ev_stats['evidence_per_pmid_mean']:.2f} ± {ev_stats['evidence_per_pmid_std']:.2f}")

    # Show sample valid relationships
    valid_rels = results["valid_relationships"]
    if valid_rels:
        print(f"\nSample Valid Relationships (showing up to 5):")
        for rel in valid_rels[:5]:
            print(f"  - {rel['entity1']} --[{rel['relationship']}]--> {rel['entity2']}")
            print(f"    PMID: {rel['pmid']}, Confidence: {rel['confidence']:.2f}")
            evidence = rel['evidence_text'][:80] + "..." if len(rel['evidence_text']) > 80 else rel['evidence_text']
            print(f"    Evidence: {evidence}")

    # Show validation failures
    invalid_results = [v for v in validation_results if not v["is_valid"]]
    if invalid_results:
        print(f"\nValidation Failures (showing up to 5):")
        for v in invalid_results[:5]:
            rel = v["relationship"]
            print(f"  - {rel['entity1']} --[{rel['relationship']}]--> {rel['entity2']}")
            print(f"    PMID valid: {v['pmid_valid']}, Evidence valid: {v['evidence_valid']}")
            print(f"    Evidence similarity: {v['evidence_similarity']:.2f}")
            if v["error_message"]:
                err = v["error_message"][:100] + "..." if len(v["error_message"]) > 100 else v["error_message"]
                print(f"    Error: {err}")

    print("=" * 80)


async def run_test(
    extractor_name: str,
    articles: list[dict],
    annotations: dict[str, list[dict]],
    output_dir: Path,
    chunk_tokens: int | None = None,
) -> dict:
    """Run test with a specific extractor."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing BatchedLiteLLMMiner with: {extractor_name}")
    if chunk_tokens:
        logger.info(f"Chunk tokens override: {chunk_tokens}")
    logger.info(f"{'='*60}")

    miner = BatchedLiteLLMMiner(
        extractor_name=extractor_name,
        evidence_threshold=0.7,
        chunk_tokens=chunk_tokens,
    )

    results = await miner.run(articles, annotations)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"batched_miner_test_{extractor_name}_{timestamp}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Print validation report
    print_validation_report(results, extractor_name)

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Test BatchedLiteLLMMiner with provenance validation"
    )
    parser.add_argument(
        "--abstracts",
        type=int,
        default=20,
        help="Number of abstracts to process (default: 20)",
    )
    parser.add_argument(
        "--extractor",
        choices=["cerebras", "gemini", "all"],
        default="cerebras",
        help="Which extractor to test (default: cerebras)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/batched_miner"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        help="Override tokens per chunk (default: from config, ~5000)",
    )
    args = parser.parse_args()

    # Fetch test data
    articles, annotations = await fetch_test_data(args.abstracts)

    if not articles:
        logger.error("No articles fetched, cannot run test")
        return

    # Determine which extractors to run
    if args.extractor == "all":
        extractors = []
        if os.environ.get("OPENROUTER_API_KEY"):
            extractors.append("cerebras")
        if os.environ.get("GOOGLE_API_KEY"):
            extractors.append("gemini")
    else:
        extractors = [args.extractor]

    if not extractors:
        logger.error("No API keys configured for requested extractors")
        return

    # Run tests
    all_results = {}
    for extractor_name in extractors:
        try:
            results = await run_test(
                extractor_name,
                articles,
                annotations,
                args.output_dir,
                chunk_tokens=args.chunk_tokens,
            )
            all_results[extractor_name] = results
        except Exception as e:
            logger.error(f"Failed to test {extractor_name}: {e}")

    # Print comparison if multiple extractors
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"{'Metric':<30} ", end="")
        for name in all_results:
            print(f"{name:<20} ", end="")
        print()
        print("-" * 80)

        metrics = [
            ("Total relationships", "total_relationships"),
            ("Valid relationships", "valid_relationships"),
            ("Validation rate", "validation_rate"),
            ("PMID failures", "pmid_failures"),
            ("Evidence failures", "evidence_failures"),
            ("Wall clock (ms)", "wall_clock_ms"),
            ("Throughput (tok/s)", "throughput_tok_per_sec"),
        ]

        for label, key in metrics:
            print(f"{label:<30} ", end="")
            for name in all_results:
                val = all_results[name]["statistics"][key]
                if isinstance(val, float):
                    if key == "validation_rate":
                        print(f"{val:.1%:<20} ", end="")
                    else:
                        print(f"{val:>18.1f} ", end="")
                else:
                    print(f"{val:>18} ", end="")
            print()

        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
