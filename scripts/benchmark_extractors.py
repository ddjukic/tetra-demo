#!/usr/bin/env python3
"""
Benchmark script comparing extractor configurations from config.toml.

Compares throughput, latency, and cost for parsing abstracts using
configured extractors (cerebras, gemini, etc.).

Usage:
    # Run benchmark with default extractors
    uv run python scripts/benchmark_extractors.py

    # Run with custom number of abstracts
    uv run python scripts/benchmark_extractors.py --abstracts 50

    # Run specific extractor only
    uv run python scripts/benchmark_extractors.py --extractor cerebras
    uv run python scripts/benchmark_extractors.py --extractor gemini

    # Run all configured extractors
    uv run python scripts/benchmark_extractors.py --extractor all

Environment Variables:
    GOOGLE_API_KEY: Required for Gemini extractor
    OPENROUTER_API_KEY: Required for OpenRouter/Cerebras extractor
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from clients.pubmed_client import PubMedClient
from extraction import create_extractor, get_config, BatchMetrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single extractor benchmark run."""

    extractor_name: str
    model: str
    total_abstracts: int
    successful_extractions: int
    failed_extractions: int
    total_relationships: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    wall_clock_time_ms: float
    total_api_latency_ms: float
    throughput_tokens_per_second: float
    completion_throughput_tokens_per_second: float
    avg_latency_ms: float
    total_cost_usd: float
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_entity_key(entity_text: str, entity_id: str) -> str:
    """Build a normalized key for entity deduplication."""
    if entity_id and entity_id.strip():
        return entity_id.strip()
    return entity_text.strip().lower()


def build_entity_pairs_for_article(
    annotations: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Build entity pairs for relationship extraction from annotations."""
    entities: dict[str, str] = {}
    for annot in annotations:
        entity_text = annot.get("entity_text", "")
        entity_id = annot.get("entity_id", "")
        key = _build_entity_key(entity_text, entity_id)
        if key and key not in entities:
            entities[key] = entity_text

    entity_texts = list(entities.values())
    pairs: list[tuple[str, str]] = []
    for i, e1 in enumerate(entity_texts):
        for e2 in entity_texts[i + 1 :]:
            pairs.append((e1, e2))

    return pairs


async def fetch_test_data(num_abstracts: int = 100) -> tuple[list[dict], dict]:
    """
    Fetch test data from PubMed and PubTator.

    Returns:
        Tuple of (articles, annotations_by_pmid)
    """
    logger.info(f"Fetching {num_abstracts} abstracts from PubMed...")

    async with PubMedClient() as client:
        # Search for orexin-related papers (good for drug discovery domain)
        query = "orexin receptor AND 2018:2024[pdat]"
        pmids = await client.search(query, max_results=num_abstracts)

        if len(pmids) < num_abstracts:
            # Expand search if needed
            query2 = "(HCRTR1 OR HCRTR2) AND 2015:2024[pdat]"
            pmids2 = await client.search(
                query2, max_results=num_abstracts - len(pmids)
            )
            pmids = list(set(pmids + pmids2))[:num_abstracts]

        logger.info(f"Found {len(pmids)} PMIDs")

        # Fetch abstracts
        articles = await client.fetch_abstracts(pmids)
        logger.info(f"Fetched {len(articles)} articles with abstracts")

        # Get annotations from PubTator (uses same client)
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


async def benchmark_extractor(
    extractor_name: str,
    articles: list[dict],
    annotations_by_pmid: dict,
    max_concurrent: int | None = None,
) -> BenchmarkResult:
    """
    Benchmark a configured extractor.

    Args:
        extractor_name: Name of extractor in config (e.g., "cerebras", "gemini")
        articles: List of articles
        annotations_by_pmid: Dict of annotations
        max_concurrent: Max concurrent requests (None = use config default)

    Returns:
        BenchmarkResult with performance metrics
    """
    config = get_config()
    extractor_config = config.get_extractor(extractor_name)

    logger.info("=" * 60)
    logger.info(f"Benchmarking: {extractor_name} ({extractor_config.MODEL})")
    logger.info("=" * 60)

    extractor = create_extractor(extractor_name)

    # Prepare abstracts with entity pairs
    abstracts_with_pairs = []
    for article in articles:
        pmid = article.get("pmid", "")
        abstract = article.get("abstract", "")
        annotations = annotations_by_pmid.get(pmid, [])

        if not abstract or not annotations:
            continue

        entity_pairs = build_entity_pairs_for_article(annotations)
        if entity_pairs:
            abstracts_with_pairs.append(
                {
                    "pmid": pmid,
                    "abstract": abstract,
                    "entity_pairs": entity_pairs,
                }
            )

    total_abstracts = len(abstracts_with_pairs)
    logger.info(f"Processing {total_abstracts} abstracts with entity pairs")

    # Run batch extraction with timing
    start_wall_clock = time.time()

    all_relationships, batch_metrics = await extractor.batch_extract(
        abstracts_with_pairs,
        max_concurrent=max_concurrent,
    )

    wall_clock_ms = (time.time() - start_wall_clock) * 1000

    result = BenchmarkResult(
        extractor_name=extractor_name,
        model=extractor_config.MODEL,
        total_abstracts=total_abstracts,
        successful_extractions=batch_metrics.successful_extractions,
        failed_extractions=batch_metrics.failed_extractions,
        total_relationships=batch_metrics.total_relationships,
        total_prompt_tokens=batch_metrics.total_prompt_tokens,
        total_completion_tokens=batch_metrics.total_completion_tokens,
        total_tokens=batch_metrics.total_tokens,
        wall_clock_time_ms=wall_clock_ms,
        total_api_latency_ms=batch_metrics.total_latency_ms,
        throughput_tokens_per_second=batch_metrics.throughput_tokens_per_second,
        completion_throughput_tokens_per_second=batch_metrics.completion_throughput,
        avg_latency_ms=batch_metrics.avg_latency_ms,
        total_cost_usd=batch_metrics.total_cost_usd,
        timestamp=datetime.now().isoformat(),
    )

    logger.info(
        f"Completed: {batch_metrics.successful_extractions}/{total_abstracts} successful"
    )
    logger.info(f"Relationships found: {batch_metrics.total_relationships}")
    logger.info(f"Wall clock time: {wall_clock_ms:.0f}ms")
    logger.info(f"Avg latency: {batch_metrics.avg_latency_ms:.0f}ms")
    logger.info(f"Throughput: {batch_metrics.throughput_tokens_per_second:.0f} tok/s")
    logger.info(f"Completion throughput: {batch_metrics.completion_throughput:.0f} tok/s")
    logger.info(f"Estimated cost: ${batch_metrics.total_cost_usd:.4f}")

    return result


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print a comparison table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    # Header
    print(f"{'Metric':<40} ", end="")
    for r in results:
        print(f"{r.extractor_name[:20]:<22} ", end="")
    print()
    print("-" * 80)

    # Metrics
    metrics = [
        ("Total Abstracts", "total_abstracts"),
        ("Successful Extractions", "successful_extractions"),
        ("Failed Extractions", "failed_extractions"),
        ("Total Relationships", "total_relationships"),
        ("Total Tokens", "total_tokens"),
        ("Prompt Tokens", "total_prompt_tokens"),
        ("Completion Tokens", "total_completion_tokens"),
        ("Wall Clock Time (ms)", "wall_clock_time_ms"),
        ("API Latency (ms)", "total_api_latency_ms"),
        ("Avg Latency (ms)", "avg_latency_ms"),
        ("Throughput (tok/s)", "throughput_tokens_per_second"),
        ("Completion (tok/s)", "completion_throughput_tokens_per_second"),
        ("Est. Cost (USD)", "total_cost_usd"),
    ]

    for label, attr in metrics:
        print(f"{label:<40} ", end="")
        for r in results:
            val = getattr(r, attr)
            if isinstance(val, float):
                print(f"{val:>20,.2f} ", end="")
            else:
                print(f"{val:>20,} ", end="")
        print()

    # Calculate speedup if we have multiple results
    if len(results) >= 2:
        print("-" * 80)
        baseline = results[0]
        for r in results[1:]:
            if baseline.wall_clock_time_ms > 0:
                speedup = baseline.wall_clock_time_ms / r.wall_clock_time_ms
                print(
                    f"Speedup vs {baseline.extractor_name[:15]}: "
                    f"{speedup:.2f}x for {r.extractor_name}"
                )
            if baseline.throughput_tokens_per_second > 0:
                tp_ratio = (
                    r.throughput_tokens_per_second / baseline.throughput_tokens_per_second
                )
                print(f"Throughput ratio: {tp_ratio:.2f}x")

    print("=" * 80)


def save_results(results: list[BenchmarkResult], output_dir: Path) -> None:
    """Save benchmark results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_results_{timestamp}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [r.to_dict() for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to {output_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark relationship extraction with different LLM providers"
    )
    parser.add_argument(
        "--abstracts",
        type=int,
        default=100,
        help="Number of abstracts to process (default: 100)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=None,
        help="Max concurrent API requests (default: from config)",
    )
    parser.add_argument(
        "--extractor",
        choices=["all", "cerebras", "gemini", "gemini_pro", "openrouter_llama"],
        default="all",
        help="Which extractor(s) to benchmark (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmarks"),
        help="Output directory for results (default: results/benchmarks)",
    )
    args = parser.parse_args()

    # Fetch test data
    articles, annotations_by_pmid = await fetch_test_data(args.abstracts)

    if not articles:
        logger.error("No articles fetched, cannot run benchmark")
        return

    results: list[BenchmarkResult] = []
    config = get_config()

    # Determine which extractors to run
    if args.extractor == "all":
        # Run extractors that have required API keys
        extractors_to_run = []
        if os.environ.get("OPENROUTER_API_KEY"):
            extractors_to_run.append("cerebras")
        if os.environ.get("GOOGLE_API_KEY"):
            extractors_to_run.append("gemini")
    else:
        extractors_to_run = [args.extractor]

    # Run benchmarks
    for extractor_name in extractors_to_run:
        try:
            result = await benchmark_extractor(
                extractor_name,
                articles,
                annotations_by_pmid,
                max_concurrent=args.concurrent,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to benchmark {extractor_name}: {e}")

    if not results:
        logger.error("No benchmarks completed - check API keys")
        return

    # Print comparison
    print_comparison(results)

    # Save results
    save_results(results, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
