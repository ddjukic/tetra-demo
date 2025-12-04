#!/usr/bin/env python3
"""
Test link prediction on a knowledge graph for novel PPI predictions.

This script:
1. Builds a KG using the orchestrator (saves to disk)
2. Loads the graph back
3. Runs link prediction ensemble to find novel protein-protein interactions

Usage:
    uv run python scripts/test_link_prediction.py [--articles N] [--query "..."]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from models.knowledge_graph import KnowledgeGraph
from pipeline.orchestrator import KGOrchestrator
from pipeline.config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def build_and_save_graph(query: str, max_articles: int) -> tuple[KnowledgeGraph, str]:
    """Build a graph using orchestrator and return it with save path."""
    print(f"\n{'='*70}")
    print("STEP 1: Building Knowledge Graph")
    print(f"{'='*70}")
    print(f"Query: {query}")
    print(f"Max articles: {max_articles}")

    config = PipelineConfig(
        save_graph=True,
        graph_output_dir="graphs",
        graph_format="graphml",
        string_extend_network=10,
    )

    orchestrator = KGOrchestrator(config=config, extractor_name="cerebras")
    graph = await orchestrator.build(user_query=query, max_articles=max_articles)

    summary = graph.to_summary()
    print(f"\nGraph built:")
    print(f"  Nodes: {summary['node_count']}")
    print(f"  Edges: {summary['edge_count']}")
    print(f"  Entity types: {summary.get('entity_types', {})}")

    # Find the saved graph path
    graphs_dir = Path("graphs")
    if graphs_dir.exists():
        graph_files = sorted(graphs_dir.glob("*.graphml"), key=lambda p: p.stat().st_mtime, reverse=True)
        if graph_files:
            saved_path = str(graph_files[0])
            print(f"  Saved to: {saved_path}")
            return graph, saved_path

    return graph, ""


def test_load_graph(path: str) -> KnowledgeGraph:
    """Test loading graph from disk."""
    print(f"\n{'='*70}")
    print("STEP 2: Loading Graph from Disk")
    print(f"{'='*70}")

    loaded = KnowledgeGraph.load(path)
    summary = loaded.to_summary()

    print(f"Loaded graph from: {path}")
    print(f"  Nodes: {summary['node_count']}")
    print(f"  Edges: {summary['edge_count']}")

    return loaded


def get_confidence_tier(percentile: float) -> tuple[str, str]:
    """
    Map percentile to confidence tier with color coding.

    Returns:
        Tuple of (tier_name, description)
    """
    if percentile >= 99:
        return ("Very Strong", "Top 1% - highly likely interaction")
    elif percentile >= 95:
        return ("Strong", "Top 5% - strong prediction")
    elif percentile >= 90:
        return ("Moderate", "Top 10% - moderate confidence")
    elif percentile >= 75:
        return ("Weak", "Top 25% - weak signal")
    else:
        return ("Low", "Below top 25% - low confidence")


def run_link_prediction(graph: KnowledgeGraph, top_k: int = 20) -> list[dict]:
    """
    Run link prediction using NetworkX algorithms for novel PPI predictions.

    Returns predictions with interpretable percentile-based confidence scores.
    """
    import networkx as nx
    import numpy as np
    from itertools import combinations

    print(f"\n{'='*70}")
    print("STEP 3: Link Prediction for Novel PPIs")
    print(f"{'='*70}")

    # Get all protein nodes
    proteins = [
        node for node, attrs in graph.entities.items()
        if attrs.get("type", "").lower() in ("protein", "gene", "unknown")
    ]

    print(f"Found {len(proteins)} protein/gene nodes")

    if len(proteins) < 5:
        print("Not enough proteins for meaningful link prediction")
        return []

    # Convert to simple undirected graph for link prediction algorithms
    # (Link prediction doesn't work on multigraphs)
    G = nx.Graph(graph.graph.to_undirected())

    # Get existing edges
    existing_edges = set(G.edges())

    # Generate candidate pairs (proteins not currently connected)
    candidate_pairs = []
    for p1, p2 in combinations(proteins, 2):
        if p1 in G and p2 in G and (p1, p2) not in existing_edges and (p2, p1) not in existing_edges:
            candidate_pairs.append((p1, p2))

    print(f"Candidate pairs to evaluate: {len(candidate_pairs)}")

    if len(candidate_pairs) == 0:
        print("All proteins are already connected")
        return []

    # Run multiple link prediction methods and combine
    predictions = {}
    available_methods = ["jaccard", "adamic_adar", "preferential_attachment", "resource_allocation"]

    # 1. Jaccard Coefficient
    print("Running Jaccard Coefficient...")
    try:
        jc_preds = nx.jaccard_coefficient(G, candidate_pairs)
        for u, v, score in jc_preds:
            key = (u, v)
            if key not in predictions:
                predictions[key] = {"source": u, "target": v, "scores": {}}
            predictions[key]["scores"]["jaccard"] = score
    except Exception as e:
        print(f"  Jaccard failed: {e}")

    # 2. Adamic-Adar Index
    print("Running Adamic-Adar Index...")
    try:
        aa_preds = nx.adamic_adar_index(G, candidate_pairs)
        for u, v, score in aa_preds:
            key = (u, v)
            if key not in predictions:
                predictions[key] = {"source": u, "target": v, "scores": {}}
            predictions[key]["scores"]["adamic_adar"] = score
    except Exception as e:
        print(f"  Adamic-Adar failed: {e}")

    # 3. Preferential Attachment
    print("Running Preferential Attachment...")
    try:
        pa_preds = nx.preferential_attachment(G, candidate_pairs)
        for u, v, score in pa_preds:
            key = (u, v)
            if key not in predictions:
                predictions[key] = {"source": u, "target": v, "scores": {}}
            predictions[key]["scores"]["preferential_attachment"] = score
    except Exception as e:
        print(f"  Preferential Attachment failed: {e}")

    # 4. Resource Allocation Index
    print("Running Resource Allocation...")
    try:
        ra_preds = nx.resource_allocation_index(G, candidate_pairs)
        for u, v, score in ra_preds:
            key = (u, v)
            if key not in predictions:
                predictions[key] = {"source": u, "target": v, "scores": {}}
            predictions[key]["scores"]["resource_allocation"] = score
    except Exception as e:
        print(f"  Resource Allocation failed: {e}")

    # Compute ensemble scores for ALL predictions (for percentile calculation)
    print(f"\nComputing ensemble scores for {len(predictions)} predictions...")

    all_predictions = []
    for key, data in predictions.items():
        scores = data["scores"]
        if not scores:
            continue

        # Count methods with non-zero contribution
        contributing_methods = [m for m, s in scores.items() if s > 0]
        method_count = len(contributing_methods)

        # Sum scores (non-zero only)
        ensemble_score = sum(s for s in scores.values() if s > 0)

        all_predictions.append({
            "source": data["source"],
            "target": data["target"],
            "score": ensemble_score,
            "method_count": method_count,
            "contributing_methods": contributing_methods,
            "components": scores,
        })

    # Guard against empty predictions
    if not all_predictions:
        print("\nNo valid predictions to analyze.")
        return []

    # Build percentile lookup from ALL scores
    all_scores = np.array([p["score"] for p in all_predictions])
    sorted_scores = np.sort(all_scores)

    # Add percentile and confidence tier to each prediction
    for pred in all_predictions:
        # Percentile = what fraction of predictions have score <= this one
        n_scores = len(sorted_scores)
        percentile = 100 * np.searchsorted(sorted_scores, pred["score"]) / n_scores if n_scores > 0 else 0
        pred["percentile"] = round(percentile, 1)

        tier, description = get_confidence_tier(percentile)
        pred["confidence_tier"] = tier
        pred["confidence_description"] = description

    # Sort by ensemble score and return top_k
    all_predictions.sort(key=lambda x: x["score"], reverse=True)

    # Print distribution stats
    print(f"\nScore distribution across {len(all_predictions)} predictions:")
    print(f"  Median score: {np.median(all_scores):.2f}")
    print(f"  95th percentile threshold: {np.percentile(all_scores, 95):.2f}")
    print(f"  99th percentile threshold: {np.percentile(all_scores, 99):.2f}")

    return all_predictions[:top_k]


def display_predictions(predictions: list[dict], graph: KnowledgeGraph):
    """Display link prediction results with interpretable confidence scores."""
    print(f"\n{'='*90}")
    print("NOVEL PPI PREDICTIONS (Not in existing graph edges)")
    print(f"{'='*90}")

    if not predictions:
        print("No novel predictions found.")
        return

    # Get existing edges for comparison
    existing_edges = set()
    for u, v, _ in graph.graph.edges(keys=True):
        existing_edges.add((u, v))
        existing_edges.add((v, u))  # Bidirectional

    # Filter truly novel predictions
    novel = [p for p in predictions if (p["source"], p["target"]) not in existing_edges]

    print(f"\nFound {len(novel)} novel protein-protein interaction predictions:")
    print()
    print("Confidence Tiers: Very Strong (>99%) | Strong (>95%) | Moderate (>90%) | Weak (>75%)")
    print("-" * 90)
    print(f"{'Rank':<5}{'Source':<16}{'Target':<16}{'Score':>8}{'Pctl':>8}{'Confidence':<14}{'Methods':<10}")
    print("-" * 90)

    for i, pred in enumerate(novel[:20], 1):
        source = pred['source'][:15]
        target = pred['target'][:15]
        methods = pred.get('method_count', 0)
        percentile = pred.get('percentile', 0)
        tier = pred.get('confidence_tier', 'N/A')

        print(f"{i:<5}{source:<16}{target:<16}{pred['score']:>8.2f}{percentile:>7.1f}%  {tier:<14}{methods}/4")

    print("-" * 90)

    # Show detailed context for top predictions
    print("\n" + "="*90)
    print("DETAILED CONTEXT FOR TOP PREDICTIONS")
    print("="*90)

    for i, pred in enumerate(novel[:5], 1):
        source = pred["source"]
        target = pred["target"]
        percentile = pred.get('percentile', 0)
        tier = pred.get('confidence_tier', 'N/A')
        description = pred.get('confidence_description', '')
        methods = pred.get('contributing_methods', [])

        # Get neighbors of each
        source_neighbors = list(graph.graph.successors(source)) + list(graph.graph.predecessors(source))
        target_neighbors = list(graph.graph.successors(target)) + list(graph.graph.predecessors(target))

        common = set(source_neighbors) & set(target_neighbors)

        print(f"\n{i}. {source} <-> {target}")
        print(f"   Score: {pred['score']:.2f} | Percentile: {percentile:.1f}% | Confidence: {tier}")
        print(f"   Interpretation: {description}")
        print(f"   Contributing methods: {', '.join(methods)}")
        print(f"   {source} neighbors ({len(source_neighbors)}): {', '.join(source_neighbors[:5])}")
        print(f"   {target} neighbors ({len(target_neighbors)}): {', '.join(target_neighbors[:5])}")
        if common:
            print(f"   Common neighbors ({len(common)}): {', '.join(list(common)[:5])}")


async def main():
    parser = argparse.ArgumentParser(description="Test link prediction for novel PPIs")
    parser.add_argument("--articles", "-a", type=int, default=20,
                       help="Number of articles to fetch")
    parser.add_argument("--query", "-q", type=str,
                       default="Build a knowledge graph for orexin signaling pathway",
                       help="Query to build the graph")
    parser.add_argument("--load", "-l", type=str, default=None,
                       help="Load existing graph instead of building")
    parser.add_argument("--top-k", "-k", type=int, default=20,
                       help="Number of top predictions to return")

    args = parser.parse_args()

    print("=" * 70)
    print("LINK PREDICTION TEST FOR NOVEL PPI DISCOVERY")
    print("=" * 70)

    if args.load:
        # Load existing graph
        graph = test_load_graph(args.load)
        saved_path = args.load
    else:
        # Build new graph
        graph, saved_path = await build_and_save_graph(args.query, args.articles)

    # Test persistence by reloading if we saved
    if saved_path:
        loaded_graph = test_load_graph(saved_path)
    else:
        loaded_graph = graph

    # Run link prediction
    predictions = run_link_prediction(loaded_graph, top_k=args.top_k)

    # Display results
    display_predictions(predictions, loaded_graph)

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
