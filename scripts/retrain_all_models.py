#!/usr/bin/env python3
"""
Retrain all Node2Vec link prediction models with W&B tracking.

This script trains and evaluates all 3 model flavors with rigorous hard negative
evaluation to obtain realistic performance metrics.

Models:
1. Original (p=1.0, q=1.0) - Balanced random walk exploration
2. Structural (p=1.0, q=0.5) - DFS-like, captures structural roles and hub connectivity
3. Homophily (p=1.0, q=2.0) - BFS-like, captures local communities and clusters

Each model is evaluated against 4 negative sampling strategies:
- Random (baseline) - Uniform random non-edges (inflated metrics)
- 2-hop (structural hard) - Pairs at distance 2, share common neighbor
- Degree-matched - Match positive edge degree distribution
- Combined (most rigorous) - 50% 2-hop + 50% degree-matched

Usage:
    # Run all training with W&B tracking
    uv run python scripts/retrain_all_models.py --wandb

    # Run without W&B (for testing)
    uv run python scripts/retrain_all_models.py

    # Skip training, only evaluate existing models
    uv run python scripts/retrain_all_models.py --skip-training --eval-only

Expected Results (with hard negatives):
    | Model | Strategy | ROC-AUC | Note |
    |-------|----------|---------|------|
    | Original | random | 0.99 | Inflated |
    | Original | combined | 0.79-0.85 | Realistic |
    | Structural | combined | 0.78-0.84 | |
    | Homophily | combined | 0.80-0.86 | |
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Model configurations
MODEL_CONFIGS = [
    {
        "name": "original",
        "p": 1.0,
        "q": 1.0,
        "description": "Balanced exploration - DeepWalk-like random walks",
        "output": "models/link_predictor_original.pkl",
    },
    {
        "name": "structural",
        "p": 1.0,
        "q": 0.5,
        "description": "DFS-like - captures structural roles and hub connectivity",
        "output": "models/link_predictor_structural.pkl",
    },
    {
        "name": "homophily",
        "p": 1.0,
        "q": 2.0,
        "description": "BFS-like - captures local communities and clusters",
        "output": "models/link_predictor_homophily.pkl",
    },
]


def train_single_model(
    config: Dict,
    data_dir: str,
    wandb_enabled: bool = False,
    wandb_project: str = "tetra-link-prediction",
    eval_strategies: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Train a single model configuration with optional W&B tracking.

    Args:
        config: Model configuration dict
        data_dir: Path to STRING data
        wandb_enabled: Enable W&B tracking
        wandb_project: W&B project name
        eval_strategies: Evaluate with all negative sampling strategies
        seed: Random seed

    Returns:
        Results dictionary with metrics
    """
    from ml.link_predictor import LinkPredictor
    from ml.hard_negative_sampling import HardNegativeSampler, evaluate_with_hard_negatives

    print("\n" + "=" * 80)
    print(f"Training: {config['name']} (p={config['p']}, q={config['q']})")
    print(f"Description: {config['description']}")
    print("=" * 80)

    # Initialize W&B run if enabled
    run = None
    if wandb_enabled:
        try:
            import wandb
            run = wandb.init(
                project=wandb_project,
                name=f"node2vec_{config['name']}",
                notes=config["description"],
                config={
                    "model_variant": config["name"],
                    "p": config["p"],
                    "q": config["q"],
                    "embedding_dim": 128,
                    "walk_length": 80,
                    "num_walks": 10,
                    "min_score": 700,
                    "test_size": 0.2,
                    "seed": seed,
                },
                reinit=True,  # Allow multiple runs in same process
            )
            print(f"W&B run: {run.url}")
        except Exception as e:
            print(f"WARNING: W&B init failed: {e}")
            run = None

    start_time = time.time()

    # Initialize predictor
    predictor = LinkPredictor(
        embedding_dim=128,
        walk_length=80,
        num_walks=10,
        p=config["p"],
        q=config["q"],
        min_score=700,
        workers=4,
        seed=seed,
    )

    # Load data
    print("\nLoading STRING data...")
    predictor.load_string_data(data_dir)

    # Train
    print("\nTraining model...")
    metrics = predictor.train(test_size=0.2)
    train_time = time.time() - start_time

    results = {
        "model": config["name"],
        "config": config,
        "train_time_seconds": train_time,
        "train_metrics": {
            "roc_auc": metrics["auc"],
            "avg_precision": metrics["average_precision"],
            "train_size": metrics["train_size"],
            "test_size": metrics["test_size"],
        },
        "hard_neg_results": {},
    }

    # Log training metrics to W&B
    if run:
        import wandb
        wandb.log({
            "train/time_seconds": train_time,
            "train/train_size": metrics["train_size"],
            "train/test_size": metrics["test_size"],
            "metrics/random_roc_auc": metrics["auc"],
            "metrics/random_avg_precision": metrics["average_precision"],
        })

    # Hard negative evaluation
    if eval_strategies:
        print("\nRunning hard negative evaluation...")

        # Initialize sampler
        sampler = HardNegativeSampler(predictor.train_graph, seed=seed)

        # Get test edges
        train_edge_set = set(tuple(sorted(e)) for e in predictor.train_graph.edges())
        test_edges = []
        for edge in predictor.edge_set:
            if edge not in train_edge_set:
                n1, n2 = edge
                if n1 in predictor.node2vec_model.wv and n2 in predictor.node2vec_model.wv:
                    test_edges.append((n1, n2))

        # Sample test edges for evaluation (for speed)
        import random
        random.seed(seed)
        if len(test_edges) > 10000:
            test_edges = random.sample(test_edges, 10000)

        # Evaluate
        hard_neg_results = evaluate_with_hard_negatives(
            predictor,
            test_edges,
            predictor.node2vec_model,
            sampler,
            strategies=["random", "2hop", "degree_matched", "combined"]
        )

        results["hard_neg_results"] = hard_neg_results

        # Log to W&B
        if run:
            import wandb
            for strategy, strat_results in hard_neg_results.items():
                wandb.log({
                    f"metrics/{strategy}_roc_auc": strat_results["roc_auc"],
                    f"metrics/{strategy}_avg_precision": strat_results["avg_precision"],
                })

    # Save model
    print(f"\nSaving model to {config['output']}...")
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(config["output"])

    # Log model artifact to W&B
    if run:
        import wandb
        artifact = wandb.Artifact(
            name=f"link_predictor_{config['name']}",
            type="model",
            description=config["description"],
        )
        artifact.add_file(config["output"])
        run.log_artifact(artifact)
        run.finish()

    return results


def print_summary_table(all_results: List[Dict]) -> str:
    """
    Print and return a formatted summary table of results.

    Args:
        all_results: List of result dictionaries

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("COMPLETE RESULTS SUMMARY")
    lines.append("=" * 90)

    # Header
    header = f"{'Model':<12} {'Strategy':<18} {'ROC-AUC':<10} {'Avg Precision':<14} {'Note':<20}"
    lines.append(header)
    lines.append("-" * 90)

    for result in all_results:
        model_name = result["model"]

        # Training (random) results first
        train_auc = result["train_metrics"]["roc_auc"]
        train_ap = result["train_metrics"]["avg_precision"]
        lines.append(f"{model_name:<12} {'random (train)':<18} {train_auc:<10.4f} {train_ap:<14.4f} {'Inflated':<20}")

        # Hard negative results
        if result.get("hard_neg_results"):
            for strategy in ["random", "2hop", "degree_matched", "combined"]:
                if strategy in result["hard_neg_results"]:
                    strat_results = result["hard_neg_results"][strategy]
                    auc = strat_results["roc_auc"]
                    ap = strat_results["avg_precision"]

                    note = ""
                    if strategy == "random":
                        note = "Inflated"
                    elif strategy == "combined":
                        note = "Most rigorous"
                    elif strategy == "2hop":
                        note = "Structural hard"
                    elif strategy == "degree_matched":
                        note = "Controls bias"

                    lines.append(f"{'':<12} {strategy:<18} {auc:<10.4f} {ap:<14.4f} {note:<20}")

        lines.append("-" * 90)

    lines.append("=" * 90)
    table = "\n".join(lines)
    print(table)
    return table


def main():
    parser = argparse.ArgumentParser(
        description="Retrain all Node2Vec link prediction models with W&B tracking"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/string/",
        help="Directory containing STRING data files",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases tracking",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="tetra-link-prediction",
        help="W&B project name",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only evaluate existing models",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run hard negative evaluation on existing models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["original", "structural", "homophily"],
        default=["original", "structural", "homophily"],
        help="Which models to train (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/",
        help="Directory for saving results JSON",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Node2Vec Link Prediction - Full Retraining with Hard Negative Evaluation")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  W&B enabled: {args.wandb}")
    print(f"  W&B project: {args.wandb_project}")
    print(f"  Models to train: {args.models}")
    print(f"  Seed: {args.seed}")
    print()

    # Check data exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Login to W&B if enabled
    if args.wandb:
        try:
            import wandb
            print("W&B is enabled, runs will be tracked.")
        except ImportError:
            print("ERROR: wandb not installed. Install with: uv pip install wandb")
            sys.exit(1)

    # Filter configs based on selected models
    configs_to_run = [c for c in MODEL_CONFIGS if c["name"] in args.models]

    # Create results directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run training
    all_results = []
    total_start = time.time()

    for config in configs_to_run:
        try:
            result = train_single_model(
                config=config,
                data_dir=args.data_dir,
                wandb_enabled=args.wandb,
                wandb_project=args.wandb_project,
                eval_strategies=True,
                seed=args.seed,
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR training {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start

    # Print summary
    summary = print_summary_table(all_results)

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(args.output_dir) / f"training_results_{timestamp}.json"

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    json_results = {
        "timestamp": timestamp,
        "total_time_seconds": total_time,
        "total_time_minutes": total_time / 60,
        "models": [convert_types(r) for r in all_results],
        "summary_table": summary,
    }

    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Models trained: {len(all_results)}")
    print(f"Results saved to: {results_file}")

    if args.wandb:
        print(f"\nW&B Dashboard: https://wandb.ai/{args.wandb_project}")

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    for result in all_results:
        model = result["model"]
        random_auc = result["train_metrics"]["roc_auc"]
        if "combined" in result.get("hard_neg_results", {}):
            combined_auc = result["hard_neg_results"]["combined"]["roc_auc"]
            drop = random_auc - combined_auc
            print(f"\n{model.upper()} Model:")
            print(f"  Random ROC-AUC: {random_auc:.4f}")
            print(f"  Combined (hard) ROC-AUC: {combined_auc:.4f}")
            print(f"  Performance drop: {drop:.4f} ({drop/random_auc*100:.1f}%)")


if __name__ == "__main__":
    main()
