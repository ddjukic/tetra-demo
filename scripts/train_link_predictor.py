#!/usr/bin/env python3
"""
Training script for the Link Predictor model with W&B tracking and hard negative evaluation.

Usage:
    # Basic training (no W&B)
    uv run python scripts/train_link_predictor.py --data-dir data/string/ --output models/link_predictor.pkl

    # Training with W&B tracking
    uv run python scripts/train_link_predictor.py --data-dir data/string/ --output models/link_predictor.pkl --wandb

    # Training with hard negative evaluation
    uv run python scripts/train_link_predictor.py --data-dir data/string/ --output models/link_predictor.pkl --wandb --eval-all-strategies
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.link_predictor import LinkPredictor
from ml.hard_negative_sampling import HardNegativeSampler, evaluate_with_hard_negatives


def main():
    parser = argparse.ArgumentParser(
        description="Train Link Predictor for protein-protein interaction prediction"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/string/",
        help="Directory containing STRING data files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/link_predictor.pkl",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--walk-length",
        type=int,
        default=80,
        help="Random walk length (default: 80)",
    )
    parser.add_argument(
        "--num-walks",
        type=int,
        default=10,
        help="Number of walks per node (default: 10)",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=1.0,
        help="Node2Vec return parameter (default: 1.0)",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=1.0,
        help="Node2Vec in-out parameter (default: 1.0)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=700,
        help="Minimum STRING combined score (default: 700)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--test-predictions",
        action="store_true",
        help="Run test predictions after training",
    )
    # W&B arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases tracking",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="tetra-link-prediction",
        help="W&B project name (default: tetra-link-prediction)",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)",
    )
    parser.add_argument(
        "--wandb-notes",
        type=str,
        default=None,
        help="W&B run notes",
    )
    # Hard negative evaluation
    parser.add_argument(
        "--eval-all-strategies",
        action="store_true",
        help="Evaluate with all negative sampling strategies (random, 2hop, degree-matched, combined)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="original",
        help="Model name for logging (original, structural, homophily)",
    )

    args = parser.parse_args()

    # Initialize W&B if enabled
    run = None
    if args.wandb:
        try:
            import wandb

            # Determine model variant name
            model_variant = args.model_name
            if args.q < 1.0:
                model_variant = "structural"
            elif args.q > 1.0:
                model_variant = "homophily"

            run_name = args.wandb_name or f"node2vec_p{args.p}_q{args.q}_{model_variant}"
            notes = args.wandb_notes or f"Node2Vec link prediction with p={args.p}, q={args.q} ({model_variant})"

            run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                notes=notes,
                config={
                    "p": args.p,
                    "q": args.q,
                    "embedding_dim": args.embedding_dim,
                    "walk_length": args.walk_length,
                    "num_walks": args.num_walks,
                    "min_score": args.min_score,
                    "test_size": args.test_size,
                    "workers": args.workers,
                    "seed": args.seed,
                    "model_variant": model_variant,
                },
            )
            print(f"W&B tracking enabled: {run.url}")
        except ImportError:
            print("WARNING: wandb not installed. Install with: uv pip install wandb")
            args.wandb = False
        except Exception as e:
            print(f"WARNING: Failed to initialize W&B: {e}")
            args.wandb = False

    print("=" * 60)
    print("Link Predictor Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output model: {args.output}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Walk length: {args.walk_length}")
    print(f"  Num walks: {args.num_walks}")
    print(f"  p: {args.p}, q: {args.q}")
    print(f"  Min score: {args.min_score}")
    print(f"  Workers: {args.workers}")
    print(f"  Test size: {args.test_size}")
    print(f"  Seed: {args.seed}")
    print(f"  W&B enabled: {args.wandb}")
    print(f"  Eval all strategies: {args.eval_all_strategies}")
    print()

    # Verify data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Check for required files
    required_files = [
        "9606.protein.physical.links.detailed.v12.0.txt.gz",
        "9606.protein.info.v12.0.txt.gz",
        "9606.protein.aliases.v12.0.txt.gz",
    ]
    for f in required_files:
        if not (data_path / f).exists():
            print(f"ERROR: Required file not found: {data_path / f}")
            sys.exit(1)

    start_time = time.time()

    # Initialize predictor
    predictor = LinkPredictor(
        embedding_dim=args.embedding_dim,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        p=args.p,
        q=args.q,
        min_score=args.min_score,
        workers=args.workers,
        seed=args.seed,
    )

    # Load STRING data
    print("\n" + "=" * 60)
    print("Step 1: Loading STRING Data")
    print("=" * 60)
    step_start = time.time()
    predictor.load_string_data(args.data_dir)
    load_time = time.time() - step_start
    print(f"Data loading completed in {load_time:.1f}s")

    if args.wandb and run:
        import wandb
        wandb.log({
            "data/load_time_seconds": load_time,
            "data/total_edges": len(predictor._all_edges),
            "data/total_nodes": predictor.graph.number_of_nodes(),
        })

    # Train with proper validation (edges split BEFORE embedding training)
    print("\n" + "=" * 60)
    print("Step 2: Training with Proper Validation (No Data Leakage)")
    print("=" * 60)
    step_start = time.time()
    metrics = predictor.train(test_size=args.test_size)
    train_time = time.time() - step_start
    print(f"Training completed in {train_time:.1f}s")

    if args.wandb and run:
        import wandb
        wandb.log({
            "train/time_seconds": train_time,
            "train/train_size": metrics["train_size"],
            "train/test_size": metrics["test_size"],
            "metrics/random_roc_auc": metrics["auc"],
            "metrics/random_avg_precision": metrics["average_precision"],
        })

    # Hard negative evaluation
    hard_neg_results = {}
    if args.eval_all_strategies:
        print("\n" + "=" * 60)
        print("Step 3: Hard Negative Evaluation")
        print("=" * 60)

        # Initialize sampler with training graph
        sampler = HardNegativeSampler(predictor.train_graph, seed=args.seed)

        # Get test edges (edges in full graph but not in training graph)
        train_edge_set = set(tuple(sorted(e)) for e in predictor.train_graph.edges())
        test_edges = []
        for edge in predictor.edge_set:
            if edge not in train_edge_set:
                n1, n2 = edge
                # Only include edges where both nodes have embeddings
                if n1 in predictor.node2vec_model.wv and n2 in predictor.node2vec_model.wv:
                    test_edges.append((n1, n2))

        print(f"Found {len(test_edges)} test edges with embeddings")

        # Limit test edges for reasonable evaluation time
        max_test = 10000
        if len(test_edges) > max_test:
            import random
            random.seed(args.seed)
            test_edges = random.sample(test_edges, max_test)
            print(f"Sampled {max_test} test edges for evaluation")

        # Evaluate with all strategies
        hard_neg_results = evaluate_with_hard_negatives(
            predictor,
            test_edges,
            predictor.node2vec_model,
            sampler,
            strategies=["random", "2hop", "degree_matched", "combined"]
        )

        if args.wandb and run:
            import wandb
            for strategy, results in hard_neg_results.items():
                wandb.log({
                    f"metrics/{strategy}_roc_auc": results["roc_auc"],
                    f"metrics/{strategy}_avg_precision": results["avg_precision"],
                })

        # Print summary table
        print("\n" + "=" * 60)
        print("Hard Negative Evaluation Summary")
        print("=" * 60)
        print(f"{'Strategy':<20} {'ROC-AUC':<12} {'Avg Precision':<12}")
        print("-" * 44)
        for strategy in ["random", "2hop", "degree_matched", "combined"]:
            if strategy in hard_neg_results:
                r = hard_neg_results[strategy]
                print(f"{strategy:<20} {r['roc_auc']:.4f}       {r['avg_precision']:.4f}")

    # Save model
    print("\n" + "=" * 60)
    print("Step 4: Saving Model")
    print("=" * 60)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictor.save(args.output)

    if args.wandb and run:
        import wandb
        # Save model as artifact
        artifact = wandb.Artifact(
            name=f"link_predictor_p{args.p}_q{args.q}",
            type="model",
            description=f"Node2Vec link predictor with p={args.p}, q={args.q}",
        )
        artifact.add_file(args.output)
        run.log_artifact(artifact)

    # Test predictions
    if args.test_predictions:
        print("\n" + "=" * 60)
        print("Step 5: Test Predictions")
        print("=" * 60)

        test_pairs = [
            ("BRCA1", "TP53"),
            ("HCRTR1", "HCRTR2"),
            ("EGFR", "ERBB2"),
            ("INS", "INSR"),
            ("CDK2", "CCNA2"),
        ]

        print("\nPredictions for sample protein pairs:")
        results = predictor.predict(test_pairs)
        for r in results:
            status = "KNOWN" if r["in_string"] else "NOVEL"
            if r.get("error"):
                print(f"  {r['protein1']} <-> {r['protein2']}: ERROR - {r['error']}")
            else:
                print(f"  {r['protein1']} <-> {r['protein2']}: score={r['ml_score']:.4f} [{status}]")

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\nBaseline Metrics (Random Negatives):")
    print(f"  ROC-AUC: {metrics['auc']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")

    if hard_neg_results:
        print(f"\nHard Negative Metrics:")
        if "combined" in hard_neg_results:
            print(f"  Combined ROC-AUC: {hard_neg_results['combined']['roc_auc']:.4f}")
            print(f"  Combined Avg Precision: {hard_neg_results['combined']['avg_precision']:.4f}")

    print(f"\nModel saved to: {args.output}")

    if args.wandb and run:
        import wandb
        wandb.log({"train/total_time_seconds": total_time})
        run.finish()
        print(f"\nW&B run completed: {run.url}")

    print("\nUsage example:")
    print('  from ml.link_predictor import LinkPredictor')
    print(f'  predictor = LinkPredictor.load("{args.output}")')
    print('  results = predictor.predict([("BRCA1", "TP53")])')


if __name__ == "__main__":
    main()
