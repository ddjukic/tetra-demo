#!/usr/bin/env python3
"""
Training script for the PyTorch Geometric Link Predictor with MPS/GPU support.

This script trains Node2Vec embeddings using PyG's native implementation,
which supports MPS (Apple Silicon), CUDA, and CPU devices.

Usage:
    # Basic training on auto-detected device (MPS on Apple Silicon)
    uv run python scripts/train_pyg_link_predictor.py --data-dir data/string/ --output models/pyg_link_predictor.pkl

    # Training with W&B tracking
    uv run python scripts/train_pyg_link_predictor.py --data-dir data/string/ --wandb --epochs 100

    # Training with hard negative evaluation
    uv run python scripts/train_pyg_link_predictor.py --data-dir data/string/ --wandb --eval-all-strategies

    # Force CPU training
    uv run python scripts/train_pyg_link_predictor.py --data-dir data/string/ --device cpu
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def log_roc_curve(
    wandb,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    strategy_name: str,
    title: str = None,
):
    """
    Log ROC curve to W&B with proper formatting.

    Args:
        wandb: wandb module
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        strategy_name: Name for the strategy (used in key)
        title: Optional title for the plot
    """
    # wandb.plot.roc_curve expects y_probas as [n_samples, n_classes]
    # where columns are probabilities for each class
    y_probas = np.column_stack([1 - y_pred_proba, y_pred_proba])

    roc_plot = wandb.plot.roc_curve(
        y_true=y_true,
        y_probas=y_probas,
        labels=["No Edge", "Edge"],
        title=title or f"ROC Curve - {strategy_name}",
    )
    wandb.log({f"roc_curves/{strategy_name}": roc_plot})


def main():
    parser = argparse.ArgumentParser(
        description="Train PyG Link Predictor with MPS/GPU support for protein-protein interaction prediction"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/string/",
        help="Directory containing STRING data files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/pyg_link_predictor.pkl",
        help="Output path for trained model",
    )

    # Node2Vec hyperparameters
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
        "--walks-per-node",
        type=int,
        default=10,
        help="Number of walks per node (default: 10)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=10,
        help="Context window size (default: 10)",
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

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=700,
        help="Minimum STRING combined score (default: 700)",
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

    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to use for training (default: auto)",
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

    # Evaluation arguments
    parser.add_argument(
        "--eval-all-strategies",
        action="store_true",
        help="Evaluate with all negative sampling strategies",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="pyg",
        help="Model name for logging",
    )
    parser.add_argument(
        "--test-predictions",
        action="store_true",
        help="Run test predictions after training",
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
                model_variant = "pyg_structural"
            elif args.q > 1.0:
                model_variant = "pyg_homophily"

            run_name = args.wandb_name or f"pyg_node2vec_p{args.p}_q{args.q}_{model_variant}"
            notes = args.wandb_notes or f"PyG Node2Vec (MPS) with p={args.p}, q={args.q} ({model_variant})"

            run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                notes=notes,
                tags=["pyg", "node2vec", args.device],
                config={
                    "framework": "pytorch_geometric",
                    "p": args.p,
                    "q": args.q,
                    "embedding_dim": args.embedding_dim,
                    "walk_length": args.walk_length,
                    "walks_per_node": args.walks_per_node,
                    "context_size": args.context_size,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "min_score": args.min_score,
                    "test_size": args.test_size,
                    "seed": args.seed,
                    "device": args.device,
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

    # Import here to allow --help without torch installed
    from ml.pyg_link_predictor import PyGLinkPredictor

    print("=" * 60)
    print("PyG Link Predictor Training (MPS/GPU Accelerated)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output model: {args.output}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Walk length: {args.walk_length}")
    print(f"  Walks per node: {args.walks_per_node}")
    print(f"  Context size: {args.context_size}")
    print(f"  p: {args.p}, q: {args.q}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Min score: {args.min_score}")
    print(f"  Test size: {args.test_size}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {args.device}")
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
    predictor = PyGLinkPredictor(
        embedding_dim=args.embedding_dim,
        walk_length=args.walk_length,
        walks_per_node=args.walks_per_node,
        context_size=args.context_size,
        p=args.p,
        q=args.q,
        min_score=args.min_score,
        device=args.device,
        seed=args.seed,
    )

    # Log device info
    if args.wandb and run:
        import wandb
        wandb.config.update({"actual_device": str(predictor.device)})

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

    # Train with proper validation
    print("\n" + "=" * 60)
    print("Step 2: Training with Proper Validation (No Data Leakage)")
    print("=" * 60)
    step_start = time.time()

    metrics = predictor.train_with_validation(
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True,
    )

    train_time = time.time() - step_start
    print(f"Training completed in {train_time:.1f}s")

    # Log metrics to W&B
    if args.wandb and run:
        import wandb

        wandb.log({
            "train/time_seconds": train_time,
            "train/embedding_time_seconds": metrics["embedding_time_seconds"],
            "train/train_size": metrics["train_size"],
            "train/test_size": metrics["test_size"],
            "train/final_loss": metrics["final_loss"],
            "metrics/random_roc_auc": metrics["auc"],
            "metrics/random_avg_precision": metrics["average_precision"],
        })

        # Log loss curve
        for epoch, loss in enumerate(metrics["loss_history"], 1):
            wandb.log({"train/loss": loss, "epoch": epoch})

        # Log ROC curve with proper format
        if "_y_test" in metrics and "_y_pred_proba" in metrics:
            log_roc_curve(
                wandb,
                np.array(metrics["_y_test"]),
                np.array(metrics["_y_pred_proba"]),
                "random",
                title="ROC Curve - Random Negatives"
            )

    # Hard negative evaluation
    hard_neg_results = {}
    if args.eval_all_strategies:
        print("\n" + "=" * 60)
        print("Step 3: Hard Negative Evaluation")
        print("=" * 60)

        from ml.hard_negative_sampling import HardNegativeSampler

        # Initialize sampler with training graph
        sampler = HardNegativeSampler(predictor.train_graph, seed=args.seed)

        # Get test edges
        train_edge_set = set(tuple(sorted(e)) for e in predictor.train_graph.edges())
        test_edges = []
        for edge in predictor.edge_set:
            if edge not in train_edge_set:
                n1, n2 = edge
                if n1 in predictor.node_to_idx and n2 in predictor.node_to_idx:
                    test_edges.append((n1, n2))

        print(f"Found {len(test_edges)} test edges with embeddings")

        # Limit test edges
        max_test = 10000
        if len(test_edges) > max_test:
            import random
            random.seed(args.seed)
            test_edges = random.sample(test_edges, max_test)
            print(f"Sampled {max_test} test edges for evaluation")

        # Evaluate each strategy
        strategies = ["random", "2hop", "degree_matched", "combined"]

        for strategy in strategies:
            print(f"\nEvaluating with {strategy} negatives...")

            # Sample negatives
            if strategy == "random":
                negatives = sampler.sample_random_negatives(len(test_edges))
            elif strategy == "2hop":
                negatives = sampler.sample_2hop_negatives(len(test_edges))
            elif strategy == "degree_matched":
                negatives = sampler.sample_degree_matched_negatives(test_edges, len(test_edges))
            elif strategy == "combined":
                negatives = sampler.sample_combined_hard_negatives(test_edges, len(test_edges))

            # Build features
            X_pos, X_neg = [], []
            for n1, n2 in test_edges:
                features = predictor._get_edge_features(n1, n2)
                if features is not None:
                    X_pos.append(features)

            for n1, n2 in negatives:
                if n1 in predictor.node_to_idx and n2 in predictor.node_to_idx:
                    features = predictor._get_edge_features(n1, n2)
                    if features is not None:
                        X_neg.append(features)

            # Combine
            X = np.array(X_pos + X_neg)
            y_true = np.array([1] * len(X_pos) + [0] * len(X_neg))

            # Predict
            from sklearn.metrics import roc_auc_score, average_precision_score
            y_pred = predictor.classifier.predict_proba(X)[:, 1]

            auc = roc_auc_score(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)

            hard_neg_results[strategy] = {
                "roc_auc": auc,
                "avg_precision": ap,
                "n_positives": len(X_pos),
                "n_negatives": len(X_neg),
                "y_true": y_true,
                "y_pred": y_pred,
            }

            print(f"  {strategy}: ROC-AUC={auc:.4f}, AP={ap:.4f}")

            # Log to W&B
            if args.wandb and run:
                import wandb
                wandb.log({
                    f"metrics/{strategy}_roc_auc": auc,
                    f"metrics/{strategy}_avg_precision": ap,
                })
                # Log ROC curve
                log_roc_curve(
                    wandb,
                    y_true,
                    y_pred,
                    strategy,
                    title=f"ROC Curve - {strategy.replace('_', ' ').title()}"
                )

        # Print summary table
        print("\n" + "=" * 60)
        print("Hard Negative Evaluation Summary")
        print("=" * 60)
        print(f"{'Strategy':<20} {'ROC-AUC':<12} {'Avg Precision':<12}")
        print("-" * 44)
        for strategy in strategies:
            if strategy in hard_neg_results:
                r = hard_neg_results[strategy]
                print(f"{strategy:<20} {r['roc_auc']:.4f}       {r['avg_precision']:.4f}")

    # Save model
    print("\n" + "=" * 60)
    print("Step 4: Saving Model")
    print("=" * 60)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictor.save(args.output)

    if args.wandb and run:
        import wandb
        artifact = wandb.Artifact(
            name=f"pyg_link_predictor_p{args.p}_q{args.q}",
            type="model",
            description=f"PyG Node2Vec link predictor with p={args.p}, q={args.q}",
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
    print(f"Device: {predictor.device}")
    print(f"\nBaseline Metrics (Random Negatives):")
    print(f"  ROC-AUC: {metrics['auc']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
    print(f"  Embedding training time: {metrics['embedding_time_seconds']:.1f}s")

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
    print('  from ml.pyg_link_predictor import PyGLinkPredictor')
    print(f'  predictor = PyGLinkPredictor.load("{args.output}")')
    print('  results = predictor.predict([("BRCA1", "TP53")])')


if __name__ == "__main__":
    main()
