#!/usr/bin/env python3
"""
Training script for the Link Predictor model.

Usage:
    uv run python scripts/train_link_predictor.py --data-dir data/string/ --output models/link_predictor.pkl
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.link_predictor import LinkPredictor


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

    args = parser.parse_args()

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
    print(f"Data loading completed in {time.time() - step_start:.1f}s")

    # Train with proper validation (edges split BEFORE embedding training)
    print("\n" + "=" * 60)
    print("Step 2: Training with Proper Validation (No Data Leakage)")
    print("=" * 60)
    step_start = time.time()
    metrics = predictor.train(test_size=args.test_size)
    print(f"Training completed in {time.time() - step_start:.1f}s")

    # Save model
    print("\n" + "=" * 60)
    print("Step 3: Saving Model")
    print("=" * 60)
    predictor.save(args.output)

    # Test predictions
    if args.test_predictions:
        print("\n" + "=" * 60)
        print("Step 4: Test Predictions")
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
    print(f"\nMetrics:")
    print(f"  ROC-AUC: {metrics['auc']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
    print(f"\nModel saved to: {args.output}")
    print("\nUsage example:")
    print('  from ml.link_predictor import LinkPredictor')
    print(f'  predictor = LinkPredictor.load("{args.output}")')
    print('  results = predictor.predict([("BRCA1", "TP53")])')


if __name__ == "__main__":
    main()
