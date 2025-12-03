"""
Hard Negative Sampling Strategies for Link Prediction Evaluation.

This module implements various negative sampling strategies to provide
more rigorous evaluation of link prediction models.

The problem with random negative sampling:
- Random node pairs are trivially distinguishable from real edges
- High-degree nodes appear disproportionately in positive edges
- Results in inflated ROC-AUC scores (often 0.99+)

Hard negative strategies implemented:
1. Random (baseline) - Uniform random non-edges
2. 2-hop - Pairs at graph distance 2 (share common neighbor but no direct edge)
3. Degree-matched - Match degree distribution of positive edges
4. Combined - 50% 2-hop + 50% degree-matched for most rigorous evaluation

References:
- "Implicit degree bias in the link prediction task" (arXiv 2024)
- "Negative sampling strategies impact biomolecular network predictions" (BMC Bio 2025)
- "Bias-aware training and evaluation of link prediction" (PNAS 2025)
"""

import random
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional

import networkx as nx
import numpy as np
from tqdm import tqdm


class HardNegativeSampler:
    """
    Implements multiple negative sampling strategies for link prediction.

    Strategies range from trivial (random) to challenging (2-hop, degree-matched).
    Using harder negatives provides more realistic performance estimates.
    """

    def __init__(self, graph: nx.Graph, seed: int = 42):
        """
        Initialize the sampler with a graph.

        Args:
            graph: NetworkX graph to sample negatives from
            seed: Random seed for reproducibility
        """
        self.graph = graph
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # Precompute essential graph properties
        self.nodes = list(graph.nodes())
        self.node_set = set(self.nodes)
        self.edge_set = set(tuple(sorted(e)) for e in graph.edges())
        self.num_nodes = len(self.nodes)
        self.num_edges = len(self.edge_set)

        # Precompute degree information
        self._degrees: Dict[str, int] = dict(graph.degree())
        self._degree_to_nodes: Dict[int, List[str]] = defaultdict(list)
        for node, degree in self._degrees.items():
            self._degree_to_nodes[degree].append(node)

        # Cache for 2-hop neighbors (lazily computed)
        self._2hop_cache: Dict[str, Set[str]] = {}

        print(f"HardNegativeSampler initialized:")
        print(f"  Nodes: {self.num_nodes:,}")
        print(f"  Edges: {self.num_edges:,}")
        print(f"  Degree range: {min(self._degrees.values())} - {max(self._degrees.values())}")

    def _get_2hop_neighbors(self, node: str) -> Set[str]:
        """
        Get all nodes at exactly distance 2 from the given node.

        These are nodes that share a common neighbor with the source
        but are not directly connected.

        Args:
            node: Source node

        Returns:
            Set of nodes at distance 2
        """
        if node in self._2hop_cache:
            return self._2hop_cache[node]

        # Get 1-hop neighbors
        neighbors_1hop = set(self.graph.neighbors(node))

        # Get 2-hop neighbors (neighbors of neighbors, excluding self and 1-hop)
        neighbors_2hop = set()
        for n1 in neighbors_1hop:
            for n2 in self.graph.neighbors(n1):
                if n2 != node and n2 not in neighbors_1hop:
                    neighbors_2hop.add(n2)

        self._2hop_cache[node] = neighbors_2hop
        return neighbors_2hop

    def sample_random_negatives(
        self,
        n: int,
        exclude_edges: Optional[Set[Tuple[str, str]]] = None
    ) -> List[Tuple[str, str]]:
        """
        Sample random negative edges (baseline approach).

        This is the typical approach that creates artificially easy negatives.

        Args:
            n: Number of negatives to sample
            exclude_edges: Additional edges to exclude (e.g., test edges)

        Returns:
            List of (node1, node2) tuples representing negative edges
        """
        exclude = self.edge_set.copy()
        if exclude_edges:
            exclude.update(exclude_edges)

        negatives = []
        seen = set()

        with tqdm(total=n, desc="Random negatives") as pbar:
            while len(negatives) < n:
                # Sample two random nodes
                idx = self.rng.integers(0, self.num_nodes, size=2)
                n1, n2 = self.nodes[idx[0]], self.nodes[idx[1]]

                if n1 == n2:
                    continue

                edge = tuple(sorted([n1, n2]))
                if edge not in exclude and edge not in seen:
                    negatives.append((n1, n2))
                    seen.add(edge)
                    pbar.update(1)

        return negatives

    def sample_2hop_negatives(
        self,
        n: int,
        exclude_edges: Optional[Set[Tuple[str, str]]] = None
    ) -> List[Tuple[str, str]]:
        """
        Sample structural hard negatives from 2-hop neighborhood.

        These pairs share a common neighbor but don't directly interact.
        This is much harder than random - forces learning fine-grained
        patterns rather than simple degree heuristics.

        Args:
            n: Number of negatives to sample
            exclude_edges: Additional edges to exclude

        Returns:
            List of (node1, node2) tuples representing hard negative edges
        """
        exclude = self.edge_set.copy()
        if exclude_edges:
            exclude.update(exclude_edges)

        negatives = []
        seen = set()

        # Precompute 2-hop neighbors for high-degree nodes (more likely to be sampled)
        print("Precomputing 2-hop neighborhoods for sampling...")
        high_degree_nodes = sorted(
            self.nodes,
            key=lambda x: self._degrees.get(x, 0),
            reverse=True
        )[:min(1000, self.num_nodes)]

        for node in tqdm(high_degree_nodes, desc="Caching 2-hop"):
            self._get_2hop_neighbors(node)

        # Sample from 2-hop pairs
        with tqdm(total=n, desc="2-hop negatives") as pbar:
            attempts = 0
            max_attempts = n * 100  # Prevent infinite loop

            while len(negatives) < n and attempts < max_attempts:
                attempts += 1

                # Pick a random source node
                n1_idx = self.rng.integers(0, self.num_nodes)
                n1 = self.nodes[n1_idx]

                # Get its 2-hop neighbors
                neighbors_2hop = self._get_2hop_neighbors(n1)
                if not neighbors_2hop:
                    continue

                # Pick a random 2-hop neighbor
                n2 = random.choice(list(neighbors_2hop))

                edge = tuple(sorted([n1, n2]))
                if edge not in exclude and edge not in seen:
                    negatives.append((n1, n2))
                    seen.add(edge)
                    pbar.update(1)

        if len(negatives) < n:
            print(f"Warning: Only found {len(negatives)}/{n} 2-hop negatives")
            # Fill remaining with random negatives
            remaining = n - len(negatives)
            additional = self.sample_random_negatives(
                remaining,
                exclude_edges=exclude.union(seen)
            )
            negatives.extend(additional)

        return negatives

    def sample_degree_matched_negatives(
        self,
        positive_edges: List[Tuple[str, str]],
        n: int,
        exclude_edges: Optional[Set[Tuple[str, str]]] = None
    ) -> List[Tuple[str, str]]:
        """
        Sample negatives matching the degree distribution of positive edges.

        This controls for the "high degree = edge" shortcut that inflates
        random negative evaluation.

        Args:
            positive_edges: List of positive edges to match distribution of
            n: Number of negatives to sample
            exclude_edges: Additional edges to exclude

        Returns:
            List of (node1, node2) tuples with matched degree distribution
        """
        exclude = self.edge_set.copy()
        if exclude_edges:
            exclude.update(exclude_edges)

        # Compute degree distribution of positive edges
        pos_degrees = []
        for n1, n2 in positive_edges:
            d1 = self._degrees.get(n1, 0)
            d2 = self._degrees.get(n2, 0)
            pos_degrees.append((d1, d2))

        # Create degree buckets with some tolerance
        degree_pairs = defaultdict(list)
        for n1, n2 in positive_edges:
            d1, d2 = self._degrees.get(n1, 0), self._degrees.get(n2, 0)
            # Use coarser buckets (log-scale) for better matching
            bucket1 = int(np.log2(d1 + 1))
            bucket2 = int(np.log2(d2 + 1))
            bucket = tuple(sorted([bucket1, bucket2]))
            degree_pairs[bucket].append((d1, d2))

        # Build degree bucket mapping
        bucket_to_nodes: Dict[int, List[str]] = defaultdict(list)
        for node in self.nodes:
            d = self._degrees.get(node, 0)
            bucket = int(np.log2(d + 1))
            bucket_to_nodes[bucket].append(node)

        negatives = []
        seen = set()

        # Sample positive edge degree pairs and find matching negatives
        with tqdm(total=n, desc="Degree-matched negatives") as pbar:
            sample_idx = 0
            attempts = 0
            max_attempts = n * 100

            while len(negatives) < n and attempts < max_attempts:
                attempts += 1

                # Get target degree distribution from a positive edge
                d1, d2 = pos_degrees[sample_idx % len(pos_degrees)]
                sample_idx += 1

                # Find nodes with similar degrees
                bucket1 = int(np.log2(d1 + 1))
                bucket2 = int(np.log2(d2 + 1))

                candidates1 = bucket_to_nodes.get(bucket1, [])
                candidates2 = bucket_to_nodes.get(bucket2, [])

                if not candidates1 or not candidates2:
                    continue

                # Sample from candidates
                n1 = random.choice(candidates1)
                n2 = random.choice(candidates2)

                if n1 == n2:
                    continue

                edge = tuple(sorted([n1, n2]))
                if edge not in exclude and edge not in seen:
                    negatives.append((n1, n2))
                    seen.add(edge)
                    pbar.update(1)

        if len(negatives) < n:
            print(f"Warning: Only found {len(negatives)}/{n} degree-matched negatives")
            remaining = n - len(negatives)
            additional = self.sample_random_negatives(
                remaining,
                exclude_edges=exclude.union(seen)
            )
            negatives.extend(additional)

        return negatives

    def sample_combined_hard_negatives(
        self,
        positive_edges: List[Tuple[str, str]],
        n: int,
        exclude_edges: Optional[Set[Tuple[str, str]]] = None
    ) -> List[Tuple[str, str]]:
        """
        Sample combined hard negatives: 50% 2-hop + 50% degree-matched.

        This is the most rigorous evaluation strategy, combining structural
        and statistical difficulty.

        Args:
            positive_edges: Positive edges for degree matching
            n: Total number of negatives to sample
            exclude_edges: Additional edges to exclude

        Returns:
            List of hard negative edges
        """
        exclude = self.edge_set.copy()
        if exclude_edges:
            exclude.update(exclude_edges)

        n_2hop = n // 2
        n_degree = n - n_2hop

        print(f"Sampling combined hard negatives: {n_2hop} 2-hop + {n_degree} degree-matched")

        # Sample 2-hop negatives
        negatives_2hop = self.sample_2hop_negatives(n_2hop, exclude_edges=exclude)
        seen_edges = set(tuple(sorted(e)) for e in negatives_2hop)

        # Sample degree-matched negatives (exclude 2-hop to avoid duplicates)
        negatives_degree = self.sample_degree_matched_negatives(
            positive_edges,
            n_degree,
            exclude_edges=exclude.union(seen_edges)
        )

        # Combine and shuffle
        all_negatives = negatives_2hop + negatives_degree
        random.shuffle(all_negatives)

        return all_negatives

    def compare_degree_distributions(
        self,
        positive_edges: List[Tuple[str, str]],
        negative_edges: List[Tuple[str, str]],
        strategy_name: str = "Unknown"
    ) -> Dict[str, float]:
        """
        Compare degree distributions between positive and negative edges.

        This diagnostic helps identify degree bias in negative sampling.

        Args:
            positive_edges: List of positive edge tuples
            negative_edges: List of negative edge tuples
            strategy_name: Name of the sampling strategy for display

        Returns:
            Dictionary with distribution statistics
        """
        def get_edge_degrees(edges):
            degrees = []
            for n1, n2 in edges:
                d1 = self._degrees.get(n1, 0)
                d2 = self._degrees.get(n2, 0)
                degrees.append(d1 + d2)  # Sum of endpoint degrees
            return np.array(degrees)

        pos_degrees = get_edge_degrees(positive_edges)
        neg_degrees = get_edge_degrees(negative_edges)

        stats = {
            "strategy": strategy_name,
            "pos_mean_degree": float(np.mean(pos_degrees)),
            "pos_median_degree": float(np.median(pos_degrees)),
            "pos_std_degree": float(np.std(pos_degrees)),
            "neg_mean_degree": float(np.mean(neg_degrees)),
            "neg_median_degree": float(np.median(neg_degrees)),
            "neg_std_degree": float(np.std(neg_degrees)),
            "degree_ratio": float(np.mean(pos_degrees) / np.mean(neg_degrees)) if np.mean(neg_degrees) > 0 else float('inf'),
        }

        print(f"\n=== Degree Distribution Comparison: {strategy_name} ===")
        print(f"Positive edges:")
        print(f"  Mean degree: {stats['pos_mean_degree']:.1f}")
        print(f"  Median degree: {stats['pos_median_degree']:.1f}")
        print(f"  Std degree: {stats['pos_std_degree']:.1f}")
        print(f"Negative edges:")
        print(f"  Mean degree: {stats['neg_mean_degree']:.1f}")
        print(f"  Median degree: {stats['neg_median_degree']:.1f}")
        print(f"  Std degree: {stats['neg_std_degree']:.1f}")
        print(f"Degree ratio (pos/neg): {stats['degree_ratio']:.2f}x")

        return stats


def evaluate_with_hard_negatives(
    predictor,
    test_edges: List[Tuple[str, str]],
    node2vec_model,
    sampler: HardNegativeSampler,
    strategies: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a link predictor with multiple negative sampling strategies.

    Args:
        predictor: Trained LinkPredictor with classifier
        test_edges: List of positive test edges
        node2vec_model: Trained Node2Vec model for embeddings
        sampler: HardNegativeSampler instance
        strategies: List of strategies to evaluate (default: all)

    Returns:
        Dictionary mapping strategy name to metrics
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    if strategies is None:
        strategies = ["random", "2hop", "degree_matched", "combined"]

    # Prepare positive test samples
    positive_samples = []
    positive_labels = []

    for n1, n2 in test_edges:
        try:
            emb1 = node2vec_model.wv[n1]
            emb2 = node2vec_model.wv[n2]
            features = emb1 * emb2  # Hadamard product
            positive_samples.append(features)
            positive_labels.append(1)
        except KeyError:
            continue

    results = {}
    n_negatives = len(positive_samples)

    # Evaluate each strategy
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Evaluating with {strategy} negatives")
        print(f"{'='*60}")

        # Sample negatives
        if strategy == "random":
            negatives = sampler.sample_random_negatives(n_negatives)
        elif strategy == "2hop":
            negatives = sampler.sample_2hop_negatives(n_negatives)
        elif strategy == "degree_matched":
            negatives = sampler.sample_degree_matched_negatives(test_edges, n_negatives)
        elif strategy == "combined":
            negatives = sampler.sample_combined_hard_negatives(test_edges, n_negatives)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Compare degree distributions
        sampler.compare_degree_distributions(test_edges, negatives, strategy)

        # Prepare negative samples
        negative_samples = []
        negative_labels = []

        for n1, n2 in negatives:
            try:
                emb1 = node2vec_model.wv[n1]
                emb2 = node2vec_model.wv[n2]
                features = emb1 * emb2
                negative_samples.append(features)
                negative_labels.append(0)
            except KeyError:
                continue

        # Combine samples
        X = np.array(positive_samples + negative_samples)
        y_true = np.array(positive_labels + negative_labels)

        # Predict
        y_pred = predictor.classifier.predict_proba(X)[:, 1]

        # Calculate metrics
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)

        results[strategy] = {
            "roc_auc": auc,
            "avg_precision": ap,
            "n_positives": len(positive_samples),
            "n_negatives": len(negative_samples),
        }

        print(f"\nResults for {strategy}:")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Avg Precision: {ap:.4f}")

    return results


def run_evaluation_experiment(
    data_dir: str,
    model_configs: List[Dict],
    output_dir: str = "results/"
) -> Dict[str, Dict]:
    """
    Run comprehensive evaluation experiment comparing all strategies.

    Args:
        data_dir: Path to STRING data directory
        model_configs: List of model config dicts with p, q, name
        output_dir: Directory to save results

    Returns:
        Dictionary with all results
    """
    from ml.link_predictor import LinkPredictor
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for config in model_configs:
        name = config["name"]
        p = config["p"]
        q = config["q"]

        print(f"\n{'='*80}")
        print(f"Training and evaluating: {name} (p={p}, q={q})")
        print(f"{'='*80}")

        # Initialize and load data
        predictor = LinkPredictor(
            embedding_dim=128,
            walk_length=80,
            num_walks=10,
            p=p,
            q=q,
            min_score=700,
            workers=4,
            seed=42,
        )
        predictor.load_string_data(data_dir)

        # Train (this splits edges, trains embeddings, trains classifier)
        metrics = predictor.train(test_size=0.2)

        # Initialize sampler with training graph
        sampler = HardNegativeSampler(predictor.train_graph, seed=42)

        # Get test edges that have embeddings
        test_edges_with_emb = []
        for edge in predictor.edge_set:
            n1, n2 = edge
            if n1 in predictor.node2vec_model.wv and n2 in predictor.node2vec_model.wv:
                if tuple(sorted([n1, n2])) not in set(
                    tuple(sorted([e[0], e[1]]))
                    for e in predictor.train_graph.edges()
                ):
                    test_edges_with_emb.append((n1, n2))

        # Evaluate with all strategies
        eval_results = evaluate_with_hard_negatives(
            predictor,
            test_edges_with_emb[:10000],  # Limit for speed
            predictor.node2vec_model,
            sampler,
            strategies=["random", "2hop", "degree_matched", "combined"]
        )

        all_results[name] = {
            "config": config,
            "train_metrics": metrics,
            "eval_results": eval_results,
        }

        # Save intermediate results
        results_path = os.path.join(output_dir, f"{name}_results.json")
        with open(results_path, "w") as f:
            # Convert numpy types for JSON serialization
            json_safe = {
                "config": config,
                "train_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                                  for k, v in metrics.items()},
                "eval_results": {
                    strat: {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in res.items()}
                    for strat, res in eval_results.items()
                }
            }
            json.dump(json_safe, f, indent=2)
        print(f"Saved results to {results_path}")

    return all_results


if __name__ == "__main__":
    # Quick test with a small synthetic graph
    import networkx as nx

    # Create test graph
    G = nx.barabasi_albert_graph(1000, 5, seed=42)
    G = nx.relabel_nodes(G, {i: f"node_{i}" for i in G.nodes()})

    print("Testing HardNegativeSampler...")
    sampler = HardNegativeSampler(G, seed=42)

    # Test each sampling strategy
    positive_edges = list(G.edges())[:100]

    print("\n1. Random sampling:")
    random_neg = sampler.sample_random_negatives(100)
    sampler.compare_degree_distributions(positive_edges, random_neg, "Random")

    print("\n2. 2-hop sampling:")
    twohop_neg = sampler.sample_2hop_negatives(100)
    sampler.compare_degree_distributions(positive_edges, twohop_neg, "2-hop")

    print("\n3. Degree-matched sampling:")
    degree_neg = sampler.sample_degree_matched_negatives(positive_edges, 100)
    sampler.compare_degree_distributions(positive_edges, degree_neg, "Degree-matched")

    print("\n4. Combined sampling:")
    combined_neg = sampler.sample_combined_hard_negatives(positive_edges, 100)
    sampler.compare_degree_distributions(positive_edges, combined_neg, "Combined")

    print("\nAll tests passed!")
