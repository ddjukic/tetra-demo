"""
Link Predictor using Node2Vec embeddings and Logistic Regression
for predicting protein-protein interactions from STRING database.

IMPORTANT: This implementation uses proper validation to avoid data leakage.
The edge split happens BEFORE training embeddings, so test edges are truly held out.

Proper link prediction workflow:
1. Load all edges from STRING
2. Split edges into train/test FIRST
3. Build training graph using only train edges
4. Train Node2Vec embeddings on the training graph only
5. Generate negative samples from nodes in training graph
6. Train classifier on train edges + train negatives
7. Evaluate on test edges + test negatives (truly unseen by embeddings)
"""

import gzip
import pickle
import os
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class LinkPredictor:
    """
    Predicts protein-protein interactions using Node2Vec embeddings
    and a Logistic Regression classifier.

    Uses proper validation methodology to avoid data leakage:
    - Edges are split BEFORE training embeddings
    - Test edges are never seen during Node2Vec training
    - This gives realistic performance estimates
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        min_score: int = 700,
        workers: int = 4,
        seed: int = 42,
    ):
        """
        Initialize the LinkPredictor.

        Args:
            embedding_dim: Dimension of node embeddings
            walk_length: Length of random walks
            num_walks: Number of walks per node
            p: Return parameter for Node2Vec
            q: In-out parameter for Node2Vec
            min_score: Minimum combined score for STRING edges
            workers: Number of parallel workers
            seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.min_score = min_score
        self.workers = workers
        self.seed = seed

        # Graph and mappings
        self.graph: Optional[nx.Graph] = None  # Full graph for inference
        self.train_graph: Optional[nx.Graph] = None  # Training graph (subset)
        self.string_id_to_gene: dict[str, str] = {}
        self.gene_to_string_id: dict[str, str] = {}
        self.alias_to_string_id: dict[str, str] = {}

        # Models
        self.node2vec_model: Optional[Word2Vec] = None
        self.classifier: Optional[LogisticRegression] = None

        # Edge set for checking existing interactions (full set for inference)
        self.edge_set: set[tuple[str, str]] = set()

        # All edges loaded from STRING (for proper splitting)
        self._all_edges: list[tuple[str, str, int]] = []  # (node1, node2, weight)

    def load_string_data(self, data_dir: str) -> None:
        """
        Load STRING physical interaction data.

        This loads the mappings and edges but does NOT build the training graph yet.
        The graph is built during train() after edges are properly split.

        Args:
            data_dir: Directory containing STRING data files
        """
        data_path = Path(data_dir)

        # Load protein info (STRING ID -> gene name mapping)
        print("Loading protein info...")
        info_file = data_path / "9606.protein.info.v12.0.txt.gz"
        with gzip.open(info_file, "rt") as f:
            # Skip header
            next(f)
            for line in tqdm(f, desc="Parsing protein info"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    string_id = parts[0]
                    gene_name = parts[1]
                    self.string_id_to_gene[string_id] = gene_name
                    # Use first occurrence for gene -> string_id mapping
                    if gene_name not in self.gene_to_string_id:
                        self.gene_to_string_id[gene_name] = string_id

        print(f"Loaded {len(self.string_id_to_gene)} protein mappings")

        # Load aliases for flexible gene name lookup
        print("Loading protein aliases...")
        aliases_file = data_path / "9606.protein.aliases.v12.0.txt.gz"
        with gzip.open(aliases_file, "rt") as f:
            # Skip header
            next(f)
            for line in tqdm(f, desc="Parsing aliases"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    string_id = parts[0]
                    alias = parts[1]
                    # Don't overwrite gene_to_string_id, use alias as fallback
                    if alias not in self.alias_to_string_id:
                        self.alias_to_string_id[alias] = string_id

        print(f"Loaded {len(self.alias_to_string_id)} aliases")

        # Load physical interactions - store as list for later splitting
        print(f"Loading physical interactions (score >= {self.min_score})...")
        self._all_edges = []
        links_file = data_path / "9606.protein.physical.links.detailed.v12.0.txt.gz"

        with gzip.open(links_file, "rt") as f:
            # Skip header
            next(f)
            for line in tqdm(f, desc="Parsing interactions"):
                parts = line.strip().split()
                if len(parts) >= 6:
                    protein1 = parts[0]
                    protein2 = parts[1]
                    combined_score = int(parts[5])

                    if combined_score >= self.min_score:
                        self._all_edges.append((protein1, protein2, combined_score))
                        # Store in edge_set for inference-time lookup
                        edge = tuple(sorted([protein1, protein2]))
                        self.edge_set.add(edge)

        print(f"Loaded {len(self._all_edges)} edges for training")

        # Build full graph for inference (contains all edges)
        self.graph = nx.Graph()
        for p1, p2, weight in self._all_edges:
            self.graph.add_edge(p1, p2, weight=weight)
        print(f"Full graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _resolve_to_string_id(self, name: str) -> Optional[str]:
        """
        Resolve a gene name or alias to STRING ID.

        Args:
            name: Gene name, alias, or STRING ID

        Returns:
            STRING ID if found, None otherwise
        """
        # Already a STRING ID
        if name.startswith("9606.ENSP"):
            return name

        # Check gene name mapping first
        if name in self.gene_to_string_id:
            return self.gene_to_string_id[name]

        # Check aliases
        if name in self.alias_to_string_id:
            return self.alias_to_string_id[name]

        # Try case-insensitive search
        name_upper = name.upper()
        for gene, string_id in self.gene_to_string_id.items():
            if gene.upper() == name_upper:
                return string_id

        return None

    def train_embeddings(self, graph: nx.Graph) -> Word2Vec:
        """
        Train Node2Vec embeddings on a given graph.

        Args:
            graph: The graph to train embeddings on

        Returns:
            Trained Word2Vec model
        """
        print("Training Node2Vec embeddings...")
        print(f"  dimensions={self.embedding_dim}, walk_length={self.walk_length}, "
              f"num_walks={self.num_walks}, p={self.p}, q={self.q}")

        node2vec = Node2Vec(
            graph,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=self.workers,
            seed=self.seed,
            quiet=False,
        )

        # Train Word2Vec model on walks
        print("Fitting Word2Vec model...")
        model = node2vec.fit(
            window=10,
            min_count=1,
            batch_words=4,
            workers=self.workers,
            seed=self.seed,
        )

        print(f"Trained embeddings for {len(model.wv)} nodes")
        return model

    def _get_edge_features(
        self, node1: str, node2: str, model: Optional[Word2Vec] = None
    ) -> Optional[np.ndarray]:
        """
        Get edge features using Hadamard product of node embeddings.

        Args:
            node1: First node STRING ID
            node2: Second node STRING ID
            model: Word2Vec model to use (defaults to self.node2vec_model)

        Returns:
            Feature vector or None if node not in vocabulary
        """
        model = model or self.node2vec_model
        if model is None:
            raise ValueError("No model provided and no trained model available.")

        try:
            emb1 = model.wv[node1]
            emb2 = model.wv[node2]
            return emb1 * emb2  # Hadamard product
        except KeyError:
            return None

    def train(self, test_size: float = 0.2) -> dict[str, float]:
        """
        Train the link predictor with PROPER validation (no data leakage).

        This method:
        1. Splits edges FIRST into train/test
        2. Builds training graph using only train edges
        3. Trains Node2Vec on training graph ONLY
        4. Samples negative edges from training graph nodes
        5. Trains classifier on train data
        6. Evaluates on TEST edges (truly held out from embeddings!)

        Args:
            test_size: Fraction of edges for testing (default 0.2)

        Returns:
            Dictionary with evaluation metrics
        """
        if not self._all_edges:
            raise ValueError("No edges loaded. Call load_string_data first.")

        print("\n" + "=" * 60)
        print("PROPER LINK PREDICTION TRAINING (No Data Leakage)")
        print("=" * 60)

        np.random.seed(self.seed)

        # Step 1: Split edges FIRST
        print(f"\nStep 1: Splitting {len(self._all_edges)} edges (test_size={test_size})")
        edges_array = np.array([(e[0], e[1]) for e in self._all_edges])
        indices = np.arange(len(edges_array))
        np.random.shuffle(indices)

        n_test = int(len(indices) * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        train_edges = [edges_array[i] for i in train_indices]
        test_edges = [edges_array[i] for i in test_indices]

        print(f"  Train edges: {len(train_edges)}")
        print(f"  Test edges: {len(test_edges)} (held out from embedding training!)")

        # Step 2: Build training graph (only train edges)
        print("\nStep 2: Building training graph from train edges only")
        self.train_graph = nx.Graph()
        train_edge_set = set()
        for p1, p2 in train_edges:
            self.train_graph.add_edge(p1, p2)
            train_edge_set.add(tuple(sorted([p1, p2])))

        print(f"  Training graph: {self.train_graph.number_of_nodes()} nodes, "
              f"{self.train_graph.number_of_edges()} edges")

        # Step 3: Train Node2Vec on training graph ONLY
        print("\nStep 3: Training Node2Vec on training graph (test edges NOT seen)")
        self.node2vec_model = self.train_embeddings(self.train_graph)

        # Step 4: Sample negative edges (from training graph nodes)
        print("\nStep 4: Sampling negative edges")
        train_nodes = list(self.train_graph.nodes())

        # Sample negatives for training (same count as train positives)
        train_negatives = []
        print(f"  Sampling {len(train_edges)} train negatives...")
        pbar = tqdm(total=len(train_edges), desc="Train negatives")
        seen_negatives = set()
        while len(train_negatives) < len(train_edges):
            i, j = np.random.randint(0, len(train_nodes), size=2)
            if i != j:
                n1, n2 = train_nodes[i], train_nodes[j]
                edge = tuple(sorted([n1, n2]))
                if edge not in self.edge_set and edge not in seen_negatives:
                    train_negatives.append((n1, n2))
                    seen_negatives.add(edge)
                    pbar.update(1)
        pbar.close()

        # Sample negatives for testing (same count as test positives)
        # Use only nodes that appear in training graph (so we have embeddings)
        test_negatives = []
        print(f"  Sampling {len(test_edges)} test negatives...")
        pbar = tqdm(total=len(test_edges), desc="Test negatives")
        while len(test_negatives) < len(test_edges):
            i, j = np.random.randint(0, len(train_nodes), size=2)
            if i != j:
                n1, n2 = train_nodes[i], train_nodes[j]
                edge = tuple(sorted([n1, n2]))
                if edge not in self.edge_set and edge not in seen_negatives:
                    test_negatives.append((n1, n2))
                    seen_negatives.add(edge)
                    pbar.update(1)
        pbar.close()

        # Step 5: Build feature matrices
        print("\nStep 5: Building feature matrices")

        # Training features (from train edges)
        X_train = []
        y_train = []

        for n1, n2 in tqdm(train_edges, desc="Train positive features"):
            features = self._get_edge_features(n1, n2)
            if features is not None:
                X_train.append(features)
                y_train.append(1)

        for n1, n2 in tqdm(train_negatives, desc="Train negative features"):
            features = self._get_edge_features(n1, n2)
            if features is not None:
                X_train.append(features)
                y_train.append(0)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Test features (from test edges - truly held out!)
        X_test = []
        y_test = []
        skipped_test = 0

        for n1, n2 in tqdm(test_edges, desc="Test positive features"):
            # Can only test edges where BOTH nodes are in training graph
            if n1 in self.node2vec_model.wv and n2 in self.node2vec_model.wv:
                features = self._get_edge_features(n1, n2)
                if features is not None:
                    X_test.append(features)
                    y_test.append(1)
            else:
                skipped_test += 1

        for n1, n2 in tqdm(test_negatives, desc="Test negative features"):
            features = self._get_edge_features(n1, n2)
            if features is not None:
                X_test.append(features)
                y_test.append(0)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        print(f"\n  Train set: {len(X_train)} samples ({sum(y_train)} pos, {len(y_train) - sum(y_train)} neg)")
        print(f"  Test set: {len(X_test)} samples ({sum(y_test)} pos, {len(y_test) - sum(y_test)} neg)")
        if skipped_test > 0:
            print(f"  (Skipped {skipped_test} test edges - nodes not in training graph)")

        # Step 6: Train classifier
        print("\nStep 6: Training Logistic Regression classifier")
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=self.seed,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.classifier.fit(X_train, y_train)

        # Step 7: Evaluate on truly held-out test set
        print("\nStep 7: Evaluating on TEST edges (never seen by embeddings!)")
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)

        metrics = {
            "auc": auc,
            "average_precision": ap,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "skipped_test_edges": skipped_test,
        }

        print("\n" + "=" * 60)
        print("EVALUATION METRICS (No Data Leakage)")
        print("=" * 60)
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        print("=" * 60)

        return metrics

    def train_classifier(self, test_size: float = 0.2) -> dict[str, float]:
        """
        Deprecated - use train() instead for proper validation.

        This method is kept for backwards compatibility but now calls train().
        """
        print("WARNING: train_classifier() is deprecated. Using train() instead.")
        return self.train(test_size=test_size)

    def predict(self, protein_pairs: list[tuple[str, str]]) -> list[dict]:
        """
        Predict interaction probabilities for protein pairs.

        Args:
            protein_pairs: List of (protein1, protein2) tuples
                           Can use gene names or STRING IDs

        Returns:
            List of prediction results
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_classifier first.")
        if self.node2vec_model is None:
            raise ValueError("Embeddings not trained. Call train_embeddings first.")

        results = []

        for p1_name, p2_name in protein_pairs:
            # Resolve names to STRING IDs
            p1_id = self._resolve_to_string_id(p1_name)
            p2_id = self._resolve_to_string_id(p2_name)

            result = {
                "protein1": p1_name,
                "protein2": p2_name,
                "ml_score": 0.0,
                "in_string": False,
                "error": None,
            }

            if p1_id is None:
                result["error"] = f"Unknown protein: {p1_name}"
                results.append(result)
                continue

            if p2_id is None:
                result["error"] = f"Unknown protein: {p2_name}"
                results.append(result)
                continue

            # Check if edge exists in STRING
            edge = tuple(sorted([p1_id, p2_id]))
            result["in_string"] = edge in self.edge_set

            # Get prediction
            features = self._get_edge_features(p1_id, p2_id)
            if features is None:
                result["error"] = "Protein not in embedding vocabulary"
                results.append(result)
                continue

            # Predict probability
            proba = self.classifier.predict_proba([features])[0, 1]
            result["ml_score"] = float(proba)

            results.append(result)

        return results

    def save(self, filepath: str) -> None:
        """
        Save the trained model to a file.

        Args:
            filepath: Path to save the model
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        state = {
            "embedding_dim": self.embedding_dim,
            "walk_length": self.walk_length,
            "num_walks": self.num_walks,
            "p": self.p,
            "q": self.q,
            "min_score": self.min_score,
            "string_id_to_gene": self.string_id_to_gene,
            "gene_to_string_id": self.gene_to_string_id,
            "alias_to_string_id": self.alias_to_string_id,
            "node2vec_model": self.node2vec_model,
            "classifier": self.classifier,
            "edge_set": self.edge_set,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "LinkPredictor":
        """
        Load a trained model from a file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded LinkPredictor instance
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        predictor = cls(
            embedding_dim=state["embedding_dim"],
            walk_length=state["walk_length"],
            num_walks=state["num_walks"],
            p=state["p"],
            q=state["q"],
            min_score=state["min_score"],
        )

        predictor.string_id_to_gene = state["string_id_to_gene"]
        predictor.gene_to_string_id = state["gene_to_string_id"]
        predictor.alias_to_string_id = state["alias_to_string_id"]
        predictor.node2vec_model = state["node2vec_model"]
        predictor.classifier = state["classifier"]
        predictor.edge_set = state["edge_set"]

        print(f"Model loaded from {filepath}")
        return predictor


if __name__ == "__main__":
    # Quick test
    predictor = LinkPredictor()
    predictor.load_string_data("data/string/")
