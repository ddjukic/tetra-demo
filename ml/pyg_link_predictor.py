"""
PyTorch Geometric Link Predictor using Node2Vec embeddings with MPS support.

This implementation uses PyG's native Node2Vec for GPU-accelerated training
on Apple Silicon (MPS), CUDA, or CPU devices.

Key improvements over gensim implementation:
- GPU acceleration via MPS (Apple Silicon) or CUDA
- Native PyTorch integration for end-to-end training
- Faster embedding computation with batch processing
- Better memory efficiency with sparse operations

Usage:
    predictor = PyGLinkPredictor(embedding_dim=128, device='mps')
    predictor.load_string_data('data/string/')
    metrics = predictor.train_with_validation(epochs=100)
    results = predictor.predict([('BRCA1', 'TP53')])
"""

import gzip
import os
import pickle
import time
from pathlib import Path
from typing import Optional, Literal

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn.models import Node2Vec
from tqdm import tqdm


DeviceType = Literal["auto", "mps", "cuda", "cpu"]


def get_device(device: DeviceType = "auto") -> torch.device:
    """
    Get the best available device for training.

    Args:
        device: Device selection - 'auto' will select best available,
                or specify 'mps', 'cuda', or 'cpu' explicitly.

    Returns:
        torch.device instance
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


class PyGLinkPredictor:
    """
    Link predictor using PyTorch Geometric's Node2Vec with GPU support.

    Uses proper validation methodology to avoid data leakage:
    - Edges are split BEFORE training embeddings
    - Test edges are never seen during Node2Vec training
    - This gives realistic performance estimates
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        walk_length: int = 80,
        walks_per_node: int = 10,
        context_size: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        min_score: int = 700,
        device: DeviceType = "auto",
        seed: int = 42,
    ):
        """
        Initialize the PyG Link Predictor.

        Args:
            embedding_dim: Dimension of node embeddings
            walk_length: Length of random walks
            walks_per_node: Number of walks per node (same as num_walks in gensim)
            context_size: Window size for skip-gram (similar to window in Word2Vec)
            p: Return parameter for Node2Vec (probability of returning to previous node)
            q: In-out parameter for Node2Vec (probability of exploring outward)
            num_negative_samples: Number of negative samples per positive
            min_score: Minimum combined score for STRING edges
            device: Device to use ('auto', 'mps', 'cuda', 'cpu')
            seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.context_size = context_size
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.min_score = min_score
        self.device = get_device(device)
        self.seed = seed

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Enable MPS fallback for unsupported ops
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        # Graph and mappings
        self.graph: Optional[nx.Graph] = None
        self.train_graph: Optional[nx.Graph] = None
        self.string_id_to_gene: dict[str, str] = {}
        self.gene_to_string_id: dict[str, str] = {}
        self.alias_to_string_id: dict[str, str] = {}

        # Node indexing for PyG
        self.node_to_idx: dict[str, int] = {}
        self.idx_to_node: dict[int, str] = {}

        # Models
        self.node2vec_model: Optional[Node2Vec] = None
        self.embeddings: Optional[torch.Tensor] = None
        self.classifier: Optional[LogisticRegression] = None

        # Edge set for checking existing interactions
        self.edge_set: set[tuple[str, str]] = set()

        # All edges loaded from STRING
        self._all_edges: list[tuple[str, str, int]] = []

        # Training metrics storage
        self.training_history: list[dict] = []

        print(f"PyGLinkPredictor initialized on device: {self.device}")

    def load_string_data(self, data_dir: str) -> None:
        """
        Load STRING physical interaction data.

        Args:
            data_dir: Directory containing STRING data files
        """
        data_path = Path(data_dir)

        # Load protein info (STRING ID -> gene name mapping)
        print("Loading protein info...")
        info_file = data_path / "9606.protein.info.v12.0.txt.gz"
        with gzip.open(info_file, "rt") as f:
            next(f)  # Skip header
            for line in tqdm(f, desc="Parsing protein info"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    string_id = parts[0]
                    gene_name = parts[1]
                    self.string_id_to_gene[string_id] = gene_name
                    if gene_name not in self.gene_to_string_id:
                        self.gene_to_string_id[gene_name] = string_id

        print(f"Loaded {len(self.string_id_to_gene)} protein mappings")

        # Load aliases
        print("Loading protein aliases...")
        aliases_file = data_path / "9606.protein.aliases.v12.0.txt.gz"
        with gzip.open(aliases_file, "rt") as f:
            next(f)
            for line in tqdm(f, desc="Parsing aliases"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    string_id = parts[0]
                    alias = parts[1]
                    if alias not in self.alias_to_string_id:
                        self.alias_to_string_id[alias] = string_id

        print(f"Loaded {len(self.alias_to_string_id)} aliases")

        # Load physical interactions
        print(f"Loading physical interactions (score >= {self.min_score})...")
        self._all_edges = []
        links_file = data_path / "9606.protein.physical.links.detailed.v12.0.txt.gz"

        with gzip.open(links_file, "rt") as f:
            next(f)
            for line in tqdm(f, desc="Parsing interactions"):
                parts = line.strip().split()
                if len(parts) >= 6:
                    protein1 = parts[0]
                    protein2 = parts[1]
                    combined_score = int(parts[5])

                    if combined_score >= self.min_score:
                        self._all_edges.append((protein1, protein2, combined_score))
                        edge = tuple(sorted([protein1, protein2]))
                        self.edge_set.add(edge)

        print(f"Loaded {len(self._all_edges)} edges for training")

        # Build full graph
        self.graph = nx.Graph()
        for p1, p2, weight in self._all_edges:
            self.graph.add_edge(p1, p2, weight=weight)
        print(f"Full graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _build_node_index(self, graph: nx.Graph) -> None:
        """Build node to index mappings for PyG."""
        self.node_to_idx = {node: idx for idx, node in enumerate(sorted(graph.nodes()))}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

    def _graph_to_edge_index(self, graph: nx.Graph) -> torch.Tensor:
        """
        Convert NetworkX graph to PyG edge_index tensor.

        Args:
            graph: NetworkX graph

        Returns:
            edge_index tensor of shape [2, num_edges * 2] (undirected)
        """
        edges = []
        for u, v in graph.edges():
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            # Add both directions for undirected graph
            edges.append([u_idx, v_idx])
            edges.append([v_idx, u_idx])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def _resolve_to_string_id(self, name: str) -> Optional[str]:
        """Resolve a gene name or alias to STRING ID."""
        if name.startswith("9606.ENSP"):
            return name

        if name in self.gene_to_string_id:
            return self.gene_to_string_id[name]

        if name in self.alias_to_string_id:
            return self.alias_to_string_id[name]

        name_upper = name.upper()
        for gene, string_id in self.gene_to_string_id.items():
            if gene.upper() == name_upper:
                return string_id

        return None

    def _get_edge_features(
        self, node1: str, node2: str, embeddings: Optional[torch.Tensor] = None
    ) -> Optional[np.ndarray]:
        """
        Get edge features using Hadamard product of node embeddings.

        Args:
            node1: First node STRING ID
            node2: Second node STRING ID
            embeddings: Optional embeddings tensor to use

        Returns:
            Feature vector or None if node not found
        """
        embeddings = embeddings if embeddings is not None else self.embeddings
        if embeddings is None:
            raise ValueError("No embeddings available")

        if node1 not in self.node_to_idx or node2 not in self.node_to_idx:
            return None

        idx1 = self.node_to_idx[node1]
        idx2 = self.node_to_idx[node2]

        emb1 = embeddings[idx1].detach().cpu().numpy()
        emb2 = embeddings[idx2].detach().cpu().numpy()

        return emb1 * emb2  # Hadamard product

    def train_embeddings(
        self,
        graph: nx.Graph,
        epochs: int = 100,
        batch_size: int = 128,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float]]:
        """
        Train Node2Vec embeddings using PyTorch Geometric.

        Args:
            graph: NetworkX graph to train on
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            verbose: Whether to print progress

        Returns:
            Tuple of (embeddings tensor, loss history)
        """
        # Build node index
        self._build_node_index(graph)
        num_nodes = len(self.node_to_idx)

        # Convert to edge_index
        edge_index = self._graph_to_edge_index(graph)

        if verbose:
            print(f"\nTraining Node2Vec embeddings (PyG) on {self.device}")
            print(f"  Nodes: {num_nodes}")
            print(f"  Edges: {edge_index.shape[1] // 2}")
            print(f"  embedding_dim={self.embedding_dim}, walk_length={self.walk_length}")
            print(f"  walks_per_node={self.walks_per_node}, context_size={self.context_size}")
            print(f"  p={self.p}, q={self.q}")
            print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}")

        # Initialize Node2Vec model
        # Use sparse gradients only on CPU/CUDA (MPS doesn't support sparse operations well)
        use_sparse = str(self.device) not in ("mps", "mps:0")

        self.node2vec_model = Node2Vec(
            edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            p=self.p,
            q=self.q,
            num_negative_samples=self.num_negative_samples,
            num_nodes=num_nodes,
            sparse=use_sparse,  # Use sparse gradients on CPU/CUDA only
        ).to(self.device)

        # Get data loader
        loader = self.node2vec_model.loader(batch_size=batch_size, shuffle=True, num_workers=0)

        # Optimizer - use SparseAdam for sparse, Adam for dense
        if use_sparse:
            optimizer = torch.optim.SparseAdam(self.node2vec_model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(self.node2vec_model.parameters(), lr=lr)

        # Training loop
        loss_history = []
        epoch_times = []

        if verbose:
            print("\nStarting training...")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            self.node2vec_model.train()
            total_loss = 0
            num_batches = 0

            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = self.node2vec_model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            epoch_time = time.time() - epoch_start
            loss_history.append(avg_loss)
            epoch_times.append(epoch_time)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch:3d}/{epochs}: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s")

        # Get final embeddings
        self.node2vec_model.eval()
        with torch.no_grad():
            self.embeddings = self.node2vec_model()

        if verbose:
            avg_epoch_time = np.mean(epoch_times)
            total_time = sum(epoch_times)
            print(f"\nTraining complete!")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Average epoch time: {avg_epoch_time:.2f}s")
            print(f"  Final loss: {loss_history[-1]:.4f}")

        return self.embeddings, loss_history

    def train_with_validation(
        self,
        test_size: float = 0.2,
        epochs: int = 100,
        batch_size: int = 128,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> dict:
        """
        Train the link predictor with proper validation (no data leakage).

        This method:
        1. Splits edges FIRST into train/test
        2. Builds training graph using only train edges
        3. Trains Node2Vec on training graph ONLY
        4. Samples negative edges from training graph nodes
        5. Trains classifier on train data
        6. Evaluates on TEST edges (truly held out from embeddings!)

        Args:
            test_size: Fraction of edges for testing
            epochs: Training epochs for Node2Vec
            batch_size: Batch size for training
            lr: Learning rate
            verbose: Whether to print progress

        Returns:
            Dictionary with evaluation metrics
        """
        if not self._all_edges:
            raise ValueError("No edges loaded. Call load_string_data first.")

        start_time = time.time()

        if verbose:
            print("\n" + "=" * 60)
            print("PROPER LINK PREDICTION TRAINING (No Data Leakage)")
            print("=" * 60)

        np.random.seed(self.seed)

        # Step 1: Split edges
        if verbose:
            print(f"\nStep 1: Splitting {len(self._all_edges)} edges (test_size={test_size})")

        edges_array = np.array([(e[0], e[1]) for e in self._all_edges])
        indices = np.arange(len(edges_array))
        np.random.shuffle(indices)

        n_test = int(len(indices) * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        train_edges = [edges_array[i] for i in train_indices]
        test_edges = [edges_array[i] for i in test_indices]

        if verbose:
            print(f"  Train edges: {len(train_edges)}")
            print(f"  Test edges: {len(test_edges)} (held out from embedding training!)")

        # Step 2: Build training graph
        if verbose:
            print("\nStep 2: Building training graph from train edges only")

        self.train_graph = nx.Graph()
        train_edge_set = set()
        for p1, p2 in train_edges:
            self.train_graph.add_edge(p1, p2)
            train_edge_set.add(tuple(sorted([p1, p2])))

        if verbose:
            print(f"  Training graph: {self.train_graph.number_of_nodes()} nodes, "
                  f"{self.train_graph.number_of_edges()} edges")

        # Step 3: Train Node2Vec
        if verbose:
            print("\nStep 3: Training Node2Vec on training graph (test edges NOT seen)")

        embed_start = time.time()
        embeddings, loss_history = self.train_embeddings(
            self.train_graph,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
        )
        embed_time = time.time() - embed_start

        # Step 4: Sample negatives
        if verbose:
            print("\nStep 4: Sampling negative edges")

        train_nodes = list(self.train_graph.nodes())

        # Sample train negatives
        train_negatives = []
        seen_negatives = set()

        pbar = tqdm(total=len(train_edges), desc="Train negatives", disable=not verbose)
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

        # Sample test negatives
        test_negatives = []
        pbar = tqdm(total=len(test_edges), desc="Test negatives", disable=not verbose)
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
        if verbose:
            print("\nStep 5: Building feature matrices")

        X_train, y_train = [], []
        for n1, n2 in tqdm(train_edges, desc="Train positive features", disable=not verbose):
            features = self._get_edge_features(n1, n2)
            if features is not None:
                X_train.append(features)
                y_train.append(1)

        for n1, n2 in tqdm(train_negatives, desc="Train negative features", disable=not verbose):
            features = self._get_edge_features(n1, n2)
            if features is not None:
                X_train.append(features)
                y_train.append(0)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Test features
        X_test, y_test = [], []
        skipped_test = 0

        for n1, n2 in tqdm(test_edges, desc="Test positive features", disable=not verbose):
            if n1 in self.node_to_idx and n2 in self.node_to_idx:
                features = self._get_edge_features(n1, n2)
                if features is not None:
                    X_test.append(features)
                    y_test.append(1)
            else:
                skipped_test += 1

        for n1, n2 in tqdm(test_negatives, desc="Test negative features", disable=not verbose):
            features = self._get_edge_features(n1, n2)
            if features is not None:
                X_test.append(features)
                y_test.append(0)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        if verbose:
            print(f"\n  Train set: {len(X_train)} samples ({sum(y_train)} pos, {len(y_train) - sum(y_train)} neg)")
            print(f"  Test set: {len(X_test)} samples ({sum(y_test)} pos, {len(y_test) - sum(y_test)} neg)")
            if skipped_test > 0:
                print(f"  (Skipped {skipped_test} test edges - nodes not in training graph)")

        # Step 6: Train classifier
        if verbose:
            print("\nStep 6: Training Logistic Regression classifier")

        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=self.seed,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.classifier.fit(X_train, y_train)

        # Step 7: Evaluate
        if verbose:
            print("\nStep 7: Evaluating on TEST edges (never seen by embeddings!)")

        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)

        total_time = time.time() - start_time

        metrics = {
            "auc": float(auc),
            "average_precision": float(ap),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "skipped_test_edges": skipped_test,
            "embedding_time_seconds": embed_time,
            "total_time_seconds": total_time,
            "final_loss": loss_history[-1] if loss_history else 0,
            "loss_history": loss_history,
            "device": str(self.device),
            # Store test data for ROC curve generation
            "_y_test": y_test.tolist(),
            "_y_pred_proba": y_pred_proba.tolist(),
        }

        if verbose:
            print("\n" + "=" * 60)
            print("EVALUATION METRICS (No Data Leakage)")
            print("=" * 60)
            print(f"  ROC-AUC: {auc:.4f}")
            print(f"  Average Precision: {ap:.4f}")
            print(f"  Training time: {embed_time:.1f}s")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Device: {self.device}")
            print("=" * 60)

        return metrics

    def predict(self, protein_pairs: list[tuple[str, str]]) -> list[dict]:
        """
        Predict interaction probabilities for protein pairs.

        Args:
            protein_pairs: List of (protein1, protein2) tuples

        Returns:
            List of prediction results
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_with_validation first.")
        if self.embeddings is None:
            raise ValueError("Embeddings not trained.")

        results = []

        for p1_name, p2_name in protein_pairs:
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

            edge = tuple(sorted([p1_id, p2_id]))
            result["in_string"] = edge in self.edge_set

            features = self._get_edge_features(p1_id, p2_id)
            if features is None:
                result["error"] = "Protein not in embedding vocabulary"
                results.append(result)
                continue

            proba = self.classifier.predict_proba([features])[0, 1]
            result["ml_score"] = float(proba)
            results.append(result)

        return results

    def save(self, filepath: str) -> None:
        """Save the trained model to a file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # Move embeddings to CPU for saving
        embeddings_cpu = self.embeddings.cpu() if self.embeddings is not None else None

        state = {
            "embedding_dim": self.embedding_dim,
            "walk_length": self.walk_length,
            "walks_per_node": self.walks_per_node,
            "context_size": self.context_size,
            "p": self.p,
            "q": self.q,
            "num_negative_samples": self.num_negative_samples,
            "min_score": self.min_score,
            "string_id_to_gene": self.string_id_to_gene,
            "gene_to_string_id": self.gene_to_string_id,
            "alias_to_string_id": self.alias_to_string_id,
            "node_to_idx": self.node_to_idx,
            "idx_to_node": self.idx_to_node,
            "embeddings": embeddings_cpu,
            "classifier": self.classifier,
            "edge_set": self.edge_set,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: DeviceType = "auto") -> "PyGLinkPredictor":
        """Load a trained model from a file."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        predictor = cls(
            embedding_dim=state["embedding_dim"],
            walk_length=state["walk_length"],
            walks_per_node=state["walks_per_node"],
            context_size=state["context_size"],
            p=state["p"],
            q=state["q"],
            num_negative_samples=state["num_negative_samples"],
            min_score=state["min_score"],
            device=device,
        )

        predictor.string_id_to_gene = state["string_id_to_gene"]
        predictor.gene_to_string_id = state["gene_to_string_id"]
        predictor.alias_to_string_id = state["alias_to_string_id"]
        predictor.node_to_idx = state["node_to_idx"]
        predictor.idx_to_node = state["idx_to_node"]
        predictor.edge_set = state["edge_set"]
        predictor.classifier = state["classifier"]

        # Move embeddings to device
        if state["embeddings"] is not None:
            predictor.embeddings = state["embeddings"].to(predictor.device)

        print(f"Model loaded from {filepath}")
        return predictor


if __name__ == "__main__":
    # Quick test
    predictor = PyGLinkPredictor()
    print(f"Initialized predictor on device: {predictor.device}")
