"""
Ensemble Link Predictor combining multiple Node2Vec models with different p/q parameters.

This module provides:
- EnsembleLinkPredictor: Combines predictions from multiple models
- estimate_graph_homophily: Analyzes graph structure to recommend optimal model

Different p/q configurations capture different graph properties:
- p=1, q=1 (balanced): Standard random walk
- p=1, q<1 (structural/DFS-like): Captures global structural roles (hubs, bridges)
- p=1, q>1 (homophily/BFS-like): Captures local neighborhood clustering

Reference: Grover & Leskovec (2016) "node2vec: Scalable Feature Learning for Networks"
"""

import statistics
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from ml.link_predictor import LinkPredictor


class EnsembleLinkPredictor:
    """
    Ensemble of Node2Vec link prediction models with different p/q parameters.

    Combines predictions from models trained with different random walk strategies
    to provide more robust and interpretable predictions.

    Model types:
    - balanced (p=1, q=1): Standard random walk, general-purpose
    - structural (p=1, q=0.5): DFS-like, captures global structural roles
    - homophily (p=1, q=2.0): BFS-like, captures local neighborhood clustering
    """

    DEFAULT_MODEL_CONFIGS = [
        {"name": "balanced", "path": "models/link_predictor.pkl", "p": 1.0, "q": 1.0},
        {"name": "structural", "path": "models/link_predictor_structural.pkl", "p": 1.0, "q": 0.5},
        {"name": "homophily", "path": "models/link_predictor_homophily.pkl", "p": 1.0, "q": 2.0},
    ]

    def __init__(
        self,
        model_paths: Optional[list[str]] = None,
        model_configs: Optional[list[dict]] = None,
    ):
        """
        Initialize ensemble with multiple link prediction models.

        Args:
            model_paths: List of paths to saved LinkPredictor models.
                        If None, uses default paths from DEFAULT_MODEL_CONFIGS.
            model_configs: Optional list of model configuration dicts with keys:
                          'name', 'path', 'p', 'q'. Overrides model_paths if provided.
        """
        self.models: list[LinkPredictor] = []
        self.model_configs: list[dict] = []

        # Determine which configs to use
        if model_configs is not None:
            configs = model_configs
        elif model_paths is not None:
            configs = [{"name": f"model_{i}", "path": p} for i, p in enumerate(model_paths)]
        else:
            configs = self.DEFAULT_MODEL_CONFIGS

        # Load models
        for config in configs:
            path = config["path"]
            if not Path(path).exists():
                print(f"Warning: Model not found at {path}, skipping")
                continue

            try:
                model = LinkPredictor.load(path)
                self.models.append(model)

                # Store config with actual p/q from loaded model
                self.model_configs.append({
                    "name": config.get("name", Path(path).stem),
                    "path": path,
                    "p": model.p,
                    "q": model.q,
                })
                print(f"Loaded model '{config.get('name', Path(path).stem)}' (p={model.p}, q={model.q})")
            except Exception as e:
                print(f"Error loading model from {path}: {e}")

        if not self.models:
            raise ValueError("No models could be loaded. Check model paths.")

        print(f"\nEnsemble initialized with {len(self.models)} models")

    def predict_single(self, protein1: str, protein2: str) -> dict:
        """
        Get ensemble predictions for a single protein pair.

        Args:
            protein1: First protein name or STRING ID
            protein2: Second protein name or STRING ID

        Returns:
            Dictionary with predictions from all models and ensemble statistics
        """
        predictions = []
        errors = []

        for model, config in zip(self.models, self.model_configs):
            result = model.predict([(protein1, protein2)])[0]

            if result.get("error"):
                errors.append({
                    "model": config["name"],
                    "error": result["error"],
                })
            else:
                predictions.append({
                    "model": config["name"],
                    "p": config["p"],
                    "q": config["q"],
                    "score": result["ml_score"],
                    "in_string": result["in_string"],
                })

        if not predictions:
            return {
                "protein1": protein1,
                "protein2": protein2,
                "predictions": [],
                "errors": errors,
                "mean_score": None,
                "std_score": None,
                "agreement": "no_predictions",
                "interpretation": "Could not make predictions: " + "; ".join(
                    [f"{e['model']}: {e['error']}" for e in errors]
                ),
            }

        scores = [p["score"] for p in predictions]
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0

        # Determine agreement level
        agreement = self._classify_agreement(scores)

        # Generate interpretation
        interpretation = self._interpret_predictions(predictions, mean_score, std_score, agreement)

        return {
            "protein1": protein1,
            "protein2": protein2,
            "predictions": predictions,
            "errors": errors if errors else None,
            "mean_score": mean_score,
            "std_score": std_score,
            "agreement": agreement,
            "interpretation": interpretation,
            "in_string": predictions[0]["in_string"] if predictions else None,
        }

    def predict_ensemble(self, protein1: str, protein2: str) -> dict:
        """
        Alias for predict_single for backwards compatibility.
        """
        return self.predict_single(protein1, protein2)

    def predict_batch(self, protein_pairs: list[tuple[str, str]]) -> list[dict]:
        """
        Get ensemble predictions for multiple protein pairs.

        Args:
            protein_pairs: List of (protein1, protein2) tuples

        Returns:
            List of prediction result dictionaries
        """
        return [self.predict_single(p1, p2) for p1, p2 in protein_pairs]

    def _classify_agreement(self, scores: list[float]) -> str:
        """
        Classify the level of agreement between model predictions.

        Args:
            scores: List of prediction scores from different models

        Returns:
            Agreement level: 'high', 'moderate', 'low', or 'conflicting'
        """
        if len(scores) < 2:
            return "single_model"

        std = statistics.stdev(scores)
        mean = statistics.mean(scores)

        # Check if predictions are on opposite sides of 0.5 threshold
        above_threshold = sum(1 for s in scores if s > 0.5)
        if above_threshold > 0 and above_threshold < len(scores):
            return "conflicting"

        # Classify based on coefficient of variation
        if mean > 0:
            cv = std / mean
            if cv < 0.1:
                return "high"
            elif cv < 0.25:
                return "moderate"
            else:
                return "low"
        else:
            if std < 0.05:
                return "high"
            elif std < 0.15:
                return "moderate"
            else:
                return "low"

    def _interpret_predictions(
        self,
        predictions: list[dict],
        mean_score: float,
        std_score: float,
        agreement: str,
    ) -> str:
        """
        Generate a human-readable interpretation of ensemble predictions.

        Args:
            predictions: List of individual model predictions
            mean_score: Mean prediction score
            std_score: Standard deviation of scores
            agreement: Agreement classification

        Returns:
            Interpretation string
        """
        if len(predictions) == 1:
            score = predictions[0]["score"]
            if score > 0.7:
                return f"Single model predicts likely interaction (score={score:.3f})."
            elif score < 0.3:
                return f"Single model predicts unlikely interaction (score={score:.3f})."
            else:
                return f"Single model gives uncertain prediction (score={score:.3f})."

        # Find specific model predictions
        structural_pred = next((p for p in predictions if p["model"] == "structural"), None)
        homophily_pred = next((p for p in predictions if p["model"] == "homophily"), None)
        balanced_pred = next((p for p in predictions if p["model"] == "balanced"), None)

        interpretation_parts = []

        # Overall assessment
        if mean_score > 0.7:
            interpretation_parts.append(
                f"Strong prediction of interaction (mean={mean_score:.3f}, std={std_score:.3f})."
            )
        elif mean_score > 0.5:
            interpretation_parts.append(
                f"Moderate prediction of interaction (mean={mean_score:.3f}, std={std_score:.3f})."
            )
        elif mean_score > 0.3:
            interpretation_parts.append(
                f"Weak/uncertain prediction (mean={mean_score:.3f}, std={std_score:.3f})."
            )
        else:
            interpretation_parts.append(
                f"Low interaction likelihood (mean={mean_score:.3f}, std={std_score:.3f})."
            )

        # Agreement analysis
        if agreement == "high":
            interpretation_parts.append("All models strongly agree.")
        elif agreement == "conflicting":
            interpretation_parts.append(self.interpret_disagreement(predictions))
        elif agreement == "low":
            interpretation_parts.append("Models show notable disagreement - interpret with caution.")

        # Specific model insights
        if structural_pred and homophily_pred:
            struct_score = structural_pred["score"]
            homo_score = homophily_pred["score"]

            if abs(struct_score - homo_score) > 0.2:
                if struct_score > homo_score:
                    interpretation_parts.append(
                        "Structural model scores higher - suggests role-based rather than "
                        "neighborhood-based relationship."
                    )
                else:
                    interpretation_parts.append(
                        "Homophily model scores higher - suggests proteins share similar "
                        "local network neighborhoods."
                    )

        return " ".join(interpretation_parts)

    def interpret_disagreement(self, predictions: list[dict]) -> str:
        """
        Interpret what model disagreement reveals about the relationship.

        Different p/q parameters capture different aspects:
        - Structural (low q): Global roles like hubs, bridges, peripheral nodes
        - Homophily (high q): Local clustering and neighborhood similarity

        Disagreement patterns:
        - Structural YES, Homophily NO: Proteins have similar global roles but
          different local neighborhoods
        - Structural NO, Homophily YES: Proteins are in similar neighborhoods but
          have different structural roles

        Args:
            predictions: List of model predictions

        Returns:
            Interpretation of what the disagreement suggests
        """
        structural_pred = next((p for p in predictions if p["model"] == "structural"), None)
        homophily_pred = next((p for p in predictions if p["model"] == "homophily"), None)
        balanced_pred = next((p for p in predictions if p["model"] == "balanced"), None)

        if not (structural_pred and homophily_pred):
            return "Cannot interpret disagreement - need both structural and homophily models."

        struct_yes = structural_pred["score"] > 0.5
        homo_yes = homophily_pred["score"] > 0.5

        if struct_yes and not homo_yes:
            return (
                "Structural model predicts interaction but homophily model does not. "
                "This suggests the proteins may share similar global network roles "
                "(e.g., both are hubs or bridges) but operate in different local "
                "neighborhoods. The relationship may be functional/role-based rather "
                "than proximity-based."
            )
        elif homo_yes and not struct_yes:
            return (
                "Homophily model predicts interaction but structural model does not. "
                "This suggests the proteins share similar local neighborhoods and "
                "may interact with similar partners, but have different global roles "
                "in the network (e.g., one is a hub, one is peripheral). The relationship "
                "may be based on shared pathway membership."
            )
        elif not struct_yes and not homo_yes:
            return (
                "Neither model predicts strong interaction. If scores differ significantly, "
                "the higher-scoring model suggests which type of relationship is more plausible."
            )
        else:  # both yes
            return "Both models agree on likely interaction."

    def get_model_summary(self) -> dict:
        """
        Get summary information about loaded models.

        Returns:
            Dictionary with model count and configurations
        """
        return {
            "num_models": len(self.models),
            "models": self.model_configs,
        }


def estimate_graph_homophily(
    graph: nx.Graph,
    attribute: str = "type",
    default_type: str = "unknown",
) -> dict:
    """
    Calculate edge homophily ratio for a graph.

    Homophily measures the tendency of nodes to connect with similar nodes.
    This is useful for understanding which Node2Vec configuration might work best:
    - High homophily (H > 0.5): Similar nodes connect -> homophily model (high q) recommended
    - Low homophily (H < 0.5): Different nodes connect -> structural model (low q) recommended

    Formula: H = (edges connecting same-type nodes) / (total edges)

    Args:
        graph: NetworkX graph with node attributes
        attribute: Node attribute name to use for type comparison (default: 'type')
        default_type: Default type for nodes without the attribute

    Returns:
        Dictionary with:
        - homophily_ratio: Float between 0 and 1
        - same_type_edges: Count of edges between same-type nodes
        - diff_type_edges: Count of edges between different-type nodes
        - total_edges: Total edge count
        - type_distribution: Distribution of node types
        - interpretation: Human-readable interpretation
        - recommended_model: Suggested model based on homophily
    """
    if graph.number_of_edges() == 0:
        return {
            "homophily_ratio": None,
            "same_type_edges": 0,
            "diff_type_edges": 0,
            "total_edges": 0,
            "type_distribution": {},
            "interpretation": "Graph has no edges.",
            "recommended_model": "balanced",
        }

    same_type_edges = 0
    diff_type_edges = 0
    type_counts: dict[str, int] = {}

    # Count node types
    for node in graph.nodes():
        node_type = graph.nodes[node].get(attribute, default_type)
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    # Count edge types
    for u, v in graph.edges():
        u_type = graph.nodes[u].get(attribute, default_type)
        v_type = graph.nodes[v].get(attribute, default_type)

        if u_type == v_type:
            same_type_edges += 1
        else:
            diff_type_edges += 1

    total_edges = same_type_edges + diff_type_edges
    homophily_ratio = same_type_edges / total_edges if total_edges > 0 else 0.0

    # Compute expected homophily under random mixing
    # E[H] = sum(p_i^2) where p_i is fraction of type i nodes
    total_nodes = graph.number_of_nodes()
    expected_homophily = sum((c / total_nodes) ** 2 for c in type_counts.values()) if total_nodes > 0 else 0

    # Generate interpretation
    if homophily_ratio > 0.7:
        interpretation = (
            f"Strong homophily (H={homophily_ratio:.3f}). Nodes strongly prefer connecting "
            f"to similar nodes. Expected under random mixing: {expected_homophily:.3f}. "
            "Homophily model (high q) recommended for capturing local community structure."
        )
        recommended = "homophily"
    elif homophily_ratio > 0.5:
        interpretation = (
            f"Moderate homophily (H={homophily_ratio:.3f}). Nodes somewhat prefer connecting "
            f"to similar nodes. Expected under random mixing: {expected_homophily:.3f}. "
            "Balanced model may work well, but homophily model could capture community effects."
        )
        recommended = "balanced"
    elif homophily_ratio > 0.3:
        interpretation = (
            f"Low homophily / slight heterophily (H={homophily_ratio:.3f}). Nodes show slight "
            f"preference for different-type connections. Expected: {expected_homophily:.3f}. "
            "Balanced or structural model recommended."
        )
        recommended = "balanced"
    else:
        interpretation = (
            f"Strong heterophily (H={homophily_ratio:.3f}). Nodes strongly prefer connecting "
            f"to different-type nodes. Expected under random mixing: {expected_homophily:.3f}. "
            "Structural model (low q) recommended to capture role-based patterns."
        )
        recommended = "structural"

    return {
        "homophily_ratio": homophily_ratio,
        "expected_random_homophily": expected_homophily,
        "same_type_edges": same_type_edges,
        "diff_type_edges": diff_type_edges,
        "total_edges": total_edges,
        "type_distribution": type_counts,
        "interpretation": interpretation,
        "recommended_model": recommended,
    }


def analyze_node_properties(graph: nx.Graph) -> dict:
    """
    Analyze structural properties of nodes to understand graph characteristics.

    This complements homophily analysis by examining node-level metrics
    that inform model selection.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary with degree distribution, clustering coefficient, and recommendations
    """
    if graph.number_of_nodes() == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "avg_degree": 0,
            "avg_clustering": 0,
            "density": 0,
            "interpretation": "Empty graph.",
            "recommended_model": "balanced",
        }

    degrees = [d for _, d in graph.degree()]
    avg_degree = np.mean(degrees)
    std_degree = np.std(degrees)
    max_degree = max(degrees)

    # Clustering coefficient (sample for large graphs)
    if graph.number_of_nodes() > 10000:
        sample_nodes = np.random.choice(list(graph.nodes()), min(5000, graph.number_of_nodes()), replace=False)
        clustering = nx.clustering(graph, sample_nodes)
        avg_clustering = np.mean(list(clustering.values()))
    else:
        avg_clustering = nx.average_clustering(graph)

    density = nx.density(graph)

    # Interpret and recommend
    interpretation_parts = []

    if avg_clustering > 0.3:
        interpretation_parts.append(
            f"High clustering (avg={avg_clustering:.3f}) suggests strong local community structure."
        )
        recommended = "homophily"
    elif avg_clustering < 0.1:
        interpretation_parts.append(
            f"Low clustering (avg={avg_clustering:.3f}) suggests sparse, role-based structure."
        )
        recommended = "structural"
    else:
        interpretation_parts.append(
            f"Moderate clustering (avg={avg_clustering:.3f})."
        )
        recommended = "balanced"

    # Degree heterogeneity analysis
    cv_degree = std_degree / avg_degree if avg_degree > 0 else 0
    if cv_degree > 2:
        interpretation_parts.append(
            f"High degree heterogeneity (CV={cv_degree:.2f}, max={max_degree}) - "
            "hub-and-spoke structure may benefit from structural model."
        )
        if recommended == "balanced":
            recommended = "structural"

    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "avg_degree": avg_degree,
        "std_degree": std_degree,
        "max_degree": max_degree,
        "avg_clustering": avg_clustering,
        "density": density,
        "interpretation": " ".join(interpretation_parts),
        "recommended_model": recommended,
    }
