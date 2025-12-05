"""
GraphRAG Q&A Agent with ReAct Planning.

This module provides a conversational agent for querying and analyzing
biomedical knowledge graphs using Google ADK with ReAct (Reason + Act) planning.

Features:
- Natural language queries about the knowledge graph
- Graph Data Science algorithms (centrality, communities, paths)
- Drug discovery algorithms (DIAMOnD, network proximity, synergy, robustness)
- ML link predictions with hypothesis generation (PyG + ensemble support)
- Multi-step ReAct planning for complex queries

Architecture:
- Module-level graph cache for fast tool access
- Module-level link predictor for PyG ML predictions
- ToolContext for session state management
- @requires_graph decorator ensures graph is loaded before tool execution
- PlanReActPlanner for structured reasoning: Thought -> Action -> Observation -> Repeat
"""

import asyncio
import logging
from functools import wraps
from pathlib import Path
from typing import Any, Optional, Protocol

# =============================================================================
# Langfuse Observability Setup (MUST be before ADK imports)
# =============================================================================
# OpenTelemetry instrumentation must be configured before importing google.adk
# for the GoogleADKInstrumentor to properly intercept ADK agent operations.

from observability.tracing import setup_langfuse_tracing

_LANGFUSE_ENABLED = setup_langfuse_tracing()

# =============================================================================
# ADK Imports (after Langfuse setup)
# =============================================================================

from google.adk.agents import LlmAgent
from google.adk.planners import PlanReActPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from models.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

if _LANGFUSE_ENABLED:
    logger.info("GraphRAG Q&A Agent: Langfuse tracing enabled")
else:
    logger.debug("GraphRAG Q&A Agent: Langfuse tracing disabled (set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY to enable)")


# =============================================================================
# Link Predictor Protocol (for type hints without hard dependency)
# =============================================================================


class LinkPredictorProtocol(Protocol):
    """Protocol for link predictors (PyGLinkPredictor or EnsemblePredictor)."""

    def predict(self, protein_pairs: list[tuple[str, str]]) -> list[dict]:
        """Predict interaction probabilities for protein pairs."""
        ...


# =============================================================================
# Module-Level Link Predictor Cache
# =============================================================================

_link_predictor: Optional[LinkPredictorProtocol] = None
_ensemble_predictor: Optional[Any] = None  # For PyGEnsembleLinkPredictor


def set_link_predictor(predictor: LinkPredictorProtocol) -> None:
    """
    Set the active link predictor for ML predictions.

    Args:
        predictor: PyGLinkPredictor or compatible predictor instance
    """
    global _link_predictor
    _link_predictor = predictor
    logger.info("Link predictor set for ML predictions")


def get_link_predictor() -> Optional[LinkPredictorProtocol]:
    """Get the active link predictor."""
    return _link_predictor


def set_ensemble_predictor(predictor: Any) -> None:
    """
    Set the ensemble predictor for multi-model predictions.

    Args:
        predictor: PyGEnsembleLinkPredictor instance
    """
    global _ensemble_predictor
    _ensemble_predictor = predictor
    logger.info("Ensemble predictor set with multiple PyG models")


def get_ensemble_predictor() -> Optional[Any]:
    """Get the active ensemble predictor."""
    return _ensemble_predictor


def load_pyg_predictor(
    model_path: str = "models/pyg_link_predictor.pkl",
    device: str = "auto"
) -> Optional[LinkPredictorProtocol]:
    """
    Load PyGLinkPredictor from saved model file.

    Args:
        model_path: Path to saved PyGLinkPredictor model
        device: Device to use ('auto', 'mps', 'cuda', 'cpu')

    Returns:
        Loaded PyGLinkPredictor or None if loading fails
    """
    global _link_predictor

    if not Path(model_path).exists():
        logger.warning(f"PyG model not found at {model_path}")
        return None

    try:
        from ml.pyg_link_predictor import PyGLinkPredictor
        predictor = PyGLinkPredictor.load(model_path, device=device)
        _link_predictor = predictor
        logger.info(f"Loaded PyGLinkPredictor from {model_path} on device {predictor.device}")
        return predictor
    except Exception as e:
        logger.error(f"Failed to load PyGLinkPredictor: {e}")
        return None


def load_pyg_ensemble(
    model_dir: str = "models",
    device: str = "auto"
) -> Optional[Any]:
    """
    Load PyG ensemble predictor with multiple models (balanced, structural, homophily).

    Args:
        model_dir: Directory containing PyG model files
        device: Device to use ('auto', 'mps', 'cuda', 'cpu')

    Returns:
        PyGEnsembleLinkPredictor or None if loading fails
    """
    global _ensemble_predictor, _link_predictor

    model_configs = [
        {"name": "balanced", "path": f"{model_dir}/pyg_link_predictor.pkl"},
        {"name": "structural", "path": f"{model_dir}/pyg_link_predictor_structural.pkl"},
        {"name": "homophily", "path": f"{model_dir}/pyg_link_predictor_homophily.pkl"},
    ]

    # Filter to existing models
    existing_configs = [c for c in model_configs if Path(c["path"]).exists()]

    if not existing_configs:
        logger.warning(f"No PyG models found in {model_dir}")
        return None

    try:
        from ml.pyg_link_predictor import PyGLinkPredictor

        # Create ensemble-like wrapper
        class PyGEnsembleLinkPredictor:
            """Ensemble of PyG link predictors for robust predictions."""

            def __init__(self, configs: list[dict], device: str):
                self.models: list[tuple[str, PyGLinkPredictor]] = []
                for config in configs:
                    try:
                        model = PyGLinkPredictor.load(config["path"], device=device)
                        self.models.append((config["name"], model))
                        logger.info(f"Loaded ensemble model '{config['name']}' from {config['path']}")
                    except Exception as e:
                        logger.warning(f"Failed to load {config['path']}: {e}")

                if not self.models:
                    raise ValueError("No models could be loaded for ensemble")

            def predict(self, protein_pairs: list[tuple[str, str]]) -> list[dict]:
                """Get ensemble predictions averaging across all models."""
                if not self.models:
                    return [{"error": "No models loaded"} for _ in protein_pairs]

                results = []
                for pair in protein_pairs:
                    model_predictions = []
                    errors = []

                    for name, model in self.models:
                        pred = model.predict([pair])[0]
                        if pred.get("error"):
                            errors.append({"model": name, "error": pred["error"]})
                        else:
                            model_predictions.append({
                                "model": name,
                                "score": pred["ml_score"],
                                "in_string": pred.get("in_string", False),
                            })

                    if model_predictions:
                        scores = [p["score"] for p in model_predictions]
                        mean_score = sum(scores) / len(scores)
                        std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5

                        results.append({
                            "protein1": pair[0],
                            "protein2": pair[1],
                            "ml_score": mean_score,
                            "model_scores": model_predictions,
                            "score_std": std_score,
                            "in_string": model_predictions[0]["in_string"],
                            "num_models": len(model_predictions),
                            "errors": errors if errors else None,
                        })
                    else:
                        results.append({
                            "protein1": pair[0],
                            "protein2": pair[1],
                            "ml_score": 0.0,
                            "error": "; ".join([f"{e['model']}: {e['error']}" for e in errors]),
                        })

                return results

            def predict_single(self, protein1: str, protein2: str) -> dict:
                """Get detailed ensemble prediction for a single pair."""
                return self.predict([(protein1, protein2)])[0]

        ensemble = PyGEnsembleLinkPredictor(existing_configs, device)
        _ensemble_predictor = ensemble
        _link_predictor = ensemble  # Use ensemble as default predictor
        logger.info(f"Loaded PyG ensemble with {len(ensemble.models)} models")
        return ensemble

    except Exception as e:
        logger.error(f"Failed to load PyG ensemble: {e}")
        return None


# =============================================================================
# Module-Level Graph Cache
# =============================================================================

_graph_cache: dict[str, KnowledgeGraph] = {}


def set_active_graph(graph: KnowledgeGraph, name: str = "default") -> None:
    """
    Set the active graph for agent tools.

    Args:
        graph: KnowledgeGraph instance to make available to tools
        name: Unique name for the graph (default: "default")
    """
    _graph_cache[name] = graph
    logger.info(
        f"Graph '{name}' cached: {graph.graph.number_of_nodes()} nodes, "
        f"{graph.graph.number_of_edges()} edges"
    )


def get_active_graph(name: str = "default") -> KnowledgeGraph | None:
    """
    Get the active graph by name.

    Args:
        name: Name of the graph to retrieve (default: "default")

    Returns:
        KnowledgeGraph if found, None otherwise
    """
    return _graph_cache.get(name)


def clear_graph_cache() -> None:
    """Clear all cached graphs."""
    _graph_cache.clear()
    logger.info("Graph cache cleared")


# =============================================================================
# Decorator for Graph Access
# =============================================================================


def requires_graph(func):
    """
    Decorator that ensures a graph is loaded before tool execution.

    If no graph is available, returns an error dictionary instead of
    executing the tool function.

    The decorated function MUST have ToolContext as its first parameter.
    """
    @wraps(func)
    def wrapper(tool_context: ToolContext, *args, **kwargs):
        graph_name = tool_context.state.get("active_graph", "default")
        graph = _graph_cache.get(graph_name)

        if graph is None:
            # Try default if specific name not found
            graph = _graph_cache.get("default")
            if graph is None:
                return {
                    "error": "No knowledge graph loaded. Build a graph first using the pipeline.",
                    "suggestion": "Use the document processing pipeline to build a knowledge graph."
                }

        # Store graph name in state for consistency
        tool_context.state["active_graph"] = graph_name
        return func(tool_context, *args, **kwargs)

    return wrapper


def _get_graph(tool_context: ToolContext) -> KnowledgeGraph:
    """
    Get the active graph for the current session.

    Call only after @requires_graph decorator has verified graph exists.
    """
    graph_name = tool_context.state.get("active_graph", "default")
    return _graph_cache.get(graph_name) or _graph_cache.get("default")


# =============================================================================
# Tool Functions for ADK Agent
# =============================================================================


@requires_graph
def get_graph_summary(tool_context: ToolContext) -> dict[str, Any]:
    """
    Get summary statistics of the knowledge graph.

    Returns node count, edge count, entity types, relationship types,
    and evidence sources.

    Args:
        tool_context: ADK tool context (injected by framework)

    Returns:
        Dictionary with graph statistics including:
        - node_count: Total number of entities
        - edge_count: Total number of relationships
        - entity_types: Count by entity type (protein, gene, disease, etc.)
        - relationship_types: Count by relationship type
        - evidence_sources: Count by evidence source (STRING, literature, ML)
        - novel_predictions: Count of ML-predicted relationships without literature support
    """
    graph = _get_graph(tool_context)
    summary = graph.to_summary()
    return {
        "node_count": summary.get("node_count", 0),
        "edge_count": summary.get("edge_count", 0),
        "relationship_count": summary.get("relationship_count", 0),
        "entity_types": summary.get("entity_types", {}),
        "relationship_types": summary.get("relationship_types", {}),
        "evidence_sources": summary.get("evidence_sources", {}),
        "ml_predicted_edges": summary.get("ml_predicted_edges", 0),
        "novel_predictions": summary.get("novel_predictions", 0),
    }


@requires_graph
def query_evidence(
    tool_context: ToolContext,
    protein1: str,
    protein2: str
) -> dict[str, Any]:
    """
    Get all evidence for a relationship between two entities.

    Retrieves all relationship data and evidence supporting the connection
    between two proteins, genes, or other entities.

    Args:
        tool_context: ADK tool context (injected by framework)
        protein1: First entity identifier (e.g., "BRCA1", "TP53")
        protein2: Second entity identifier

    Returns:
        Dictionary with:
        - relationships: All relationships between the entities
        - evidence_count: Total number of evidence sources
        - has_string: Whether STRING database evidence exists
        - has_literature: Whether literature evidence exists
        - has_ml: Whether ML prediction evidence exists
    """
    graph = _get_graph(tool_context)

    relationships = graph.get_relationship(protein1, protein2)
    reverse_rels = graph.get_relationship(protein2, protein1)

    all_rels = {}
    if relationships:
        all_rels.update(relationships)
    if reverse_rels:
        for k, v in reverse_rels.items():
            if k not in all_rels:
                all_rels[f"{k}_reverse"] = v

    if not all_rels:
        return {
            "protein1": protein1,
            "protein2": protein2,
            "relationships": {},
            "evidence_count": 0,
            "message": f"No relationship found between {protein1} and {protein2}."
        }

    evidence_count = 0
    has_string = False
    has_literature = False
    has_ml = False

    for rel_data in all_rels.values():
        evidence = rel_data.get("evidence", [])
        evidence_count += len(evidence)
        for ev in evidence:
            source_type = ev.get("source_type", "")
            if source_type == "string":
                has_string = True
            elif source_type == "literature":
                has_literature = True
            elif source_type == "ml_predicted":
                has_ml = True

    return {
        "protein1": protein1,
        "protein2": protein2,
        "relationships": all_rels,
        "evidence_count": evidence_count,
        "has_string": has_string,
        "has_literature": has_literature,
        "has_ml": has_ml,
    }


@requires_graph
def find_path(
    tool_context: ToolContext,
    source: str,
    target: str
) -> dict[str, Any]:
    """
    Find the shortest path between two entities in the knowledge graph.

    Traces the relationship chain connecting two proteins, genes, diseases,
    or other entities.

    Args:
        tool_context: ADK tool context (injected by framework)
        source: Starting entity identifier
        target: Ending entity identifier

    Returns:
        Dictionary with:
        - path_exists: Whether a path was found
        - distance: Number of hops (edges) in shortest path
        - path: List of steps with relationship details
        - alternative_paths_count: Number of alternative paths found
    """
    graph = _get_graph(tool_context)
    result = graph.find_shortest_path(source, target)

    if result.distance < 0:
        return {
            "source": source,
            "target": target,
            "path_exists": False,
            "message": f"No path found between {source} and {target}."
        }

    return {
        "source": source,
        "target": target,
        "path_exists": True,
        "distance": result.distance,
        "path": result.shortest_path,
        "alternative_paths_count": result.path_count,
        "interpretation": f"Found path of length {result.distance} with {result.path_count} total paths."
    }


@requires_graph
def compute_centrality(
    tool_context: ToolContext,
    method: str = "pagerank"
) -> dict[str, Any]:
    """
    Compute entity importance using centrality metrics.

    Identifies the most important/influential entities in the network
    using various centrality algorithms.

    Args:
        tool_context: ADK tool context (injected by framework)
        method: Centrality method - "pagerank" (default), "betweenness", "degree", or "closeness"

    Returns:
        Dictionary with:
        - method: The centrality method used
        - top_entities: List of top 10 entities with their scores
        - total_nodes: Total number of nodes analyzed
        - interpretation: Explanation of what the scores mean
    """
    graph = _get_graph(tool_context)
    result = graph.compute_centrality(method, top_k=10)

    return {
        "method": result.method,
        "top_entities": [
            {"entity": entity, "score": round(score, 4)}
            for entity, score in result.top_entities
        ],
        "total_nodes": len(result.all_scores),
        "interpretation": f"Entities ranked by {method}. Higher scores indicate greater importance in the network."
    }


@requires_graph
def detect_communities(tool_context: ToolContext) -> dict[str, Any]:
    """
    Detect communities/clusters in the knowledge graph.

    Uses the Louvain algorithm to identify tightly connected clusters
    that may represent protein complexes, pathways, or functional modules.

    Args:
        tool_context: ADK tool context (injected by framework)

    Returns:
        Dictionary with:
        - num_communities: Number of communities found
        - modularity: Quality score of the partition (0-1, higher is better)
        - communities: List of communities with member entities
        - interpretation: Summary of findings
    """
    graph = _get_graph(tool_context)
    result = graph.detect_communities()

    return {
        "num_communities": result.num_communities,
        "modularity": round(result.modularity, 4),
        "communities": result.communities,
        "interpretation": f"Found {result.num_communities} communities with modularity {result.modularity:.3f}."
    }


@requires_graph
def run_diamond(
    tool_context: ToolContext,
    seed_genes: list[str],
    max_iterations: int = 200
) -> dict[str, Any]:
    """
    Run DIAMOnD algorithm to detect disease modules from seed genes.

    Based on Ghiassian et al. (2015) for identifying disease-associated
    gene modules through network connectivity analysis.

    Args:
        tool_context: ADK tool context (injected by framework)
        seed_genes: Known disease-associated genes (e.g., from GWAS or known biomarkers)
        max_iterations: Maximum genes to add to the module (default: 200)

    Returns:
        Dictionary with:
        - seed_genes: Input seed genes that were found in graph
        - module_genes: Genes added to the disease module
        - module_size: Total size of the module
        - top_candidates: Highest-scoring candidates for module membership
        - interpretation: Summary of module expansion
    """
    graph = _get_graph(tool_context)
    result = graph.run_diamond(seed_genes, max_iterations)

    return {
        "seed_genes": result.seed_genes,
        "module_genes": result.module_genes[:20],
        "module_size": result.module_size,
        "iterations_run": result.iterations_run,
        "top_candidates": [
            {"gene": gene, "score": round(score, 4)}
            for gene, score in result.ranked_candidates[:10]
        ],
        "interpretation": f"Expanded from {len(result.seed_genes)} seeds to {result.module_size} genes."
    }


@requires_graph
def calculate_proximity(
    tool_context: ToolContext,
    drug_targets: list[str],
    disease_genes: list[str]
) -> dict[str, Any]:
    """
    Calculate network proximity between drug targets and disease genes.

    Based on Guney et al. (2016) for predicting drug efficacy through
    network-based drug-disease relationships.

    Args:
        tool_context: ADK tool context (injected by framework)
        drug_targets: List of drug target genes/proteins
        disease_genes: List of disease-associated genes

    Returns:
        Dictionary with:
        - observed_distance: Actual network distance
        - expected_distance: Expected distance by random chance
        - z_score: Statistical significance (negative = closer than random)
        - p_value: Probability value
        - is_significant: Whether the proximity is statistically significant
        - interpretation: Explanation of the result
    """
    graph = _get_graph(tool_context)
    result = graph.calculate_network_proximity(drug_targets, disease_genes, n_random=100)

    return {
        "drug_targets": result.drug_targets,
        "disease_genes": result.disease_genes,
        "observed_distance": round(result.observed_distance, 4) if result.observed_distance != float('inf') else None,
        "expected_distance": round(result.expected_distance, 4) if result.expected_distance != float('inf') else None,
        "z_score": round(result.z_score, 4),
        "p_value": round(result.p_value, 4),
        "is_significant": result.is_significant,
        "interpretation": result.interpretation
    }


@requires_graph
def predict_synergy(
    tool_context: ToolContext,
    target1: str,
    target2: str,
    disease_genes: list[str]
) -> dict[str, Any]:
    """
    Predict synergy potential for two drug targets in combination therapy.

    Based on Cheng et al. (2019) for network-based drug combination prediction.
    Evaluates whether targeting both proteins would have complementary effects.

    Args:
        tool_context: ADK tool context (injected by framework)
        target1: First drug target (gene/protein name)
        target2: Second drug target (gene/protein name)
        disease_genes: Disease-associated genes for context

    Returns:
        Dictionary with:
        - synergy_score: Overall synergy potential (0-1, higher is better)
        - target_separation: Network distance between targets
        - module_coverage: Fraction of disease module reached by both targets
        - complementarity: How non-overlapping the target neighborhoods are
        - interpretation: Explanation of synergy potential
    """
    graph = _get_graph(tool_context)
    result = graph.predict_synergy(target1, target2, disease_genes)

    return {
        "target1": result.target1,
        "target2": result.target2,
        "synergy_score": round(result.synergy_score, 4),
        "target_separation": result.target_separation,
        "module_coverage": round(result.module_coverage, 4),
        "complementarity": round(result.complementarity, 4),
        "interpretation": result.interpretation
    }


@requires_graph
def analyze_robustness(
    tool_context: ToolContext,
    target: str,
    disease_genes: list[str]
) -> dict[str, Any]:
    """
    Analyze network robustness impact of targeting a specific node.

    Evaluates therapeutic potential by measuring disease impact vs
    global network disruption. A good target disrupts disease pathways
    while minimizing effects on healthy cellular functions.

    Args:
        tool_context: ADK tool context (injected by framework)
        target: Drug target to analyze
        disease_genes: Disease-associated genes

    Returns:
        Dictionary with:
        - disease_impact: Fraction of disease pathways affected
        - global_impact: Fraction of global network affected
        - therapeutic_index: Ratio of disease to global impact (higher is better)
        - compensatory_paths: Number of alternative pathways that could compensate
        - interpretation: Safety and efficacy assessment
    """
    graph = _get_graph(tool_context)
    result = graph.analyze_robustness(target, disease_genes)

    return {
        "target": result.target,
        "disease_impact": round(result.disease_impact, 4),
        "global_impact": round(result.global_impact, 4),
        "therapeutic_index": round(result.therapeutic_index, 4),
        "compensatory_paths": result.compensatory_paths,
        "interpretation": result.interpretation
    }


@requires_graph
def get_predictions(
    tool_context: ToolContext,
    min_score: float = 0.5
) -> dict[str, Any]:
    """
    Get ML link predictions above a confidence threshold.

    Retrieves novel predicted interactions that don't have literature
    support but are predicted by the ML model.

    Args:
        tool_context: ADK tool context (injected by framework)
        min_score: Minimum ML prediction score (0-1, default: 0.7)

    Returns:
        Dictionary with:
        - predictions: List of predicted interactions with scores
        - count: Number of predictions found
        - threshold: The score threshold used
    """
    graph = _get_graph(tool_context)
    predictions = graph.get_novel_predictions(min_score)

    return {
        "predictions": predictions[:20],  # Limit to top 20
        "count": len(predictions),
        "threshold": min_score,
        "interpretation": f"Found {len(predictions)} novel predictions above {min_score} confidence threshold."
    }


@requires_graph
def predict_interaction(
    tool_context: ToolContext,
    protein1: str,
    protein2: str
) -> dict[str, Any]:
    """
    Predict interaction probability between two proteins using PyG ML model.

    Uses trained PyTorch Geometric Node2Vec link predictor to compute
    the probability that two proteins interact. If ensemble mode is enabled,
    returns predictions from multiple models (balanced, structural, homophily).

    Args:
        tool_context: ADK tool context (injected by framework)
        protein1: First protein name (e.g., "BRCA1", "TP53")
        protein2: Second protein name

    Returns:
        Dictionary with:
        - protein1: First protein name
        - protein2: Second protein name
        - ml_score: Predicted interaction probability (0-1)
        - in_string: Whether the interaction exists in STRING database
        - model_scores: Individual model predictions (if ensemble mode)
        - confidence: Interpretation of the prediction strength
        - error: Error message if prediction failed
    """
    predictor = get_link_predictor()

    if predictor is None:
        return {
            "protein1": protein1,
            "protein2": protein2,
            "ml_score": None,
            "error": "No ML predictor loaded. Call load_pyg_predictor() or load_pyg_ensemble() first.",
            "suggestion": "Load the PyG link predictor using load_pyg_predictor('models/pyg_link_predictor.pkl')"
        }

    try:
        result = predictor.predict([(protein1, protein2)])[0]

        if result.get("error"):
            return {
                "protein1": protein1,
                "protein2": protein2,
                "ml_score": None,
                "error": result["error"],
            }

        ml_score = result.get("ml_score", 0.0)

        # Interpret the prediction
        if ml_score >= 0.8:
            confidence = "high"
            interpretation = f"Strong prediction of interaction (score={ml_score:.3f})"
        elif ml_score >= 0.6:
            confidence = "moderate"
            interpretation = f"Moderate prediction of interaction (score={ml_score:.3f})"
        elif ml_score >= 0.4:
            confidence = "low"
            interpretation = f"Weak/uncertain prediction (score={ml_score:.3f})"
        else:
            confidence = "very_low"
            interpretation = f"Low interaction likelihood (score={ml_score:.3f})"

        response = {
            "protein1": protein1,
            "protein2": protein2,
            "ml_score": round(ml_score, 4),
            "in_string": result.get("in_string", False),
            "confidence": confidence,
            "interpretation": interpretation,
        }

        # Add ensemble details if available
        if "model_scores" in result:
            response["model_scores"] = result["model_scores"]
            response["score_std"] = round(result.get("score_std", 0.0), 4)
            response["num_models"] = result.get("num_models", 1)

            # Enhanced interpretation for ensemble
            if result.get("score_std", 0) > 0.1:
                response["interpretation"] += f" (models disagree, std={result['score_std']:.3f})"
            else:
                response["interpretation"] += " (models agree)"

        return response

    except Exception as e:
        logger.error(f"Prediction failed for {protein1}-{protein2}: {e}")
        return {
            "protein1": protein1,
            "protein2": protein2,
            "ml_score": None,
            "error": str(e),
        }


@requires_graph
def predict_batch_interactions(
    tool_context: ToolContext,
    protein_pairs: list[list[str]]
) -> dict[str, Any]:
    """
    Predict interaction probabilities for multiple protein pairs.

    Efficiently batch-processes multiple protein pairs using the PyG model.

    Args:
        tool_context: ADK tool context (injected by framework)
        protein_pairs: List of [protein1, protein2] pairs

    Returns:
        Dictionary with:
        - predictions: List of prediction results
        - count: Number of predictions made
        - avg_score: Average prediction score
        - high_confidence: Number of high-confidence predictions (>= 0.7)
    """
    predictor = get_link_predictor()

    if predictor is None:
        return {
            "predictions": [],
            "count": 0,
            "error": "No ML predictor loaded. Call load_pyg_predictor() or load_pyg_ensemble() first."
        }

    try:
        # Convert list of lists to list of tuples
        pairs = [(p[0], p[1]) for p in protein_pairs if len(p) >= 2]

        if not pairs:
            return {
                "predictions": [],
                "count": 0,
                "error": "No valid protein pairs provided."
            }

        results = predictor.predict(pairs)

        # Process results
        valid_predictions = []
        scores = []

        for result in results:
            if not result.get("error"):
                ml_score = result.get("ml_score", 0.0)
                scores.append(ml_score)
                valid_predictions.append({
                    "protein1": result["protein1"],
                    "protein2": result["protein2"],
                    "ml_score": round(ml_score, 4),
                    "in_string": result.get("in_string", False),
                })
            else:
                valid_predictions.append({
                    "protein1": result.get("protein1"),
                    "protein2": result.get("protein2"),
                    "ml_score": None,
                    "error": result["error"],
                })

        # Sort by score descending
        valid_predictions.sort(
            key=lambda x: x.get("ml_score") or 0.0,
            reverse=True
        )

        avg_score = sum(scores) / len(scores) if scores else 0.0
        high_confidence = sum(1 for s in scores if s >= 0.7)

        return {
            "predictions": valid_predictions[:50],  # Limit to 50
            "count": len(valid_predictions),
            "avg_score": round(avg_score, 4),
            "high_confidence": high_confidence,
            "interpretation": f"Processed {len(pairs)} pairs. {high_confidence} high-confidence predictions (>= 0.7)."
        }

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return {
            "predictions": [],
            "count": 0,
            "error": str(e),
        }


@requires_graph
def run_link_predictions(
    tool_context: ToolContext,
    min_score: float = 0.3,
    max_predictions: int = 50
) -> dict[str, Any]:
    """
    Run ML link predictor on all protein pairs and add predictions to the graph.

    This tool evaluates ALL protein pairs not currently connected in the graph,
    runs the ML predictor on them, and adds high-scoring predictions as new
    edges with ml_score metadata. After running this, use get_predictions()
    to retrieve the results.

    Predictions are categorized as:
    - enrichment: Known STRING interactions not yet in graph (high confidence)
    - novel_prediction: ML predictions NOT in STRING (truly novel hypotheses)

    Args:
        tool_context: ADK tool context (injected by framework)
        min_score: Minimum ML score threshold (0-1, default 0.3)
        max_predictions: Maximum predictions to add to graph (default 50)

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - enrichment_count: Number of STRING-backed edges added
        - novel_count: Number of novel predictions added
        - total_pairs_evaluated: Total candidate pairs checked
        - top_predictions: List of top predictions added
    """
    from models.knowledge_graph import RelationshipType, EvidenceSource

    predictor = get_link_predictor()
    graph = _get_graph(tool_context)

    if predictor is None:
        return {
            "status": "error",
            "message": "No ML predictor loaded. The predictor may not be initialized.",
            "enrichment_count": 0,
            "novel_count": 0,
            "total_pairs_evaluated": 0,
            "top_predictions": [],
        }

    # Get all protein/gene entities
    proteins = [
        entity_id
        for entity_id, data in graph.entities.items()
        if data.get("type") in ["protein", "gene", "unknown"]
    ]

    if len(proteins) < 2:
        return {
            "status": "error",
            "message": f"Not enough proteins for prediction (found {len(proteins)})",
            "enrichment_count": 0,
            "novel_count": 0,
            "total_pairs_evaluated": 0,
            "top_predictions": [],
        }

    # Generate candidate pairs not already in graph
    candidate_pairs = []
    for i, p1 in enumerate(proteins):
        for p2 in proteins[i + 1:]:
            # Check if edge exists (in either direction)
            existing = graph.get_relationship(p1, p2)
            if not existing:
                candidate_pairs.append((p1, p2))

    if not candidate_pairs:
        return {
            "status": "success",
            "message": "All protein pairs already have edges in the graph",
            "enrichment_count": 0,
            "novel_count": 0,
            "total_pairs_evaluated": 0,
            "top_predictions": [],
        }

    logger.info(f"Running ML predictions on {len(candidate_pairs)} candidate pairs")

    # Run predictions
    try:
        results = predictor.predict(candidate_pairs)
    except Exception as e:
        logger.error(f"ML prediction failed: {e}")
        return {
            "status": "error",
            "message": f"ML prediction failed: {str(e)}",
            "enrichment_count": 0,
            "novel_count": 0,
            "total_pairs_evaluated": len(candidate_pairs),
            "top_predictions": [],
        }

    # Categorize results
    enrichment_results = []  # in_string=True
    novel_results = []  # in_string=False, score >= threshold

    for result in results:
        if result.get("error"):
            continue

        ml_score = result.get("ml_score")
        if ml_score is None:
            continue

        in_string = result.get("in_string", False)

        if in_string:
            # Known STRING interaction - add as enrichment
            enrichment_results.append(result)
        elif ml_score >= min_score:
            # Novel prediction above threshold
            novel_results.append(result)

    # Sort by score
    enrichment_results.sort(key=lambda x: x.get("ml_score", 0), reverse=True)
    novel_results.sort(key=lambda x: x.get("ml_score", 0), reverse=True)

    # Add to graph (limit to max_predictions total)
    added_enrichment = []
    added_novel = []

    # Add enrichment edges (half of max_predictions)
    max_enrichment = max_predictions // 2
    for result in enrichment_results[:max_enrichment]:
        graph.add_relationship(
            source=result["protein1"],
            target=result["protein2"],
            rel_type=RelationshipType.INTERACTS_WITH,
            ml_score=result["ml_score"],
            evidence=[{
                "source_type": EvidenceSource.ML_PREDICTED.value,
                "source_id": "link_predictor",
                "confidence": result["ml_score"],
            }],
            link_category="enrichment",
        )
        added_enrichment.append({
            "protein1": result["protein1"],
            "protein2": result["protein2"],
            "ml_score": round(result["ml_score"], 4),
            "in_string": True,
            "category": "enrichment",
        })

    # Add novel predictions (remaining half)
    max_novel = max_predictions - len(added_enrichment)
    for result in novel_results[:max_novel]:
        graph.add_relationship(
            source=result["protein1"],
            target=result["protein2"],
            rel_type=RelationshipType.HYPOTHESIZED,
            ml_score=result["ml_score"],
            evidence=[{
                "source_type": EvidenceSource.ML_PREDICTED.value,
                "source_id": "link_predictor",
                "confidence": result["ml_score"],
            }],
            link_category="novel_prediction",
        )
        added_novel.append({
            "protein1": result["protein1"],
            "protein2": result["protein2"],
            "ml_score": round(result["ml_score"], 4),
            "in_string": False,
            "category": "novel_prediction",
        })

    # Combine for top predictions
    all_added = added_enrichment + added_novel
    all_added.sort(key=lambda x: x["ml_score"], reverse=True)

    logger.info(
        f"Added {len(added_enrichment)} enrichment edges and "
        f"{len(added_novel)} novel predictions to graph"
    )

    return {
        "status": "success",
        "message": f"Added {len(added_enrichment)} enrichment edges and {len(added_novel)} novel predictions",
        "enrichment_count": len(added_enrichment),
        "novel_count": len(added_novel),
        "total_pairs_evaluated": len(candidate_pairs),
        "total_enrichment_available": len(enrichment_results),
        "total_novel_available": len(novel_results),
        "top_predictions": all_added[:20],
    }


@requires_graph
def generate_hypothesis(
    tool_context: ToolContext,
    protein1: str,
    protein2: str
) -> dict[str, Any]:
    """
    Generate a testable hypothesis for a predicted interaction.

    Creates a structured hypothesis for a potential protein-protein
    interaction, including supporting context from the network and
    ML prediction scores from PyG models.

    Args:
        tool_context: ADK tool context (injected by framework)
        protein1: First protein in the predicted interaction
        protein2: Second protein in the predicted interaction

    Returns:
        Dictionary with:
        - hypothesis: Structured hypothesis statement
        - ml_prediction: PyG model prediction scores
        - supporting_evidence: Network context supporting the prediction
        - suggested_experiments: Validation experiments to test the hypothesis
        - confidence_factors: Factors contributing to prediction confidence
    """
    graph = _get_graph(tool_context)

    # Get ML prediction from PyG model if available
    ml_prediction = None
    ml_confidence = "unknown"
    predictor = get_link_predictor()

    if predictor is not None:
        try:
            pred_result = predictor.predict([(protein1, protein2)])[0]
            if not pred_result.get("error"):
                ml_score = pred_result.get("ml_score", 0.0)
                ml_prediction = {
                    "score": round(ml_score, 4),
                    "in_string": pred_result.get("in_string", False),
                }

                # Add ensemble details if available
                if "model_scores" in pred_result:
                    ml_prediction["model_scores"] = pred_result["model_scores"]
                    ml_prediction["score_std"] = round(pred_result.get("score_std", 0.0), 4)
                    ml_prediction["num_models"] = pred_result.get("num_models", 1)

                # Determine confidence level
                if ml_score >= 0.8:
                    ml_confidence = "high"
                elif ml_score >= 0.6:
                    ml_confidence = "moderate"
                elif ml_score >= 0.4:
                    ml_confidence = "low"
                else:
                    ml_confidence = "very_low"
            else:
                ml_prediction = {"error": pred_result.get("error")}
        except Exception as e:
            logger.warning(f"ML prediction failed for hypothesis: {e}")
            ml_prediction = {"error": str(e)}

    # Get context from the graph
    p1_summary = graph.get_entity_interactions_summary(protein1)
    p2_summary = graph.get_entity_interactions_summary(protein2)

    # Check for shared neighbors (potential mechanism)
    p1_neighbors = set()
    p2_neighbors = set()

    for neighbor_id, _ in graph.get_neighbors(protein1, 20):
        p1_neighbors.add(neighbor_id)
    for neighbor_id, _ in graph.get_neighbors(protein2, 20):
        p2_neighbors.add(neighbor_id)

    shared_neighbors = list(p1_neighbors & p2_neighbors)

    # Find path between them
    path_result = graph.find_shortest_path(protein1, protein2)
    path_info = None
    if path_result.distance > 0:
        path_info = {
            "distance": path_result.distance,
            "path": path_result.shortest_path
        }

    # Check for existing evidence
    existing_rel = graph.get_relationship(protein1, protein2)

    # Generate hypothesis statement based on ML score
    if ml_prediction and ml_prediction.get("score"):
        ml_score = ml_prediction["score"]
        if ml_score >= 0.7:
            hypothesis_stmt = (
                f"Based on network topology analysis (ML score: {ml_score:.2f}), "
                f"{protein1} and {protein2} are predicted to functionally interact. "
                "This may occur through shared pathway components or direct binding."
            )
        elif ml_score >= 0.5:
            hypothesis_stmt = (
                f"{protein1} and {protein2} show moderate likelihood of interaction "
                f"(ML score: {ml_score:.2f}), potentially through indirect mechanisms "
                "or shared protein complexes."
            )
        else:
            hypothesis_stmt = (
                f"{protein1} and {protein2} show weak evidence for direct interaction "
                f"(ML score: {ml_score:.2f}), but may have functional relationships "
                "worth investigating."
            )
    else:
        hypothesis_stmt = (
            f"{protein1} and {protein2} may functionally interact, "
            "potentially through shared pathway components or protein complexes."
        )

    hypothesis = {
        "protein1": protein1,
        "protein2": protein2,
        "hypothesis_statement": hypothesis_stmt,
        "ml_prediction": ml_prediction,
        "ml_confidence": ml_confidence,
        "supporting_evidence": {
            "protein1_context": p1_summary[:500] if len(p1_summary) > 500 else p1_summary,
            "protein2_context": p2_summary[:500] if len(p2_summary) > 500 else p2_summary,
            "shared_neighbors": shared_neighbors[:10],
            "network_path": path_info,
            "existing_relationship": existing_rel is not None
        },
        "suggested_experiments": [
            "Co-immunoprecipitation (Co-IP) to test physical interaction",
            "Yeast two-hybrid (Y2H) assay for binary interaction",
            "Proximity ligation assay (PLA) for in situ validation",
            "CRISPR knockout followed by phenotype analysis",
            "RNA-seq after perturbation of one protein"
        ],
        "confidence_factors": {
            "ml_score": ml_prediction.get("score") if ml_prediction else None,
            "ml_confidence": ml_confidence,
            "shared_neighbors_count": len(shared_neighbors),
            "path_distance": path_result.distance if path_result.distance > 0 else "disconnected",
            "has_supporting_evidence": existing_rel is not None
        }
    }

    # Add interpretation based on ensemble model agreement
    if ml_prediction and ml_prediction.get("model_scores"):
        scores = [m["score"] for m in ml_prediction["model_scores"]]
        structural_score = next(
            (m["score"] for m in ml_prediction["model_scores"] if m["model"] == "structural"),
            None
        )
        homophily_score = next(
            (m["score"] for m in ml_prediction["model_scores"] if m["model"] == "homophily"),
            None
        )

        if structural_score is not None and homophily_score is not None:
            if abs(structural_score - homophily_score) > 0.15:
                if structural_score > homophily_score:
                    hypothesis["model_interpretation"] = (
                        "Structural model scores higher than homophily model, suggesting "
                        "a role-based rather than neighborhood-based relationship."
                    )
                else:
                    hypothesis["model_interpretation"] = (
                        "Homophily model scores higher than structural model, suggesting "
                        "proteins share similar local network neighborhoods."
                    )
            else:
                hypothesis["model_interpretation"] = (
                    "All models agree on prediction, suggesting consistent evidence "
                    "across different network perspectives."
                )

    return hypothesis


@requires_graph
def get_entity_neighborhood(
    tool_context: ToolContext,
    entity: str,
    max_neighbors: int = 10
) -> dict[str, Any]:
    """
    Get the neighborhood of an entity in the knowledge graph.

    Returns all directly connected entities with relationship details.

    Args:
        tool_context: ADK tool context (injected by framework)
        entity: Entity identifier to get neighbors for
        max_neighbors: Maximum number of neighbors to return (default: 10)

    Returns:
        Dictionary with:
        - entity: The queried entity
        - neighbors: List of neighbors with relationship details
        - count: Number of neighbors found
    """
    graph = _get_graph(tool_context)

    if entity not in graph.entities:
        return {
            "entity": entity,
            "neighbors": [],
            "count": 0,
            "error": f"Entity '{entity}' not found in the knowledge graph."
        }

    neighbors = graph.get_neighbors(entity, max_neighbors)

    formatted_neighbors = []
    for neighbor_id, rel_data in neighbors:
        formatted_neighbors.append({
            "neighbor": neighbor_id,
            "direction": rel_data.get("direction", "unknown"),
            "relation_type": rel_data.get("relation_type", "unknown"),
            "ml_score": rel_data.get("ml_score"),
            "evidence_count": len(rel_data.get("evidence", [])),
        })

    return {
        "entity": entity,
        "neighbors": formatted_neighbors,
        "count": len(formatted_neighbors),
    }


# =============================================================================
# Agent Instruction
# =============================================================================

GRAPH_AGENT_INSTRUCTION = """You are an expert biomedical knowledge graph analyst specializing in drug discovery and systems biology.

You help researchers explore and analyze relationships between proteins, genes, diseases, drugs, and biological mechanisms within a knowledge graph.

## Your Capabilities

1. **Graph Exploration**:
   - Summarize the graph (get_graph_summary)
   - Query evidence between entities (query_evidence)
   - Find paths between entities (find_path)
   - Get entity neighborhoods (get_entity_neighborhood)

2. **Importance Analysis**:
   - Compute centrality metrics (compute_centrality) - PageRank, betweenness, degree, closeness
   - Detect communities/clusters (detect_communities)

3. **Drug Discovery Algorithms**:
   - DIAMOnD disease module detection (run_diamond)
   - Network proximity analysis (calculate_proximity)
   - Drug synergy prediction (predict_synergy)
   - Target robustness analysis (analyze_robustness)

4. **ML Link Predictions** (PyTorch Geometric):
   - Predict interaction between two proteins (predict_interaction) - Uses PyG Node2Vec model
   - Batch predict multiple protein pairs (predict_batch_interactions)
   - Get novel link predictions from graph (get_predictions)
   - Generate testable hypotheses with ML scores (generate_hypothesis)

## ML Prediction Models

The system uses PyTorch Geometric Node2Vec models trained on STRING physical interactions:
- **Balanced model** (p=1, q=1): Standard random walks for general predictions
- **Structural model** (p=1, q=0.5): DFS-like walks capturing global structural roles
- **Homophily model** (p=1, q=2): BFS-like walks capturing local neighborhood similarity

When ensemble mode is enabled, predictions average across all three models with interpretation of model agreement/disagreement.

## ReAct Reasoning Process

For each query, follow this structured reasoning:

1. **Thought**: Analyze what the user is asking and what information you need.
2. **Action**: Call the appropriate tool(s) to get data.
3. **Observation**: Analyze the tool results.
4. **Repeat**: If more information is needed, continue with another Thought-Action-Observation cycle.
5. **Answer**: Provide a comprehensive response synthesizing all findings.

## Guidelines

- Always start by understanding the graph structure with get_graph_summary if you haven't already.
- When asked about specific entities, check if they exist using query_evidence or get_entity_neighborhood.
- For "important" or "central" proteins, use compute_centrality with pagerank or betweenness.
- For drug discovery questions, combine multiple tools (e.g., DIAMOnD + proximity + synergy).
- **For novel interaction predictions**, use predict_interaction to get ML confidence scores.
- **For hypothesis generation**, use generate_hypothesis which includes ML predictions automatically.
- Always explain the biological significance of your findings.
- Distinguish between experimentally validated relationships and ML predictions.
- Suggest follow-up analyses when relevant.
- Be precise about statistical significance and limitations.

## Example Queries and Approaches

"What proteins are most important in this network?"
-> Use compute_centrality with pagerank, then detect_communities to see if they cluster.

"Is there a connection between BRCA1 and TP53?"
-> Use query_evidence first, then predict_interaction to get ML score if no direct relationship.

"Would targeting both EGFR and KRAS be synergistic for cancer?"
-> Use predict_synergy with relevant disease genes, analyze_robustness for each target.

"Find disease module genes starting from known Alzheimer's genes."
-> Use run_diamond with the known genes as seeds, then compute_centrality on the module.

"Predict if CDK1 interacts with AURKA"
-> Use predict_interaction to get ML-based interaction probability and model agreement.

"Generate a hypothesis for potential CDK2-CCNE1 interaction"
-> Use generate_hypothesis which includes ML predictions, network context, and suggested experiments.

Remember: You are helping scientists make discoveries. Be thorough, accurate, and highlight actionable insights.
"""


# =============================================================================
# Agent Creation
# =============================================================================


def create_graph_agent(model: str = "gemini-2.5-flash") -> LlmAgent:
    """
    Create the GraphRAG Q&A agent with PlanReActPlanner.

    Args:
        model: Gemini model to use (default: gemini-2.5-flash)

    Returns:
        Configured LlmAgent instance with ReAct planning
    """
    planner = PlanReActPlanner()

    agent = LlmAgent(
        model=model,
        name="graph_query_agent",
        description="Expert biomedical knowledge graph analyst for drug discovery research with PyG ML predictions",
        instruction=GRAPH_AGENT_INSTRUCTION,
        planner=planner,
        tools=[
            # Graph exploration
            get_graph_summary,
            query_evidence,
            find_path,
            get_entity_neighborhood,
            # Importance analysis
            compute_centrality,
            detect_communities,
            # Drug discovery algorithms
            run_diamond,
            calculate_proximity,
            predict_synergy,
            analyze_robustness,
            # ML predictions (PyG)
            predict_interaction,
            predict_batch_interactions,
            get_predictions,
            generate_hypothesis,
        ],
    )

    logger.info(f"Created GraphRAG Q&A agent with PlanReActPlanner, model={model}")
    return agent


# =============================================================================
# Session Manager
# =============================================================================


class GraphAgentManager:
    """
    Manages GraphRAG agent sessions for conversational Q&A.

    Provides a simple interface for:
    - Setting the active knowledge graph
    - Processing user queries with session continuity
    - Retrieving conversation history
    """

    def __init__(
        self,
        graph: KnowledgeGraph | None = None,
        model: str = "gemini-2.5-flash",
        link_predictor: LinkPredictorProtocol | None = None,
    ):
        """
        Initialize the GraphRAG agent manager.

        Args:
            graph: Optional KnowledgeGraph to use. If None, graph must be
                   set later using set_graph() or set_active_graph().
            model: Gemini model to use (default: gemini-2.5-flash)
            link_predictor: Optional link predictor for ML predictions.
                           If provided, sets the module-level predictor.
        """
        if graph:
            set_active_graph(graph)

        if link_predictor:
            set_link_predictor(link_predictor)
            logger.info("Link predictor initialized in GraphAgentManager")

        self.model = model
        self.agent = create_graph_agent(model)
        self.session_service = InMemorySessionService()
        self.app_name = "graph_query_agent"

        self.runner = Runner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=self.session_service,
        )

        self._sessions: dict[str, bool] = {}

        logger.info("GraphAgentManager initialized")

    def set_graph(self, graph: KnowledgeGraph, name: str = "default") -> None:
        """
        Set the active knowledge graph.

        Args:
            graph: KnowledgeGraph instance
            name: Name to cache the graph under
        """
        set_active_graph(graph, name)

    async def _ensure_session(self, user_id: str, session_id: str) -> None:
        """Ensure a session exists, creating if necessary."""
        session_key = f"{user_id}:{session_id}"
        if session_key not in self._sessions:
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
            )
            self._sessions[session_key] = True
            logger.debug(f"Created session: {session_key}")

    async def query(
        self,
        user_id: str,
        session_id: str,
        query: str
    ) -> dict[str, Any]:
        """
        Process a user query through the GraphRAG agent.

        Args:
            user_id: User identifier for session management
            session_id: Session identifier for conversation continuity
            query: Natural language query

        Returns:
            Dictionary with:
            - query: The original query
            - response: Agent's response text
            - tool_calls: List of tools used
            - thoughts: Model's reasoning (if available)
            - session_id: Session identifier
        """
        await self._ensure_session(user_id, session_id)

        content = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )

        response_text = ""
        thoughts = []
        tool_calls = []

        try:
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content
            ):
                # Collect tool calls
                if hasattr(event, 'tool_calls') and event.tool_calls:
                    for tc in event.tool_calls:
                        tool_calls.append({
                            "name": tc.name,
                            "args": tc.args if hasattr(tc, 'args') else {}
                        })

                # Get final response
                if event.is_final_response() and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
                        if hasattr(part, 'thought') and part.thought:
                            thoughts.append(part.thought)

            return {
                "query": query,
                "response": response_text.strip(),
                "tool_calls": tool_calls,
                "thoughts": thoughts,
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "query": query,
                "response": f"I encountered an error processing your query: {str(e)}",
                "tool_calls": [],
                "thoughts": [],
                "session_id": session_id,
                "error": str(e),
            }

    def query_sync(
        self,
        user_id: str,
        session_id: str,
        query: str
    ) -> dict[str, Any]:
        """
        Synchronous wrapper for query.

        Args:
            user_id: User identifier
            session_id: Session identifier
            query: Natural language query

        Returns:
            Query result dictionary
        """
        return asyncio.run(self.query(user_id, session_id, query))

    async def get_history(
        self,
        user_id: str,
        session_id: str
    ) -> list[dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            List of messages with role and text
        """
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )

        if not session:
            return []

        history = []
        for event in session.events:
            if hasattr(event, 'content') and event.content:
                text = ""
                if event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text += part.text
                history.append({
                    "role": event.content.role,
                    "text": text,
                })

        return history

    async def clear_session(self, user_id: str, session_id: str) -> None:
        """
        Clear a session's conversation history.

        Args:
            user_id: User identifier
            session_id: Session identifier
        """
        session_key = f"{user_id}:{session_id}"
        if session_key in self._sessions:
            del self._sessions[session_key]
            logger.info(f"Session cleared: {session_key}")


# =============================================================================
# Module Exports
# =============================================================================

def is_langfuse_enabled() -> bool:
    """Check if Langfuse tracing is enabled for this module."""
    return _LANGFUSE_ENABLED


__all__ = [
    # Observability
    "is_langfuse_enabled",
    # Graph cache functions
    "set_active_graph",
    "get_active_graph",
    "clear_graph_cache",
    # Link predictor functions
    "set_link_predictor",
    "get_link_predictor",
    "set_ensemble_predictor",
    "get_ensemble_predictor",
    "load_pyg_predictor",
    "load_pyg_ensemble",
    # Agent creation
    "create_graph_agent",
    "GraphAgentManager",
    "GRAPH_AGENT_INSTRUCTION",
    # Tool functions (for direct use)
    "predict_interaction",
    "predict_batch_interactions",
    "generate_hypothesis",
]
