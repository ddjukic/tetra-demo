"""
Scientific Knowledge Graph Agent - Hybrid Architecture with Q&A Focus.

This agent uses the hybrid architecture:
- build_knowledge_graph: Calls KGOrchestrator (DataFetchAgent + KGPipeline)
- expand_graph: Adds more data to existing graph
- Q&A tools: Query, analyze, and explore the built graph

Run with: adk api_server kg_agent/ --port 8080
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

from pipeline.orchestrator import KGOrchestrator, get_orchestrator, set_orchestrator
from models.knowledge_graph import KnowledgeGraph, RelationshipType, EvidenceSource

# =============================================================================
# Initialize Dependencies at Module Load
# =============================================================================

# Load link predictor (PyG preferred with GPU support, gensim as fallback)
_pyg_model_path = project_root / "models" / "pyg_link_predictor.pkl"
_gensim_model_path = project_root / "models" / "gensim_link_predictor.pkl"
_link_predictor = None

# Try PyG first (requires torch, torch_geometric from [gpu] extras)
if _pyg_model_path.exists():
    try:
        from ml.pyg_link_predictor import PyGLinkPredictor
        _link_predictor = PyGLinkPredictor.load(str(_pyg_model_path))
    except ImportError:
        pass  # PyTorch not installed, will try gensim

# Fallback to gensim-based predictor
if _link_predictor is None and _gensim_model_path.exists():
    try:
        from ml.link_predictor import LinkPredictor
        _link_predictor = LinkPredictor.load(str(_gensim_model_path))
    except Exception:
        pass

# Last resort: create empty predictor that returns no predictions
if _link_predictor is None:
    from ml.link_predictor import LinkPredictor
    _link_predictor = LinkPredictor()

# Create and set the orchestrator
_orchestrator = KGOrchestrator(extractor_name="cerebras")
set_orchestrator(_orchestrator)


# =============================================================================
# Graph Building Tools (via Orchestrator)
# =============================================================================

async def build_knowledge_graph(
    user_query: str,
    max_articles: int = 50,
    tool_context: ToolContext = None,
) -> dict:
    """
    Build a new knowledge graph from a user query.

    This uses the hybrid architecture:
    1. DataFetchAgent collects data (STRING, PubMed) using LLM reasoning
    2. KGPipeline processes data (extraction, graph building) deterministically

    Args:
        user_query: Natural language description of the graph to build.
            Examples:
            - "Build a KG for the orexin signaling pathway"
            - "Create a knowledge graph about BRCA1 and DNA repair"
            - "Explore insulin signaling in diabetes"
        max_articles: Maximum articles to fetch from PubMed. Default 50.

    Returns:
        Summary of the built graph including node/edge counts and entity types.
    """
    orchestrator = get_orchestrator()

    try:
        graph = await orchestrator.build(
            user_query=user_query,
            max_articles=max_articles,
        )

        summary = graph.to_summary()

        return {
            "status": "success",
            "message": f"Built knowledge graph with {summary['node_count']} nodes and {summary['edge_count']} edges",
            "summary": summary,
            "node_count": summary.get("node_count", 0),
            "edge_count": summary.get("edge_count", 0),
            "entity_types": summary.get("entity_types", {}),
            "relationship_types": summary.get("relationship_types", {}),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to build graph: {str(e)}",
            "node_count": 0,
            "edge_count": 0,
        }


async def expand_graph(
    expansion_query: str,
    max_articles: int = 30,
    tool_context: ToolContext = None,
) -> dict:
    """
    Expand the current graph with additional data.

    Use this to add more entities, relationships, or explore related topics.

    Args:
        expansion_query: Query describing what to add to the graph.
            Examples:
            - "Add narcolepsy disease associations"
            - "Include downstream signaling targets"
            - "Add information about sleep disorders"
        max_articles: Maximum additional articles to fetch. Default 30.

    Returns:
        Summary of the expanded graph.
    """
    orchestrator = get_orchestrator()

    if not orchestrator.has_graph:
        return {
            "status": "error",
            "message": "No graph to expand. Use build_knowledge_graph first.",
        }

    try:
        graph = await orchestrator.expand(
            expansion_query=expansion_query,
            max_articles=max_articles,
        )

        summary = graph.to_summary()

        return {
            "status": "success",
            "message": f"Expanded graph to {summary['node_count']} nodes and {summary['edge_count']} edges",
            "summary": summary,
            "node_count": summary.get("node_count", 0),
            "edge_count": summary.get("edge_count", 0),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to expand graph: {str(e)}",
        }


# =============================================================================
# Q&A Tools (Work with Built Graph)
# =============================================================================

def get_graph_summary(tool_context: ToolContext = None) -> dict:
    """Get summary statistics of the current knowledge graph."""
    orchestrator = get_orchestrator()

    if not orchestrator.has_graph:
        return {
            "status": "error",
            "message": "No knowledge graph built. Use build_knowledge_graph first.",
            "node_count": 0,
            "edge_count": 0,
        }

    return orchestrator.get_graph_summary()


def query_evidence(
    protein1: str,
    protein2: str,
    tool_context: ToolContext = None,
) -> dict:
    """
    Get all evidence for a relationship between two proteins.

    Args:
        protein1: First protein/gene name (e.g., "HCRTR1").
        protein2: Second protein/gene name (e.g., "HCRT").

    Returns:
        All relationships and evidence between the two entities.
    """
    orchestrator = get_orchestrator()
    graph = orchestrator.graph

    if graph is None:
        return {"status": "error", "message": "No graph built."}

    relationships = graph.get_relationship(protein1, protein2)
    reverse_rels = graph.get_relationship(protein2, protein1)

    all_rels = {}
    if relationships:
        all_rels.update(relationships)
    if reverse_rels:
        for k, v in reverse_rels.items():
            if k not in all_rels:
                all_rels[f"{k}_reverse"] = v

    evidence_count = 0
    has_string = False
    has_literature = False
    has_ml = False

    for rel_data in all_rels.values():
        evidence = rel_data.get("evidence", [])
        evidence_count += len(evidence)
        for ev in evidence:
            source_type = ev.get("source_type", "")
            if source_type == EvidenceSource.STRING.value:
                has_string = True
            elif source_type == EvidenceSource.LITERATURE.value:
                has_literature = True
            elif source_type == EvidenceSource.ML_PREDICTED.value:
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


def find_path(
    source: str,
    target: str,
    tool_context: ToolContext = None,
) -> dict:
    """
    Find the shortest path between two entities.

    Traces the relationship chain connecting two proteins, genes, or diseases.

    Args:
        source: Source entity name (e.g., "HCRTR1").
        target: Target entity name (e.g., "narcolepsy").

    Returns:
        Path details including intermediaries and relationship types.
    """
    orchestrator = get_orchestrator()
    graph = orchestrator.graph

    if graph is None:
        return {"status": "error", "message": "No graph built."}

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
    }


def get_protein_neighborhood(
    protein: str,
    max_neighbors: int = 10,
    tool_context: ToolContext = None,
) -> dict:
    """
    Get the neighborhood of a protein in the knowledge graph.

    Args:
        protein: Protein/gene name (e.g., "HCRTR1").
        max_neighbors: Maximum neighbors to return. Default 10.

    Returns:
        List of connected entities with relationship details.
    """
    orchestrator = get_orchestrator()
    graph = orchestrator.graph

    if graph is None:
        return {"status": "error", "message": "No graph built."}

    neighbors = graph.get_neighbors(protein, max_neighbors)

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
        "protein": protein,
        "neighbors": formatted_neighbors,
        "count": len(formatted_neighbors),
    }


def compute_centrality(
    method: str = "pagerank",
    top_k: int = 10,
    tool_context: ToolContext = None,
) -> dict:
    """
    Compute centrality scores to identify important hub proteins.

    Args:
        method: Centrality method - "pagerank", "betweenness", "degree", "closeness".
        top_k: Number of top entities to return. Default 10.

    Returns:
        Ranked list of entities by centrality score.
    """
    orchestrator = get_orchestrator()
    graph = orchestrator.graph

    if graph is None:
        return {"status": "error", "message": "No graph built."}

    result = graph.compute_centrality(method, top_k)
    return {
        "method": result.method,
        "top_entities": [
            {"entity": entity, "score": round(score, 4)}
            for entity, score in result.top_entities
        ],
        "total_nodes": len(result.all_scores),
    }


def detect_communities(tool_context: ToolContext = None) -> dict:
    """
    Detect functional communities/modules in the knowledge graph.

    Uses the Louvain algorithm to identify tightly connected clusters
    that may represent protein complexes or functional modules.

    Returns:
        List of communities with their members and modularity score.
    """
    orchestrator = get_orchestrator()
    graph = orchestrator.graph

    if graph is None:
        return {"status": "error", "message": "No graph built."}

    result = graph.detect_communities()
    return {
        "num_communities": result.num_communities,
        "modularity": round(result.modularity, 4),
        "communities": result.communities,
    }


def predict_novel_links(
    min_ml_score: float = 0.3,
    max_enrichment: int = 50,
    max_novel: int = 50,
    tool_context: ToolContext = None,
) -> dict:
    """
    Combined link enrichment and novel prediction tool.

    This tool evaluates ALL protein pairs not currently in the graph and categorizes
    them into two types:

    1. **Enrichment** (in_string=True): Known STRING interactions that were not in the
       original graph. These are high-confidence edges from the STRING database.

    2. **Novel Predictions** (in_string=False, score >= threshold): ML-predicted
       interactions that are NOT in STRING - truly novel hypotheses.

    Both categories are added to the graph with appropriate metadata for visualization:
    - Enrichment edges: link_category="enrichment" (for green coloring)
    - Novel prediction edges: link_category="novel_prediction" (for orange coloring)

    Args:
        min_ml_score: Minimum ML score for novel predictions (0-1). Default 0.3.
            Enrichment edges are added regardless of score since they have STRING evidence.
        max_enrichment: Maximum enrichment edges to add. Default 50.
        max_novel: Maximum novel predictions to add. Default 50.

    Returns:
        Structured response with enrichment and novel_predictions lists, plus stats.
    """
    orchestrator = get_orchestrator()
    graph = orchestrator.graph

    if graph is None:
        return {"status": "error", "message": "No graph built."}

    # Get all protein entities
    proteins = [
        entity_id
        for entity_id, data in graph.entities.items()
        if data.get("type") in ["protein", "gene"]
    ]

    if len(proteins) < 2:
        return {
            "status": "error",
            "message": "Not enough proteins for prediction",
            "enrichment": [],
            "novel_predictions": [],
            "stats": {
                "total_pairs_evaluated": 0,
                "enrichment_count": 0,
                "novel_prediction_count": 0,
                "proteins_not_in_string": 0,
            },
        }

    # Generate ALL pairs not already in graph (no sampling limit)
    candidate_pairs = []
    for i, p1 in enumerate(proteins):
        for p2 in proteins[i + 1:]:
            existing = graph.get_relationship(p1, p2)
            reverse_existing = graph.get_relationship(p2, p1)
            if not existing and not reverse_existing:
                candidate_pairs.append((p1, p2))

    if not candidate_pairs:
        return {
            "status": "success",
            "message": "All protein pairs already have relationships",
            "enrichment": [],
            "novel_predictions": [],
            "stats": {
                "total_pairs_evaluated": 0,
                "enrichment_count": 0,
                "novel_prediction_count": 0,
                "proteins_not_in_string": 0,
            },
        }

    # Run predictions on ALL candidate pairs
    try:
        predictions = _link_predictor.predict(candidate_pairs)
    except ValueError as e:
        return {
            "status": "error",
            "message": f"Link predictor not available: {e}",
            "enrichment": [],
            "novel_predictions": [],
            "stats": {
                "total_pairs_evaluated": len(candidate_pairs),
                "enrichment_count": 0,
                "novel_prediction_count": 0,
                "proteins_not_in_string": 0,
            },
        }

    # Separate into enrichment vs novel predictions
    enrichment_results = []
    novel_results = []
    proteins_not_in_string = 0

    for pred in predictions:
        if pred.get("error"):
            # Count proteins not found in STRING
            if "Unknown protein" in str(pred.get("error", "")):
                proteins_not_in_string += 1
            continue

        ml_score = pred.get("ml_score", 0.0)
        in_string = pred.get("in_string", False)

        if in_string:
            # Enrichment: Known STRING interaction not in graph
            enrichment_results.append({
                "protein1": pred["protein1"],
                "protein2": pred["protein2"],
                "ml_score": round(ml_score, 4),
                "in_string": True,
            })
        elif ml_score >= min_ml_score:
            # Novel prediction: Not in STRING but ML predicts interaction
            novel_results.append({
                "protein1": pred["protein1"],
                "protein2": pred["protein2"],
                "ml_score": round(ml_score, 4),
                "in_string": False,
            })

    # Sort by ML score (highest first)
    enrichment_results.sort(key=lambda x: x["ml_score"], reverse=True)
    novel_results.sort(key=lambda x: x["ml_score"], reverse=True)

    # Limit results
    top_enrichment = enrichment_results[:max_enrichment]
    top_novel = novel_results[:max_novel]

    # Add enrichment edges to graph
    for pred in top_enrichment:
        graph.add_relationship(
            source=pred["protein1"],
            target=pred["protein2"],
            rel_type=RelationshipType.INTERACTS_WITH,
            ml_score=pred["ml_score"],
            link_category="enrichment",
            evidence=[{
                "source_type": EvidenceSource.STRING.value,
                "source_id": "string_enrichment",
                "confidence": pred["ml_score"],
            }],
        )

    # Add novel predictions to graph
    for pred in top_novel:
        graph.add_relationship(
            source=pred["protein1"],
            target=pred["protein2"],
            rel_type=RelationshipType.HYPOTHESIZED,
            ml_score=pred["ml_score"],
            link_category="novel_prediction",
            evidence=[{
                "source_type": EvidenceSource.ML_PREDICTED.value,
                "source_id": "link_predictor",
                "confidence": pred["ml_score"],
            }],
        )

    return {
        "status": "success",
        "message": f"Added {len(top_enrichment)} enrichment edges and {len(top_novel)} novel predictions",
        "enrichment": top_enrichment,
        "novel_predictions": top_novel,
        "stats": {
            "total_pairs_evaluated": len(candidate_pairs),
            "enrichment_count": len(top_enrichment),
            "novel_prediction_count": len(top_novel),
            "proteins_not_in_string": proteins_not_in_string,
            "total_enrichment_available": len(enrichment_results),
            "total_novel_available": len(novel_results),
        },
    }


def run_diamond_module(
    seed_genes: list[str],
    max_iterations: int = 200,
    tool_context: ToolContext = None,
) -> dict:
    """
    Run DIAMOnD algorithm to detect disease modules from seed genes.

    Based on Ghiassian et al. (2015) for identifying disease-associated
    gene modules through network connectivity analysis.

    Args:
        seed_genes: Known disease-associated genes (e.g., from GWAS).
        max_iterations: Maximum genes to add to module. Default 200.

    Returns:
        Disease module genes and ranked expansion candidates.
    """
    orchestrator = get_orchestrator()
    graph = orchestrator.graph

    if graph is None:
        return {"status": "error", "message": "No graph built."}

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
    }


def calculate_drug_disease_proximity(
    drug_targets: list[str],
    disease_genes: list[str],
    tool_context: ToolContext = None,
) -> dict:
    """
    Calculate network proximity between drug targets and disease genes.

    Based on Guney et al. (2016) for predicting drug efficacy through
    network-based drug-disease relationships.

    Args:
        drug_targets: List of drug target genes/proteins.
        disease_genes: List of disease-associated genes.

    Returns:
        Proximity metrics with statistical significance.
    """
    orchestrator = get_orchestrator()
    graph = orchestrator.graph

    if graph is None:
        return {"status": "error", "message": "No graph built."}

    result = graph.calculate_network_proximity(
        drug_targets, disease_genes, n_random=100
    )

    return {
        "drug_targets": result.drug_targets,
        "disease_genes": result.disease_genes,
        "observed_distance": round(result.observed_distance, 4) if result.observed_distance != float('inf') else None,
        "z_score": round(result.z_score, 4),
        "p_value": round(result.p_value, 4),
        "is_significant": result.is_significant,
        "interpretation": result.interpretation,
    }


def get_capabilities(tool_context: ToolContext = None) -> dict:
    """Get a description of what this agent can do."""
    return {
        "capabilities": [
            "Build knowledge graphs from natural language queries",
            "Expand existing graphs with additional data",
            "Query evidence for specific protein pairs",
            "Find paths between entities",
            "Compute centrality metrics (PageRank, betweenness, etc.)",
            "Detect communities/modules using Louvain algorithm",
            "Predict novel protein-protein interactions using ML",
            "Run DIAMOnD disease module detection",
            "Calculate drug-disease network proximity",
        ],
        "data_sources": [
            "STRING (protein-protein interactions)",
            "PubMed (biomedical literature)",
            "PubTator (named entity recognition)",
        ],
        "architecture": "Hybrid: LLM-driven data fetching + deterministic processing",
    }


# =============================================================================
# Agent Instruction
# =============================================================================

AGENT_INSTRUCTION = """You are a Scientific Knowledge Graph Agent for drug discovery research.

## Your Architecture

You use a hybrid architecture that separates:
1. **Data Fetching** (LLM-driven): Smart query construction for STRING and PubMed
2. **Data Processing** (Deterministic): Efficient relationship extraction and graph building

## Primary Workflow

For new knowledge graph requests:
1. Use `build_knowledge_graph(user_query)` - This handles everything:
   - Extracts seed proteins from the query
   - Fetches STRING protein interactions
   - Constructs optimal PubMed queries
   - Fetches articles and NER annotations
   - Extracts relationships using batched LLM mining
   - Builds the knowledge graph

2. Use `get_graph_summary()` to see what was built

3. Use Q&A tools to explore:
   - `query_evidence(protein1, protein2)` - Get evidence for relationships
   - `find_path(source, target)` - Find connection paths
   - `get_protein_neighborhood(protein)` - See connected entities
   - `compute_centrality(method)` - Find hub proteins
   - `detect_communities()` - Find functional modules

## Expanding the Graph

Use `expand_graph(query)` to add more data:
- "Add narcolepsy disease associations"
- "Include downstream signaling targets"
- "Add sleep disorder pathways"

## Analysis Tools

- `predict_novel_links()` - Combined link enrichment + novel prediction:
  - **Enrichment**: Adds known STRING interactions not in graph (green edges)
  - **Novel Predictions**: ML-predicted interactions not in STRING (orange edges)
  - Returns both categories with stats for visualization
- `run_diamond_module(seed_genes)` - Disease module detection
- `calculate_drug_disease_proximity(drug_targets, disease_genes)` - Drug efficacy prediction

## Example Interaction

User: "Build a knowledge graph for the orexin signaling pathway"

You: I'll build a knowledge graph for the orexin signaling pathway.
[Call build_knowledge_graph("orexin signaling pathway")]

After building, summarize:
- Number of proteins/genes found
- Key relationships identified
- Disease associations
- Suggest follow-up analyses

## Important Notes

- Always start with `build_knowledge_graph()` for new topics
- Use `expand_graph()` to iteratively grow the graph
- Provide scientific context when explaining results
- Distinguish between STRING interactions, literature evidence, and ML predictions"""


# =============================================================================
# Root Agent Definition
# =============================================================================

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="kg_agent",
    description="Scientific Knowledge Graph Agent - builds and analyzes biological knowledge graphs for drug discovery.",
    instruction=AGENT_INSTRUCTION,
    tools=[
        # Graph building
        build_knowledge_graph,
        expand_graph,
        # Q&A and analysis
        get_graph_summary,
        query_evidence,
        find_path,
        get_protein_neighborhood,
        compute_centrality,
        detect_communities,
        predict_novel_links,
        run_diamond_module,
        calculate_drug_disease_proximity,
        get_capabilities,
    ],
)
