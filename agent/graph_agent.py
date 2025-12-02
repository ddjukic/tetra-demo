"""
GraphRAG Q&A Agent with ReAct Planning.

This module provides a conversational agent for querying and analyzing
biomedical knowledge graphs using Google ADK with ReAct (Reason + Act) planning.

Features:
- Natural language queries about the knowledge graph
- Graph Data Science algorithms (centrality, communities, paths)
- Drug discovery algorithms (DIAMOnD, network proximity, synergy, robustness)
- ML link predictions with hypothesis generation
- Multi-step ReAct planning for complex queries

Architecture:
- Module-level graph cache for fast tool access
- ToolContext for session state management
- @requires_graph decorator ensures graph is loaded before tool execution
- PlanReActPlanner for structured reasoning: Thought -> Action -> Observation -> Repeat
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Optional

from google.adk.agents import LlmAgent
from google.adk.planners import PlanReActPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from models.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


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
    min_score: float = 0.7
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
def generate_hypothesis(
    tool_context: ToolContext,
    protein1: str,
    protein2: str
) -> dict[str, Any]:
    """
    Generate a testable hypothesis for a predicted interaction.

    Creates a structured hypothesis for a potential protein-protein
    interaction, including supporting context from the network.

    Args:
        tool_context: ADK tool context (injected by framework)
        protein1: First protein in the predicted interaction
        protein2: Second protein in the predicted interaction

    Returns:
        Dictionary with:
        - hypothesis: Structured hypothesis statement
        - supporting_evidence: Network context supporting the prediction
        - suggested_experiments: Validation experiments to test the hypothesis
        - confidence_factors: Factors contributing to prediction confidence
    """
    graph = _get_graph(tool_context)

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

    hypothesis = {
        "protein1": protein1,
        "protein2": protein2,
        "hypothesis_statement": f"{protein1} and {protein2} may functionally interact, potentially through shared pathway components or protein complexes.",
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
            "shared_neighbors_count": len(shared_neighbors),
            "path_distance": path_result.distance if path_result.distance > 0 else "disconnected",
            "has_supporting_evidence": existing_rel is not None
        }
    }

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

4. **ML Predictions**:
   - Get novel link predictions (get_predictions)
   - Generate hypotheses for predictions (generate_hypothesis)

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
- Always explain the biological significance of your findings.
- Distinguish between experimentally validated relationships and ML predictions.
- Suggest follow-up analyses when relevant.
- Be precise about statistical significance and limitations.

## Example Queries and Approaches

"What proteins are most important in this network?"
-> Use compute_centrality with pagerank, then detect_communities to see if they cluster.

"Is there a connection between BRCA1 and TP53?"
-> Use query_evidence first, then find_path if no direct relationship.

"Would targeting both EGFR and KRAS be synergistic for cancer?"
-> Use predict_synergy with relevant disease genes, analyze_robustness for each target.

"Find disease module genes starting from known Alzheimer's genes."
-> Use run_diamond with the known genes as seeds, then compute_centrality on the module.

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
        description="Expert biomedical knowledge graph analyst for drug discovery research",
        instruction=GRAPH_AGENT_INSTRUCTION,
        planner=planner,
        tools=[
            get_graph_summary,
            query_evidence,
            find_path,
            compute_centrality,
            detect_communities,
            run_diamond,
            calculate_proximity,
            predict_synergy,
            analyze_robustness,
            get_predictions,
            generate_hypothesis,
            get_entity_neighborhood,
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
        model: str = "gemini-2.5-flash"
    ):
        """
        Initialize the GraphRAG agent manager.

        Args:
            graph: Optional KnowledgeGraph to use. If None, graph must be
                   set later using set_graph() or set_active_graph().
            model: Gemini model to use (default: gemini-2.5-flash)
        """
        if graph:
            set_active_graph(graph)

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

__all__ = [
    "set_active_graph",
    "get_active_graph",
    "clear_graph_cache",
    "create_graph_agent",
    "GraphAgentManager",
    "GRAPH_AGENT_INSTRUCTION",
]
