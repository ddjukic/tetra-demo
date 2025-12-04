"""
Orchestrator Agent using Google ADK (Agent Development Kit).

Uses Gemini with LlmAgent and tool functions for:
- Building knowledge graphs from multiple data sources
- ML-based link prediction for novel interactions
- Literature search and relationship extraction
- Hypothesis generation and validation suggestions
"""

import asyncio
import logging
from typing import Any, Optional

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from agent.tools import AgentTools

logger = logging.getLogger(__name__)


# Module-level state for tool access
_tools_instance: Optional[AgentTools] = None


def set_tools(tools: AgentTools) -> None:
    """Set the global tools instance for agent functions."""
    global _tools_instance
    _tools_instance = tools


def get_tools() -> AgentTools:
    """Get the global tools instance."""
    if _tools_instance is None:
        raise RuntimeError("Tools not initialized. Call set_tools() first.")
    return _tools_instance


# =============================================================================
# Tool Functions for ADK Agent
# =============================================================================

async def get_string_network(seed_proteins: list[str], min_score: int = 700) -> dict:
    """
    Fetch known protein interaction network from STRING database for seed proteins.

    Args:
        seed_proteins: List of protein/gene names (e.g., ['HCRTR1', 'HCRTR2', 'PPARG'])
        min_score: Minimum STRING confidence score (0-1000), default 700 for high confidence

    Returns:
        Dictionary with interactions, proteins_found, and count.
    """
    tools = get_tools()
    return await tools.get_string_network(seed_proteins, min_score)


async def get_string_partners(proteins: list[str], limit: int = 30) -> dict:
    """
    Get interaction partners from STRING for proteins (expands the network).

    Args:
        proteins: List of protein names to find partners for
        limit: Maximum partners per protein

    Returns:
        Dictionary with partners and count.
    """
    tools = get_tools()
    return await tools.get_string_partners(proteins, limit)


async def search_literature(query: str, max_results: int = 50) -> dict:
    """
    Search PubMed for relevant biomedical literature.

    Args:
        query: PubMed search query (supports Boolean operators)
        max_results: Maximum number of results

    Returns:
        Dictionary with articles, pmids, and count.
    """
    tools = get_tools()
    return await tools.search_literature(query, max_results)


async def get_entity_annotations(pmids: list[str]) -> dict:
    """
    Get NER annotations (genes, diseases, chemicals) from PubTator for articles.

    Args:
        pmids: List of PubMed IDs to annotate

    Returns:
        Dictionary with annotations and annotations_by_pmid.
    """
    tools = get_tools()
    return await tools.get_entity_annotations(pmids)


async def extract_relationships(articles: list[dict], annotations_by_pmid: dict) -> dict:
    """
    Extract typed relationships between entities from abstracts using LLM.

    Args:
        articles: List of article objects with pmid and abstract
        annotations_by_pmid: Dict mapping PMID to list of annotations

    Returns:
        Dictionary with relationships and count.
    """
    tools = get_tools()
    return await tools.extract_relationships(articles, annotations_by_pmid)


def build_knowledge_graph(
    string_interactions: list[dict],
    literature_relationships: list[dict],
    entities: dict,
) -> dict:
    """
    Build an evidence-backed knowledge graph from STRING interactions and literature relationships.

    Args:
        string_interactions: Interactions from STRING
        literature_relationships: Extracted relationships from literature
        entities: Dict mapping entity type to list of entity dicts

    Returns:
        Dictionary with summary and statistics.
    """
    tools = get_tools()
    return tools.build_knowledge_graph(
        string_interactions, literature_relationships, entities
    )


def predict_novel_links(min_ml_score: float = 0.7, max_predictions: int = 20) -> dict:
    """
    Apply ML link predictor to find novel protein-protein interactions.

    Args:
        min_ml_score: Minimum ML prediction score (0-1)
        max_predictions: Maximum number of predictions to return

    Returns:
        Dictionary with predictions and count.
    """
    tools = get_tools()
    return tools.predict_novel_links(min_ml_score, max_predictions)


async def infer_novel_relationships(predictions: list[dict], max_inferences: int = 5) -> dict:
    """
    Use LLM to infer relationship types for top ML predictions.

    Args:
        predictions: List of predictions from predict_novel_links
        max_inferences: Maximum number of inferences to run

    Returns:
        Dictionary with inferences and count.
    """
    tools = get_tools()
    return await tools.infer_novel_relationships(predictions, max_inferences)


def query_evidence(protein1: str, protein2: str) -> dict:
    """
    Get all evidence for a relationship between two proteins.

    Args:
        protein1: First protein identifier
        protein2: Second protein identifier

    Returns:
        Dictionary with relationships and evidence counts.
    """
    tools = get_tools()
    return tools.query_evidence(protein1, protein2)


def get_graph_summary() -> dict:
    """
    Get summary statistics of the current knowledge graph.

    Returns:
        Dictionary with node_count, edge_count, and other statistics.
    """
    tools = get_tools()
    return tools.get_graph_summary()


def get_protein_neighborhood(protein: str, max_neighbors: int = 10) -> dict:
    """
    Get the neighborhood (connected proteins) of a specific protein.

    Args:
        protein: Protein identifier
        max_neighbors: Maximum neighbors to return

    Returns:
        Dictionary with neighbors and count.
    """
    tools = get_tools()
    return tools.get_protein_neighborhood(protein, max_neighbors)


async def generate_hypothesis(protein1: str, protein2: str) -> dict:
    """
    Generate a detailed testable hypothesis for a predicted protein interaction.

    Args:
        protein1: First protein
        protein2: Second protein

    Returns:
        Structured hypothesis dictionary.
    """
    tools = get_tools()
    return await tools.generate_hypothesis(protein1, protein2)


def get_capabilities() -> dict:
    """
    Get a description of what this agent can do.

    Returns:
        Dictionary describing capabilities, data sources, and models.
    """
    tools = get_tools()
    return tools.get_capabilities()


# =============================================================================
# Agent Definition
# =============================================================================

AGENT_INSTRUCTION = """You are a Scientific Knowledge Graph Agent that helps drug discovery scientists
explore biological systems and generate hypotheses for novel protein interactions.

You have access to the following tools:
1. get_string_network(seed_proteins, min_score) - Get known interactions from STRING database
2. get_string_partners(proteins, limit) - Get interaction partners from STRING
3. search_literature(query, max_results) - Search PubMed for relevant literature
4. get_entity_annotations(pmids) - Get NER annotations (genes, diseases, chemicals) from PubTator
5. extract_relationships(articles, annotations_by_pmid) - Extract typed relationships using LLM
6. build_knowledge_graph(string_interactions, literature_relationships, entities) - Build knowledge graph
7. predict_novel_links(min_ml_score, max_predictions) - Run ML link predictor for novel interactions
8. infer_novel_relationships(predictions, max_inferences) - Infer relationship types for predictions
9. query_evidence(protein1, protein2) - Get all evidence for a relationship
10. get_graph_summary() - Get current graph statistics
11. get_protein_neighborhood(protein, max_neighbors) - Get neighborhood of a protein
12. generate_hypothesis(protein1, protein2) - Generate testable hypothesis for an interaction
13. get_capabilities() - Get description of what the agent can do

When a user asks to explore a biological system:
1. Identify seed proteins from the query (e.g., for "orexin system" use HCRTR1, HCRTR2)
2. Get STRING network for seed proteins to find known interactions
3. Search PubMed for relevant literature about the proteins
4. Get NER annotations from PubTator
5. Extract relationships from the literature
6. Build the knowledge graph combining STRING and literature data
7. Run link prediction to find novel interaction candidates
8. Infer relationship types for top predictions
9. Summarize findings with evidence

When asked "What can you do?" or similar, use get_capabilities() to describe your abilities.

Always explain what you're doing step by step and present findings clearly with evidence.
Distinguish between known interactions (from STRING/literature) and ML predictions.
When presenting novel predictions, include ML scores and suggest validation experiments."""


def create_kg_agent(tools: AgentTools, model: str = "gemini-2.5-flash") -> LlmAgent:
    """
    Create the Knowledge Graph Agent with all tools.

    Args:
        tools: AgentTools instance with all dependencies
        model: Gemini model to use (default: gemini-2.5-flash)

    Returns:
        Configured LlmAgent instance
    """
    # Set global tools for function access
    set_tools(tools)

    # Create the agent
    agent = LlmAgent(
        model=model,
        name="scientific_knowledge_graph_agent",
        description="An agent that builds and analyzes biological knowledge graphs for drug discovery, using STRING database, PubMed literature, and ML link prediction.",
        instruction=AGENT_INSTRUCTION,
        tools=[
            get_string_network,
            get_string_partners,
            search_literature,
            get_entity_annotations,
            extract_relationships,
            build_knowledge_graph,
            predict_novel_links,
            infer_novel_relationships,
            query_evidence,
            get_graph_summary,
            get_protein_neighborhood,
            generate_hypothesis,
            get_capabilities,
        ],
    )

    return agent


# For ADK CLI discovery
root_agent = None  # Will be set after tools are initialized


class ADKOrchestrator:
    """
    High-level wrapper for the ADK-based agent.

    Provides a simple interface for running queries and managing sessions.
    """

    def __init__(
        self,
        tools: AgentTools,
        model: str = "gemini-2.5-flash",
        app_name: str = "tetra_kg_agent",
    ):
        """
        Initialize the ADK orchestrator.

        Args:
            tools: AgentTools instance with all dependencies
            model: Gemini model to use
            app_name: Application name for session management
        """
        self.tools = tools
        self.model = model
        self.app_name = app_name

        # Create agent
        self.agent = create_kg_agent(tools, model)

        # Session management
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name=app_name,
            session_service=self.session_service,
        )

        self.current_user_id = "default_user"
        self.current_session_id = "default_session"
        self._session_created = False

    async def _ensure_session(self, user_id: str, session_id: str) -> None:
        """Ensure session exists, create if needed."""
        if not self._session_created:
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
            )
            self._session_created = True
            logger.debug(f"Created session: app={self.app_name}, user={user_id}, session={session_id}")

    async def run(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Process user query through the ADK agent.

        Args:
            user_query: The user's natural language query
            user_id: Optional user ID for session
            session_id: Optional session ID

        Returns:
            Agent's final response as a string
        """
        from google.genai import types

        user_id = user_id or self.current_user_id
        session_id = session_id or self.current_session_id

        try:
            # Ensure session exists
            await self._ensure_session(user_id, session_id)
            # Prepare the user's message in ADK format
            content = types.Content(role='user', parts=[types.Part(text=user_query)])

            final_response_text = "Agent did not produce a final response."
            accumulated_text = []  # Accumulate text from all events

            # run_async returns an async generator - iterate over events
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
            ):
                # Log intermediate events for debugging
                logger.debug(f"Event: author={event.author}, final={event.is_final_response()}")

                # Accumulate text from events (not just final)
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text and part.text.strip():
                            accumulated_text.append(part.text)

                # Check for final response
                if event.is_final_response():
                    logger.debug(f"Final event: content={event.content}, actions={event.actions}")
                    if accumulated_text:
                        # Use accumulated text from all events
                        final_response_text = "\n".join(accumulated_text)
                        logger.debug(f"Accumulated {len(accumulated_text)} text parts")
                    elif event.actions and event.actions.escalate:
                        final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                    else:
                        logger.warning(f"Final event has no accumulated text")
                    break

            return final_response_text

        except Exception as e:
            logger.error(f"Error in ADK agent: {e}")
            return f"I encountered an error: {str(e)}"

    async def chat(
        self,
        user_message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Alias for run() for backwards compatibility."""
        return await self.run(user_message, user_id, session_id)

    def clear_session(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Clear session history."""
        user_id = user_id or self.current_user_id
        session_id = session_id or self.current_session_id
        # Create new session
        self.current_session_id = f"{session_id}_{hash(user_id)}"
