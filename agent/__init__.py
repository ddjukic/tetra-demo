"""
Agent module for the Scientific Knowledge Graph Agent.

Provides:
- AgentTools: Collection of tools for graph queries, ML predictions, and hypothesis generation
- DataFetchAgent: LLM-driven data collection agent for hybrid architecture
- OrchestratorAgent: OpenAI-based agent with function calling (legacy, requires openai)
- ADKOrchestrator: Google ADK-based agent with Gemini (recommended)
- create_kg_agent: Factory function for creating ADK orchestrator agent
- GraphAgentManager: Conversational GraphRAG Q&A agent with ReAct planning
- create_graph_agent: Factory function for creating GraphRAG query agent

Note: Uses lazy imports to avoid requiring all dependencies at package load time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Eager imports - always available
from agent.tools import AgentTools
from agent.graph_agent import (
    GraphAgentManager,
    create_graph_agent,
    set_active_graph,
    get_active_graph,
    clear_graph_cache,
    GRAPH_AGENT_INSTRUCTION,
)

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    from agent.data_fetch_agent import DataFetchAgent, create_data_fetch_agent
    from agent.orchestrator import OrchestratorAgent, SimpleOrchestratorAgent
    from agent.adk_orchestrator import ADKOrchestrator, create_kg_agent


def __getattr__(name: str):
    """Lazy import for optional modules."""
    if name in ("DataFetchAgent", "create_data_fetch_agent"):
        from agent.data_fetch_agent import DataFetchAgent, create_data_fetch_agent
        return {"DataFetchAgent": DataFetchAgent, "create_data_fetch_agent": create_data_fetch_agent}[name]
    elif name in ("OrchestratorAgent", "SimpleOrchestratorAgent"):
        from agent.orchestrator import OrchestratorAgent, SimpleOrchestratorAgent
        return {"OrchestratorAgent": OrchestratorAgent, "SimpleOrchestratorAgent": SimpleOrchestratorAgent}[name]
    elif name in ("ADKOrchestrator", "create_kg_agent"):
        from agent.adk_orchestrator import ADKOrchestrator, create_kg_agent
        return {"ADKOrchestrator": ADKOrchestrator, "create_kg_agent": create_kg_agent}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Tools
    "AgentTools",
    # Data Fetch Agent (hybrid architecture)
    "DataFetchAgent",
    "create_data_fetch_agent",
    # Legacy OpenAI agent (lazy loaded)
    "OrchestratorAgent",
    "SimpleOrchestratorAgent",
    # ADK Orchestrator (graph building, lazy loaded)
    "ADKOrchestrator",
    "create_kg_agent",
    # GraphRAG Q&A Agent (conversational queries)
    "GraphAgentManager",
    "create_graph_agent",
    "set_active_graph",
    "get_active_graph",
    "clear_graph_cache",
    "GRAPH_AGENT_INSTRUCTION",
]
