"""
Orchestrator Agent for the Scientific Knowledge Graph Agent.

Uses OpenAI's function calling to orchestrate tool usage for:
- Building knowledge graphs from multiple data sources
- ML-based link prediction for novel interactions
- Literature search and relationship extraction
- Hypothesis generation and validation suggestions
"""

import json
import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from agent.tools import AgentTools

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = '''You are a Scientific Knowledge Graph Agent that helps drug discovery scientists
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
When presenting novel predictions, include ML scores and suggest validation experiments.'''


class OrchestratorAgent:
    """
    Main agent that interprets queries and orchestrates tools.

    Uses OpenAI function calling to intelligently select and
    execute tools based on user queries about biological systems.
    """

    def __init__(
        self,
        tools: AgentTools,
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize the orchestrator agent.

        Args:
            tools: AgentTools instance for executing tools
            model: OpenAI model to use for orchestration
        """
        self.tools = tools
        self.model = model
        self.client = AsyncOpenAI()
        self.conversation_history: list[dict[str, Any]] = []

        # Get tool definitions in OpenAI function calling format
        self.functions = self.tools.get_tool_definitions()

    async def run(
        self,
        user_query: str,
        max_iterations: int = 10,
    ) -> str:
        """
        Process user query through the agent loop.

        Implements the function calling loop:
        1. Send query to LLM with tool definitions
        2. If LLM requests tool calls, execute them
        3. Send tool results back to LLM
        4. Repeat until LLM returns a final response

        Args:
            user_query: The user's natural language query
            max_iterations: Maximum tool call iterations to prevent infinite loops

        Returns:
            Agent's final response as a string
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_query,
        })

        # Build messages for the API call
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.conversation_history,
        ]

        for iteration in range(max_iterations):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.functions if self.functions else None,
                    tool_choice="auto" if self.functions else None,
                    temperature=0.3,
                )

                message = response.choices[0].message

                # Check if the model wants to call tools
                if message.tool_calls:
                    logger.info(f"Iteration {iteration + 1}: {len(message.tool_calls)} tool calls")

                    # Add assistant message with tool calls to context
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    })

                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        try:
                            args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                        logger.info(f"Executing tool: {func_name}")
                        logger.debug(f"Tool args: {args}")

                        # Execute the tool
                        result = await self.tools.execute_tool(func_name, args)

                        # Truncate large results to avoid token limits
                        result_str = json.dumps(result, default=str)
                        if len(result_str) > 15000:
                            result_str = result_str[:15000] + "...[truncated]"

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        })

                else:
                    # No tool calls, model is ready to respond
                    response_text = message.content or "I couldn't generate a response."

                    # Add to conversation history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response_text,
                    })

                    return response_text

            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                error_response = f"I encountered an error: {str(e)}"
                self.conversation_history.append({
                    "role": "assistant",
                    "content": error_response,
                })
                return error_response

        # Max iterations reached
        fallback = (
            "I reached the maximum number of tool calls. "
            "This query may be too complex. Please try a simpler request."
        )
        self.conversation_history.append({
            "role": "assistant",
            "content": fallback,
        })
        return fallback

    # Alias for backwards compatibility
    async def chat(
        self,
        user_message: str,
        max_iterations: int = 10,
    ) -> str:
        """Alias for run() for backwards compatibility."""
        return await self.run(user_message, max_iterations)

    async def run_stream(
        self,
        user_query: str,
        max_iterations: int = 10,
    ) -> AsyncIterator[str]:
        """
        Process user query with streaming response.

        Yields response chunks as they become available.

        Args:
            user_query: The user's query
            max_iterations: Maximum tool call iterations

        Yields:
            Response text chunks
        """
        # For now, use non-streaming and yield full result
        # A full implementation would use streaming API
        response = await self.run(user_query, max_iterations)
        yield response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def get_history(self) -> list[dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history.copy()


class SimpleOrchestratorAgent:
    """
    Simplified orchestrator that doesn't use function calling.

    Useful for quick queries when the full agent loop isn't needed.
    """

    def __init__(
        self,
        tools: AgentTools,
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize the simple orchestrator.

        Args:
            tools: AgentTools instance
            model: OpenAI model to use
        """
        self.tools = tools
        self.model = model
        self.client = AsyncOpenAI()

    async def quick_query(
        self,
        query: str,
        context: str = "",
    ) -> str:
        """
        Quick query without tool calling.

        Uses graph context directly in the prompt.

        Args:
            query: User query
            context: Additional context to include

        Returns:
            Response string
        """
        # Get graph summary for context
        summary = self.tools.get_graph_summary()

        prompt = f"""You are a knowledge graph assistant for drug discovery research.

Current graph state:
- Nodes: {summary.get('node_count', 0)}
- Edges: {summary.get('edge_count', 0)}
- Entity types: {summary.get('entity_types', {})}
- Relationship types: {summary.get('relationship_types', {})}
- ML predictions: {summary.get('ml_predicted_edges', 0)}

{context}

User query: {query}

Provide a helpful response based on the available information."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful biomedical knowledge assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            return response.choices[0].message.content or "No response generated."

        except Exception as e:
            logger.error(f"Quick query error: {e}")
            return f"Error: {str(e)}"
