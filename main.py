#!/usr/bin/env python3
"""
Scientific Knowledge Graph Agent - CLI Entry Point

This is the main entry point for the Scientific Knowledge Graph Agent.
It loads the link predictor model, initializes all clients and extractors,
and starts an interactive chat session using Google ADK (Gemini).

Usage:
    uv run python main.py
    uv run python main.py --interactive
    uv run python main.py --query "What can you do?"

Environment:
    GOOGLE_API_KEY: Required for Gemini LLM operations
    NCBI_API_KEY: Optional for higher PubMed rate limits
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.link_predictor import LinkPredictor
from clients.string_client import StringClient
from clients.pubmed_client import PubMedClient
from extraction import create_batched_miner
from extraction.relationship_inferrer import RelationshipInferrer
from agent.tools import AgentTools
from agent.adk_orchestrator import ADKOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_interactive(agent: ADKOrchestrator) -> None:
    """Run interactive chat session."""
    print("\n=== Scientific Knowledge Graph Agent (Google ADK) ===")
    print("I can help you explore biological systems and generate hypotheses.")
    print("Try: 'What can you do?' or 'Explore the orexin signaling system'")
    print("Type 'exit', 'quit', or Ctrl+C to quit.\n")

    while True:
        try:
            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if query.lower() == "clear":
                agent.clear_session()
                print("Conversation session cleared.\n")
                continue

            print("\nThinking...")
            response = await agent.run(query)
            print(f"\nAgent: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


async def run_single_query(agent: ADKOrchestrator, query: str) -> None:
    """Run a single query and print the response."""
    print(f"\nQuery: {query}")
    print("\nProcessing...")
    response = await agent.run(query)
    print(f"\n{response}\n")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scientific Knowledge Graph Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python main.py
    uv run python main.py --query "What can you do?"
    uv run python main.py --query "Explore the BRCA1 interaction network"
    uv run python main.py --model gemini-2.5-pro
        """,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query and exit",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        default=True,
        help="Run in interactive mode (default)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--link-predictor", "-l",
        type=str,
        default="models/pyg_link_predictor.pkl",
        help="Path to link predictor model (default: models/pyg_link_predictor.pkl)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Load environment variables
    load_dotenv()

    # Check for Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment.")
        sys.exit(1)

    # Load link predictor
    link_predictor_path = Path(args.link_predictor)
    if link_predictor_path.exists():
        print(f"Loading link predictor from {link_predictor_path}...")
        try:
            link_predictor = LinkPredictor.load(str(link_predictor_path))
            print(f"Loaded predictor with {len(link_predictor.gene_to_string_id)} gene mappings")
        except Exception as e:
            print(f"Warning: Failed to load link predictor: {e}")
            print("Creating empty link predictor...")
            link_predictor = LinkPredictor()
    else:
        print(f"Warning: Link predictor not found at {link_predictor_path}")
        print("Creating empty link predictor (ML predictions will be limited)...")
        link_predictor = LinkPredictor()

    # Initialize clients
    print("Initializing API clients...")
    ncbi_api_key = os.getenv("NCBI_API_KEY")
    string_client = StringClient()
    pubmed_client = PubMedClient(api_key=ncbi_api_key)

    # Initialize extractors - use fast batched miner (gemini default)
    print("Initializing extractors...")
    relationship_miner = create_batched_miner(extractor_name="gemini")
    relationship_inferrer = RelationshipInferrer()

    # Initialize tools
    print("Initializing agent tools...")
    tools = AgentTools(
        link_predictor=link_predictor,
        string_client=string_client,
        pubmed_client=pubmed_client,
        relationship_miner=relationship_miner,
        relationship_inferrer=relationship_inferrer,
    )

    # Initialize ADK orchestrator with Gemini
    print(f"Initializing ADK orchestrator with {args.model}...")
    agent = ADKOrchestrator(tools=tools, model=args.model)

    try:
        if args.query:
            # Single query mode
            await run_single_query(agent, args.query)
        else:
            # Interactive mode
            await run_interactive(agent)
    finally:
        # Cleanup
        print("\nClosing connections...")
        await string_client.close()
        await pubmed_client.close()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
