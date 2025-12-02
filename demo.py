#!/usr/bin/env python3
"""
Scientific Knowledge Graph Agent - Orexin System Demo

This demo script showcases the agent's capabilities by exploring
the orexin/hypocretin signaling system, which is relevant to:
- Sleep/wake regulation
- Narcolepsy
- Appetite and metabolism
- Drug discovery targets

The orexin system involves:
- HCRTR1 (Orexin Receptor 1) - couples to Gq proteins
- HCRTR2 (Orexin Receptor 2) - couples to Gq/Gi proteins
- HCRT (Hypocretin/Orexin precursor) - produces orexin A and B
- Related signaling proteins and downstream effectors

Usage:
    uv run python demo.py
    uv run python demo.py --full  # Run full exploration pipeline
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
from extraction.relationship_extractor import RelationshipExtractor
from extraction.relationship_inferrer import RelationshipInferrer
from agent.tools import AgentTools
from agent.orchestrator import OrchestratorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_quick_demo(agent: OrchestratorAgent) -> None:
    """Run a quick demo showing basic capabilities."""
    print("\n" + "=" * 60)
    print("QUICK DEMO: Scientific Knowledge Graph Agent")
    print("=" * 60)

    # Demo 1: Get capabilities
    print("\n--- Demo 1: Agent Capabilities ---")
    response = await agent.run("What can you do?")
    print(f"\n{response}\n")

    # Demo 2: Simple query about orexin receptors
    print("\n--- Demo 2: Query About Orexin Receptors ---")
    response = await agent.run(
        "Tell me about the orexin receptor HCRTR1. "
        "What are its known interaction partners according to STRING?"
    )
    print(f"\n{response}\n")

    print("\n" + "=" * 60)
    print("Quick demo complete!")
    print("Run with --full flag for complete system exploration.")
    print("=" * 60 + "\n")


async def run_full_demo(agent: OrchestratorAgent) -> None:
    """Run full orexin system exploration demo."""
    print("\n" + "=" * 60)
    print("FULL DEMO: Orexin System Exploration")
    print("=" * 60)
    print("""
The orexin/hypocretin system is crucial for:
- Sleep/wake regulation (disrupted in narcolepsy)
- Appetite and metabolism control
- Reward and addiction pathways
- Potential drug targets for insomnia and narcolepsy

We'll explore this system step by step.
    """)

    # Step 1: Explore the orexin system
    print("\n--- Step 1: Explore Known Orexin Network ---")
    print("Fetching known interactions for orexin receptors from STRING...")
    response = await agent.run(
        "Explore the orexin signaling system. Start with the key proteins: "
        "HCRTR1 (orexin receptor 1), HCRTR2 (orexin receptor 2), and HCRT (orexin precursor). "
        "Get their known interactions from STRING database."
    )
    print(f"\n{response}\n")

    # Step 2: Search literature
    print("\n--- Step 2: Search Recent Literature ---")
    print("Searching PubMed for recent orexin signaling research...")
    response = await agent.run(
        "Search PubMed for recent research on orexin signaling and sleep regulation. "
        "Find articles that mention HCRTR1 or HCRTR2 interactions."
    )
    print(f"\n{response}\n")

    # Step 3: Build knowledge graph
    print("\n--- Step 3: Build Knowledge Graph ---")
    print("Combining STRING and literature data into a knowledge graph...")
    response = await agent.run(
        "Now build a knowledge graph from the STRING interactions and any relationships "
        "you can extract from the literature. Then give me a summary of the graph."
    )
    print(f"\n{response}\n")

    # Step 4: Predict novel links
    print("\n--- Step 4: Predict Novel Interactions ---")
    print("Running ML link predictor to find potential novel interactions...")
    response = await agent.run(
        "Run the ML link predictor to find potential novel protein-protein interactions "
        "in the orexin network. What are the top predictions that aren't already known?"
    )
    print(f"\n{response}\n")

    # Step 5: Generate hypothesis
    print("\n--- Step 5: Generate Research Hypothesis ---")
    print("Generating a testable hypothesis for the top prediction...")
    response = await agent.run(
        "For the most interesting novel prediction, generate a detailed research hypothesis. "
        "Include the biological rationale and suggested validation experiments."
    )
    print(f"\n{response}\n")

    print("\n" + "=" * 60)
    print("Full demo complete!")
    print("=" * 60)
    print("""
Summary of what we did:
1. Explored known orexin protein interactions from STRING
2. Searched PubMed for recent literature
3. Built an evidence-backed knowledge graph
4. Applied ML to predict novel interactions
5. Generated a testable research hypothesis

This demonstrates the agent's ability to:
- Integrate multiple data sources (STRING, PubMed, PubTator)
- Extract relationships from text using LLMs
- Apply ML for link prediction
- Generate scientific hypotheses

Try your own queries with: uv run python main.py
    """)


async def run_stepwise_demo(tools: AgentTools) -> None:
    """Run demo using tools directly (bypasses LLM, useful for testing)."""
    print("\n" + "=" * 60)
    print("STEPWISE DEMO: Direct Tool Execution")
    print("=" * 60)

    # Step 1: Get STRING network
    print("\n--- Step 1: Fetch STRING Network ---")
    seed_proteins = ["HCRTR1", "HCRTR2", "HCRT"]
    print(f"Seed proteins: {seed_proteins}")

    string_result = await tools.get_string_network(seed_proteins, min_score=400)
    print(f"Found {string_result['count']} interactions")
    print(f"Proteins in network: {string_result['proteins_found'][:10]}...")

    # Step 2: Search literature
    print("\n--- Step 2: Search PubMed ---")
    lit_result = await tools.search_literature(
        "orexin receptor signaling HCRTR1 HCRTR2",
        max_results=20
    )
    print(f"Found {lit_result['count']} articles")
    if lit_result['articles']:
        print(f"First article: {lit_result['articles'][0].get('title', 'N/A')[:80]}...")

    # Step 3: Get annotations
    print("\n--- Step 3: Get Entity Annotations ---")
    if lit_result['pmids']:
        annot_result = await tools.get_entity_annotations(lit_result['pmids'][:10])
        print(f"Found {annot_result['count']} annotations")
        print(f"Entity types: {annot_result['entity_types']}")
    else:
        annot_result = {"annotations_by_pmid": {}, "annotations": []}

    # Step 4: Build graph
    print("\n--- Step 4: Build Knowledge Graph ---")
    graph_result = tools.build_knowledge_graph(
        string_interactions=string_result.get('interactions', []),
        literature_relationships=[],  # Would need LLM extraction
        entities={"Gene": annot_result.get('annotations', [])},
    )
    print(f"Graph: {graph_result['node_count']} nodes, {graph_result['edge_count']} edges")

    # Step 5: Predict links
    print("\n--- Step 5: Predict Novel Links ---")
    pred_result = tools.predict_novel_links(min_ml_score=0.5, max_predictions=5)
    print(f"Found {pred_result['count']} predictions")
    for pred in pred_result.get('predictions', [])[:3]:
        print(f"  {pred['protein1']} <-> {pred['protein2']}: {pred['ml_score']:.3f}")

    # Step 6: Get summary
    print("\n--- Step 6: Graph Summary ---")
    summary = tools.get_graph_summary()
    print(f"Summary: {summary}")

    print("\n" + "=" * 60)
    print("Stepwise demo complete!")
    print("=" * 60 + "\n")


async def main() -> None:
    """Main demo entry point."""
    parser = argparse.ArgumentParser(
        description="Scientific Knowledge Graph Agent - Orexin System Demo",
    )
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Run full exploration pipeline (takes longer)",
    )
    parser.add_argument(
        "--stepwise", "-s",
        action="store_true",
        help="Run stepwise demo (bypasses LLM, useful for testing)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load environment
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        sys.exit(1)

    # Load link predictor
    link_predictor_path = Path("models/link_predictor.pkl")
    if link_predictor_path.exists():
        print("Loading link predictor...")
        link_predictor = LinkPredictor.load(str(link_predictor_path))
    else:
        print("Warning: Link predictor not found, creating empty one...")
        link_predictor = LinkPredictor()

    # Initialize components
    print("Initializing components...")
    string_client = StringClient()
    pubmed_client = PubMedClient(api_key=os.getenv("NCBI_API_KEY"))
    relationship_extractor = RelationshipExtractor(model=args.model)
    relationship_inferrer = RelationshipInferrer(model=args.model)

    tools = AgentTools(
        link_predictor=link_predictor,
        string_client=string_client,
        pubmed_client=pubmed_client,
        relationship_extractor=relationship_extractor,
        relationship_inferrer=relationship_inferrer,
    )

    try:
        if args.stepwise:
            # Stepwise demo (direct tool execution)
            await run_stepwise_demo(tools)
        else:
            # LLM-based demo
            agent = OrchestratorAgent(tools=tools, model=args.model)

            if args.full:
                await run_full_demo(agent)
            else:
                await run_quick_demo(agent)
    finally:
        # Cleanup
        await string_client.close()
        await pubmed_client.close()


if __name__ == "__main__":
    asyncio.run(main())
