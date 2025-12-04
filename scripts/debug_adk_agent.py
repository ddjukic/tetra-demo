#!/usr/bin/env python3
"""
Debug script to trace ADK agent execution step by step.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduce noise from some loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def test_individual_tools():
    """Test each tool individually to find what's hanging."""
    from agent.tools import AgentTools
    from clients.pubmed_client import PubMedClient
    from clients.string_client import STRINGClient
    from extraction import create_batched_miner
    from models.knowledge_graph import KnowledgeGraph

    print("\n" + "="*60)
    print("TESTING INDIVIDUAL TOOLS")
    print("="*60)

    # Initialize dependencies
    kg = KnowledgeGraph()
    pubmed = PubMedClient()
    string_client = STRINGClient()
    miner = create_batched_miner(extractor_name="gemini")

    tools = AgentTools(
        knowledge_graph=kg,
        pubmed_client=pubmed,
        string_client=string_client,
        relationship_miner=miner,
    )

    # Test 1: STRING Network
    print("\n[TEST 1] get_string_network...")
    try:
        result = await tools.get_string_network(["HCRTR1", "HCRTR2", "HCRT"], min_score=700)
        print(f"  ✓ SUCCESS: {result.get('count', 0)} interactions found")
        print(f"  Proteins: {result.get('proteins_found', [])}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return

    # Test 2: PubMed Search
    print("\n[TEST 2] search_literature...")
    try:
        result = await tools.search_literature("orexin signaling", max_results=5)
        print(f"  ✓ SUCCESS: {result.get('count', 0)} articles found")
        articles = result.get('articles', [])
        pmids = [a.get('pmid') for a in articles]
        print(f"  PMIDs: {pmids}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return

    # Test 3: PubTator Annotations
    print("\n[TEST 3] get_entity_annotations...")
    try:
        result = await tools.get_entity_annotations(pmids[:3])  # Just 3 to be quick
        print(f"  ✓ SUCCESS: Got annotations")
        annotations_by_pmid = result.get('annotations_by_pmid', {})
        print(f"  PMIDs with annotations: {list(annotations_by_pmid.keys())}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return

    # Test 4: Extract Relationships (THIS IS LIKELY THE SLOW PART)
    print("\n[TEST 4] extract_relationships (with 2 articles)...")
    try:
        # Just test with 2 articles
        test_articles = articles[:2]
        test_annotations = {pmid: annotations_by_pmid.get(pmid, []) for pmid in pmids[:2]}

        print(f"  Testing with {len(test_articles)} articles...")
        result = await tools.extract_relationships(test_articles, test_annotations)
        print(f"  ✓ SUCCESS: {result.get('count', 0)} relationships extracted")

        # Show sample
        rels = result.get('relationships', [])
        for rel in rels[:3]:
            print(f"    - {rel.get('entity1')} --[{rel.get('relationship')}]--> {rel.get('entity2')}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 5: Build Knowledge Graph
    print("\n[TEST 5] build_knowledge_graph...")
    try:
        # Get STRING interactions from test 1
        string_interactions = result.get('interactions', [])[:3] if 'interactions' in result else []
        lit_relationships = result.get('relationships', [])[:5]

        # Build simple entity dict
        entities = {
            "gene": [{"name": "HCRTR1"}, {"name": "HCRTR2"}, {"name": "HCRT"}]
        }

        result = tools.build_knowledge_graph(
            string_interactions=[],  # Empty for now
            literature_relationships=lit_relationships,
            entities=entities
        )
        print(f"  ✓ SUCCESS: Graph built")
        print(f"  Summary: {result.get('summary', {})}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 6: Get Graph Summary
    print("\n[TEST 6] get_graph_summary...")
    try:
        result = tools.get_graph_summary()
        print(f"  ✓ SUCCESS:")
        print(f"  Nodes: {result.get('node_count', 0)}")
        print(f"  Edges: {result.get('edge_count', 0)}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")

    print("\n" + "="*60)
    print("ALL INDIVIDUAL TOOL TESTS COMPLETE")
    print("="*60)


async def test_adk_agent_simple():
    """Test ADK agent with a very simple query."""
    from agent.tools import AgentTools
    from agent.adk_orchestrator import ADKOrchestrator
    from clients.pubmed_client import PubMedClient
    from clients.string_client import STRINGClient
    from extraction import create_batched_miner
    from models.knowledge_graph import KnowledgeGraph

    print("\n" + "="*60)
    print("TESTING ADK AGENT (simple query)")
    print("="*60)

    # Initialize
    kg = KnowledgeGraph()
    pubmed = PubMedClient()
    string_client = STRINGClient()
    miner = create_batched_miner(extractor_name="gemini")

    tools = AgentTools(
        knowledge_graph=kg,
        pubmed_client=pubmed,
        string_client=string_client,
        relationship_miner=miner,
    )

    orchestrator = ADKOrchestrator(tools=tools, model="gemini-2.5-flash")

    # Simple query - just STRING network
    query = "Get the STRING network for BRCA1 and TP53"

    print(f"\nQuery: {query}")
    print("-" * 60)

    try:
        import time
        start = time.time()
        response = await orchestrator.run(query)
        elapsed = time.time() - start

        print(f"\n[Response in {elapsed:.1f}s]:")
        print(response[:2000] if len(response) > 2000 else response)
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tools", "agent", "both"], default="tools")
    args = parser.parse_args()

    if args.mode in ("tools", "both"):
        await test_individual_tools()

    if args.mode in ("agent", "both"):
        await test_adk_agent_simple()


if __name__ == "__main__":
    asyncio.run(main())
