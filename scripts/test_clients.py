#!/usr/bin/env python3
"""Test script for STRING and PubMed API clients.

Run with: uv run python scripts/test_clients.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.string_client import StringClient
from clients.pubmed_client import PubMedClient


async def test_string_client() -> None:
    """Test STRING API client."""
    print("=" * 60)
    print("Testing STRING API Client")
    print("=" * 60)

    async with StringClient() as client:
        # Test 1: Resolve protein identifiers
        print("\n1. Resolving protein identifiers...")
        proteins = ["BRCA1", "TP53", "EGFR"]
        resolved = await client.resolve_identifiers(proteins)
        print(f"   Resolved {len(resolved)} protein(s)")
        for r in resolved[:3]:
            print(f"   - {r.get('queryItem')} -> {r.get('preferredName')} ({r.get('stringId')})")

        # Test 2: Get interaction network
        print("\n2. Getting interaction network for BRCA1, TP53...")
        interactions = await client.get_network(["BRCA1", "TP53"])
        print(f"   Found {len(interactions)} interaction(s)")
        for i in interactions[:3]:
            print(f"   - {i.get('preferredName_A')} <-> {i.get('preferredName_B')}: score={i.get('score')}")

        # Test 3: Get interaction partners
        print("\n3. Getting interaction partners for BRCA1...")
        partners = await client.get_interaction_partners(["BRCA1"], limit=5)
        print(f"   Found {len(partners)} partner interaction(s)")
        for p in partners[:5]:
            print(f"   - {p.get('preferredName_A')} <-> {p.get('preferredName_B')}: score={p.get('score')}")

        # Test 4: Get functional annotations
        print("\n4. Getting functional annotations for BRCA1, TP53...")
        annotations = await client.get_functional_annotation(["BRCA1", "TP53"])
        print(f"   Found {len(annotations)} annotation(s)")
        for a in annotations[:5]:
            print(f"   - [{a.get('category')}] {a.get('description')}")

        # Test 5: Get enrichment analysis
        print("\n5. Getting enrichment analysis for BRCA1, TP53, EGFR...")
        enrichment = await client.get_enrichment(["BRCA1", "TP53", "EGFR"])
        print(f"   Found {len(enrichment)} enrichment term(s)")
        for e in enrichment[:5]:
            print(f"   - [{e.get('category')}] {e.get('description')} (p={e.get('p_value', 'N/A')})")

    print("\nSTRING client tests completed!")


async def test_pubmed_client() -> None:
    """Test PubMed API client."""
    print("\n" + "=" * 60)
    print("Testing PubMed API Client")
    print("=" * 60)

    async with PubMedClient() as client:
        # Test 1: Search PubMed
        print("\n1. Searching PubMed for 'BRCA1 breast cancer'...")
        pmids = await client.search("BRCA1 breast cancer", max_results=5)
        print(f"   Found {len(pmids)} article(s)")
        print(f"   PMIDs: {pmids}")

        if pmids:
            # Test 2: Fetch abstracts
            print("\n2. Fetching article abstracts...")
            articles = await client.fetch_abstracts(pmids)
            print(f"   Retrieved {len(articles)} article(s)")
            for a in articles[:3]:
                title = a.get("title", "")[:60] + "..." if len(a.get("title", "")) > 60 else a.get("title", "")
                print(f"   - [{a.get('year')}] {title}")
                print(f"     Journal: {a.get('journal', 'N/A')}")
                print(f"     Authors: {', '.join(a.get('authors', [])[:3])}...")

            # Test 3: Get PubTator annotations
            print("\n3. Getting PubTator annotations...")
            annotations = await client.get_pubtator_annotations(pmids[:3])
            print(f"   Found {len(annotations)} annotation(s)")

            # Group by type
            by_type: dict[str, list] = {}
            for ann in annotations:
                t = ann.get("entity_type", "Unknown")
                if t not in by_type:
                    by_type[t] = []
                by_type[t].append(ann)

            for entity_type, anns in by_type.items():
                unique_entities = set(a.get("entity_text") for a in anns)
                print(f"   - {entity_type}: {len(unique_entities)} unique entities")
                examples = list(unique_entities)[:5]
                print(f"     Examples: {', '.join(examples)}")

        # Test 4: Search by gene
        print("\n4. Searching by gene (TP53)...")
        tp53_pmids = await client.search_by_gene("TP53", max_results=3)
        print(f"   Found {len(tp53_pmids)} article(s) for TP53")

        # Test 5: Search by disease
        print("\n5. Searching by disease (lung cancer)...")
        disease_pmids = await client.search_by_disease("lung cancer", max_results=3)
        print(f"   Found {len(disease_pmids)} article(s) for lung cancer")

    print("\nPubMed client tests completed!")


async def main() -> None:
    """Run all client tests."""
    print("Testing Scientific Knowledge Graph API Clients")
    print("=" * 60)

    try:
        await test_string_client()
        await test_pubmed_client()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
