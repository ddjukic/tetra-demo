#!/usr/bin/env python3
"""
Test script for Knowledge Graph and LLM extraction components.

This script demonstrates:
1. Creating a KnowledgeGraph with entities and relationships
2. Testing RelationshipExtractor with a real BRCA1 abstract
3. Testing RelationshipInferrer with graph context

Usage:
    uv run python scripts/test_extraction.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env if present
from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def test_knowledge_graph():
    """Test KnowledgeGraph creation and operations."""
    print("=" * 60)
    print("Testing KnowledgeGraph")
    print("=" * 60)

    from models.knowledge_graph import (
        KnowledgeGraph,
        RelationshipType,
        EvidenceSource,
        EntityType,
    )

    # Create a new knowledge graph
    kg = KnowledgeGraph()

    # Add some entities
    kg.add_entity(
        entity_id="BRCA1",
        entity_type=EntityType.GENE.value,
        name="BRCA1 DNA repair associated",
        aliases=["BRCC1", "FANCS"],
        description="DNA repair protein involved in homologous recombination"
    )

    kg.add_entity(
        entity_id="TP53",
        entity_type=EntityType.GENE.value,
        name="Tumor protein p53",
        aliases=["p53", "LFS1"],
        description="Tumor suppressor protein"
    )

    kg.add_entity(
        entity_id="BARD1",
        entity_type=EntityType.GENE.value,
        name="BRCA1-associated RING domain 1",
        aliases=["BARD1"],
        description="Binds to BRCA1 to form E3 ubiquitin ligase"
    )

    kg.add_entity(
        entity_id="RAD51",
        entity_type=EntityType.GENE.value,
        name="RAD51 recombinase",
        aliases=["RECA", "RAD51A"],
        description="Homologous recombination protein"
    )

    kg.add_entity(
        entity_id="ATM",
        entity_type=EntityType.GENE.value,
        name="ATM serine/threonine kinase",
        aliases=["ATA", "TEL1"],
        description="DNA damage response kinase"
    )

    # Add relationships with evidence
    kg.add_relationship(
        source="BRCA1",
        target="BARD1",
        rel_type=RelationshipType.BINDS_TO,
        evidence=[
            {
                "source_type": EvidenceSource.LITERATURE.value,
                "source_id": "PMID:9788440",
                "confidence": 0.95,
                "text_snippet": "BRCA1 and BARD1 form a stable heterodimer"
            },
            {
                "source_type": EvidenceSource.STRING.value,
                "source_id": "STRING:9606.ENSP00000261584-9606.ENSP00000260947",
                "confidence": 0.99,
                "text_snippet": "Physical interaction (STRING score: 0.999)"
            }
        ]
    )

    kg.add_relationship(
        source="ATM",
        target="BRCA1",
        rel_type=RelationshipType.ACTIVATES,
        evidence=[
            {
                "source_type": EvidenceSource.LITERATURE.value,
                "source_id": "PMID:10973485",
                "confidence": 0.88,
                "text_snippet": "ATM phosphorylates BRCA1 at multiple sites in response to DNA damage"
            }
        ]
    )

    kg.add_relationship(
        source="BRCA1",
        target="RAD51",
        rel_type=RelationshipType.REGULATES,
        evidence=[
            {
                "source_type": EvidenceSource.LITERATURE.value,
                "source_id": "PMID:10783165",
                "confidence": 0.82,
                "text_snippet": "BRCA1 regulates RAD51 localization to DNA damage sites"
            }
        ]
    )

    kg.add_relationship(
        source="BRCA1",
        target="TP53",
        rel_type=RelationshipType.INTERACTS_WITH,
        evidence=[
            {
                "source_type": EvidenceSource.LITERATURE.value,
                "source_id": "PMID:9053861",
                "confidence": 0.85,
                "text_snippet": "BRCA1 and p53 interact to regulate gene expression"
            }
        ]
    )

    # Add an ML-predicted relationship without literature support
    kg.add_relationship(
        source="BRCA1",
        target="CHEK2",
        rel_type=RelationshipType.HYPOTHESIZED,
        ml_score=0.87,
        reasoning="High embedding similarity in DNA damage response pathway"
    )

    # Ensure CHEK2 entity exists
    kg.add_entity(
        entity_id="CHEK2",
        entity_type=EntityType.GENE.value,
        name="Checkpoint kinase 2",
        aliases=["CHK2", "RAD53"],
        description="Cell cycle checkpoint kinase"
    )

    # Add another novel prediction
    kg.add_relationship(
        source="RAD51",
        target="PALB2",
        rel_type=RelationshipType.HYPOTHESIZED,
        ml_score=0.92,
        reasoning="Both involved in BRCA1-mediated DNA repair"
    )

    kg.add_entity(
        entity_id="PALB2",
        entity_type=EntityType.GENE.value,
        name="Partner and localizer of BRCA2",
        aliases=["FANCN"],
        description="BRCA2-interacting protein"
    )

    # Test graph operations
    print("\n--- Graph Summary ---")
    summary = kg.to_summary()
    print(json.dumps(summary, indent=2))

    print("\n--- BRCA1 Neighbors ---")
    neighbors = kg.get_neighbors("BRCA1", max_neighbors=5)
    for neighbor_id, data in neighbors:
        direction = data.get("direction", "?")
        rel_type = data.get("relation_type", "unknown")
        evidence_count = len(data.get("evidence", []))
        print(f"  {direction}: {neighbor_id} ({rel_type}, {evidence_count} evidence)")

    print("\n--- Novel Predictions ---")
    novel = kg.get_novel_predictions(min_ml_score=0.8)
    for pred in novel:
        print(f"  {pred['source']} -> {pred['target']}: score={pred['ml_score']:.2f}")

    print("\n--- Entity Interactions Summary ---")
    print(kg.get_entity_interactions_summary("BRCA1"))

    print("\n--- JSON Export (truncated) ---")
    export = kg.to_dict()
    print(f"Entities: {len(export['entities'])}")
    print(f"Relationships: {len(export['relationships'])}")

    print("\n[PASS] KnowledgeGraph tests passed")
    return kg


async def test_relationship_extractor():
    """Test RelationshipExtractor with a real BRCA1 abstract."""
    print("\n" + "=" * 60)
    print("Testing RelationshipExtractor")
    print("=" * 60)

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[SKIP] OPENAI_API_KEY not found in environment")
        print("To test the extractor, set OPENAI_API_KEY in .env file")
        return None

    from extraction.relationship_extractor import RelationshipExtractor

    extractor = RelationshipExtractor(model="gpt-4o-mini")

    # Real BRCA1 abstract from PubMed (PMID: 9788440)
    abstract = """
    BRCA1 is a tumor suppressor gene implicated in breast and ovarian cancer susceptibility.
    Here we report that BRCA1 interacts with BARD1, a protein with RING and BRCT motifs.
    The BRCA1-BARD1 complex functions as a ubiquitin ligase, and this activity is essential
    for DNA repair. We show that ATM kinase phosphorylates BRCA1 in response to DNA damage,
    which promotes the interaction with RAD51 at sites of double-strand breaks.
    Furthermore, BRCA1 regulates the localization of RAD51 to DNA damage foci.
    Loss of BRCA1 function leads to defective homologous recombination and increased
    sensitivity to DNA damaging agents. The p53 tumor suppressor also interacts with
    BRCA1 and together they regulate genes involved in cell cycle arrest and apoptosis.
    """

    entity_pairs = [
        ("BRCA1", "BARD1"),
        ("BRCA1", "ATM"),
        ("BRCA1", "RAD51"),
        ("BRCA1", "p53"),
        ("ATM", "BRCA1"),  # Test directionality
    ]

    print("\n--- Extracting relationships ---")
    print(f"Abstract length: {len(abstract)} chars")
    print(f"Entity pairs: {len(entity_pairs)}")

    relationships = await extractor.extract_relationships(
        abstract=abstract,
        entity_pairs=entity_pairs,
        pmid="9788440"
    )

    print(f"\nExtracted {len(relationships)} relationships:")
    for rel in relationships:
        print(f"\n  {rel.get('entity1')} -> {rel.get('entity2')}")
        print(f"    Relationship: {rel.get('relationship')}")
        print(f"    Confidence: {rel.get('confidence', 0):.2f}")
        evidence = rel.get('evidence_text', '')[:100]
        print(f"    Evidence: {evidence}...")

    print("\n[PASS] RelationshipExtractor tests passed")
    return relationships


async def test_relationship_inferrer(kg):
    """Test RelationshipInferrer with graph context."""
    print("\n" + "=" * 60)
    print("Testing RelationshipInferrer")
    print("=" * 60)

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[SKIP] OPENAI_API_KEY not found in environment")
        print("To test the inferrer, set OPENAI_API_KEY in .env file")
        return None

    from extraction.relationship_inferrer import RelationshipInferrer

    inferrer = RelationshipInferrer(model="gpt-4o-mini")

    print("\n--- Inferring relationship for novel prediction ---")
    print("Prediction: BRCA1 <-> CHEK2 (ML score: 0.87)")

    result = await inferrer.infer_relationship(
        protein_a="BRCA1",
        protein_b="CHEK2",
        ml_score=0.87,
        graph=kg
    )

    print(f"\n  Hypothesized relationship: {result.get('hypothesized_relationship')}")
    print(f"  Confidence: {result.get('confidence')}")
    print(f"  Reasoning: {result.get('reasoning', '')[:200]}...")

    experiments = result.get('validation_experiments', [])
    if experiments:
        print("  Validation experiments:")
        for exp in experiments[:3]:
            print(f"    - {exp}")

    print("\n[PASS] RelationshipInferrer tests passed")
    return result


async def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Scientific Knowledge Graph Agent - Component Tests")
    print("#" * 60)

    # Test 1: Knowledge Graph
    kg = test_knowledge_graph()

    # Test 2: Relationship Extractor
    await test_relationship_extractor()

    # Test 3: Relationship Inferrer (requires graph context)
    await test_relationship_inferrer(kg)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
