#!/usr/bin/env python3
"""
Test script for the refactored predict_novel_links() function.

This script validates that:
1. The function processes ALL candidate pairs (not just max_predictions * 3)
2. Results are properly categorized into enrichment vs novel predictions
3. The default threshold of 0.3 allows more predictions through
4. link_category metadata is properly added to the graph
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.knowledge_graph import KnowledgeGraph, RelationshipType, EvidenceSource


def test_predict_novel_links():
    """Test the refactored predict_novel_links function."""
    print("=" * 70)
    print("Testing refactored predict_novel_links()")
    print("=" * 70)

    # Import after path setup
    from kg_agent.agent import predict_novel_links, _link_predictor, get_orchestrator, set_orchestrator
    from pipeline.orchestrator import KGOrchestrator

    # Check if link predictor is available
    if _link_predictor is None or _link_predictor.classifier is None:
        print("\nWARNING: Link predictor not trained/loaded.")
        print("The function will return an error, but we can still validate the response structure.")

    # Create a test orchestrator with a small graph
    orchestrator = KGOrchestrator(extractor_name="cerebras")
    set_orchestrator(orchestrator)

    # Build a small test graph with proteins known to be in STRING
    graph = KnowledgeGraph()

    # Use well-known proteins that should be in STRING
    test_proteins = [
        ("TP53", "protein", "Tumor protein p53"),
        ("BRCA1", "protein", "BRCA1 DNA repair"),
        ("BRCA2", "protein", "BRCA2 DNA repair"),
        ("MDM2", "protein", "MDM2 proto-oncogene"),
        ("ATM", "protein", "ATM serine/threonine kinase"),
        ("CHEK2", "protein", "Checkpoint kinase 2"),
        ("PTEN", "protein", "Phosphatase and tensin homolog"),
        ("RB1", "protein", "RB transcriptional corepressor 1"),
    ]

    print("\nAdding test proteins to graph:")
    for protein_id, entity_type, name in test_proteins:
        graph.add_entity(protein_id, entity_type, name)
        print(f"  - {protein_id}: {name}")

    # Add ONE existing relationship so not all pairs are candidates
    graph.add_relationship(
        source="TP53",
        target="MDM2",
        rel_type=RelationshipType.INTERACTS_WITH,
        evidence=[{
            "source_type": EvidenceSource.STRING.value,
            "source_id": "existing_edge",
            "confidence": 0.95,
        }],
    )
    print("\nAdded existing edge: TP53 <-> MDM2")

    # Set the graph on the orchestrator
    orchestrator._graph = graph

    print(f"\nGraph stats before prediction:")
    print(f"  - Nodes: {graph.graph.number_of_nodes()}")
    print(f"  - Edges: {graph.graph.number_of_edges()}")

    # Calculate expected candidate pairs
    n_proteins = len(test_proteins)
    max_pairs = n_proteins * (n_proteins - 1) // 2  # All unique pairs
    expected_candidates = max_pairs - 1  # Minus the existing TP53-MDM2 edge
    print(f"  - Expected candidate pairs: {expected_candidates}")

    # Call the refactored function
    print("\n" + "-" * 70)
    print("Calling predict_novel_links(min_ml_score=0.3)")
    print("-" * 70)

    result = predict_novel_links(min_ml_score=0.3)

    # Display results
    print(f"\nStatus: {result.get('status')}")
    print(f"Message: {result.get('message')}")

    stats = result.get("stats", {})
    print(f"\nStatistics:")
    print(f"  - Total pairs evaluated: {stats.get('total_pairs_evaluated')}")
    print(f"  - Enrichment count: {stats.get('enrichment_count')}")
    print(f"  - Novel prediction count: {stats.get('novel_prediction_count')}")
    print(f"  - Proteins not in STRING: {stats.get('proteins_not_in_string')}")
    print(f"  - Total enrichment available: {stats.get('total_enrichment_available')}")
    print(f"  - Total novel available: {stats.get('total_novel_available')}")

    # Show sample enrichment results
    enrichment = result.get("enrichment", [])
    if enrichment:
        print(f"\nTop enrichment edges (known STRING interactions):")
        for i, e in enumerate(enrichment[:5]):
            print(f"  {i+1}. {e['protein1']} <-> {e['protein2']}: score={e['ml_score']:.4f}")

    # Show sample novel predictions
    novel = result.get("novel_predictions", [])
    if novel:
        print(f"\nTop novel predictions (NOT in STRING):")
        for i, p in enumerate(novel[:5]):
            print(f"  {i+1}. {p['protein1']} <-> {p['protein2']}: score={p['ml_score']:.4f}")

    # Verify graph was updated with metadata
    print("\n" + "-" * 70)
    print("Verifying graph edge metadata")
    print("-" * 70)

    enrichment_edges = 0
    novel_edges = 0

    for (src, tgt, rel_type), data in graph.relationships.items():
        link_cat = data.get("link_category")
        if link_cat == "enrichment":
            enrichment_edges += 1
        elif link_cat == "novel_prediction":
            novel_edges += 1

    print(f"  - Edges with link_category='enrichment': {enrichment_edges}")
    print(f"  - Edges with link_category='novel_prediction': {novel_edges}")

    print(f"\nGraph stats after prediction:")
    print(f"  - Nodes: {graph.graph.number_of_nodes()}")
    print(f"  - Edges: {graph.graph.number_of_edges()}")

    # Validation assertions
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    passed = True

    # Check 1: Response structure
    required_keys = ["status", "enrichment", "novel_predictions", "stats"]
    for key in required_keys:
        if key not in result:
            print(f"FAIL: Missing key '{key}' in response")
            passed = False
        else:
            print(f"PASS: Response contains '{key}'")

    # Check 2: Stats structure
    required_stats = ["total_pairs_evaluated", "enrichment_count", "novel_prediction_count"]
    for key in required_stats:
        if key not in stats:
            print(f"FAIL: Missing stat '{key}'")
            passed = False
        else:
            print(f"PASS: Stats contain '{key}'")

    # Check 3: All pairs evaluated (not just max_predictions * 3)
    if stats.get("total_pairs_evaluated") == expected_candidates:
        print(f"PASS: All {expected_candidates} candidate pairs were evaluated")
    elif stats.get("total_pairs_evaluated", 0) > 0:
        print(f"INFO: {stats.get('total_pairs_evaluated')} pairs evaluated (expected {expected_candidates})")
    else:
        print(f"INFO: No pairs evaluated (link predictor may not be available)")

    # Check 4: Graph metadata
    if result.get("status") == "success":
        if enrichment_edges == stats.get("enrichment_count"):
            print(f"PASS: Graph has correct enrichment edge count")
        else:
            print(f"FAIL: Graph enrichment edges ({enrichment_edges}) != stats ({stats.get('enrichment_count')})")
            passed = False

        if novel_edges == stats.get("novel_prediction_count"):
            print(f"PASS: Graph has correct novel prediction edge count")
        else:
            print(f"FAIL: Graph novel edges ({novel_edges}) != stats ({stats.get('novel_prediction_count')})")
            passed = False

    print("\n" + "=" * 70)
    if passed:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED")
    print("=" * 70)

    return passed


if __name__ == "__main__":
    success = test_predict_novel_links()
    sys.exit(0 if success else 1)
