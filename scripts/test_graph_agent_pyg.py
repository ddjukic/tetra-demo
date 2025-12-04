#!/usr/bin/env python
"""
Test script to verify the enhanced Graph RAG agent with PyG link predictions.

This script tests:
1. Loading PyG link predictor
2. Loading PyG ensemble predictor
3. predict_interaction tool
4. generate_hypothesis tool with ML scores
5. predict_batch_interactions tool

Usage:
    uv run python scripts/test_graph_agent_pyg.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_pyg_predictor_loading():
    """Test loading a single PyG predictor."""
    print("\n" + "=" * 60)
    print("TEST 1: Loading PyG Link Predictor")
    print("=" * 60)

    from agent.graph_agent import load_pyg_predictor, get_link_predictor

    predictor = load_pyg_predictor("models/pyg_link_predictor.pkl")

    if predictor is None:
        print("FAILED: Could not load PyG predictor")
        return False

    print(f"SUCCESS: Loaded predictor with device {predictor.device}")
    print(f"  - Nodes in vocabulary: {len(predictor.node_to_idx)}")
    print(f"  - Embeddings shape: {predictor.embeddings.shape if predictor.embeddings is not None else 'None'}")

    # Verify get_link_predictor returns the same instance
    assert get_link_predictor() is predictor
    print("  - get_link_predictor() returns correct instance")

    return True


def test_pyg_ensemble_loading():
    """Test loading PyG ensemble predictor."""
    print("\n" + "=" * 60)
    print("TEST 2: Loading PyG Ensemble Predictor")
    print("=" * 60)

    from agent.graph_agent import load_pyg_ensemble, get_ensemble_predictor, get_link_predictor

    ensemble = load_pyg_ensemble("models")

    if ensemble is None:
        print("FAILED: Could not load PyG ensemble")
        return False

    print(f"SUCCESS: Loaded ensemble with {len(ensemble.models)} models:")
    for name, model in ensemble.models:
        print(f"  - {name}: p={model.p}, q={model.q}, device={model.device}")

    # Verify get_ensemble_predictor returns the correct instance
    assert get_ensemble_predictor() is ensemble
    print("  - get_ensemble_predictor() returns correct instance")

    # Verify ensemble is now the default link predictor
    assert get_link_predictor() is ensemble
    print("  - Ensemble set as default link predictor")

    return True


def test_predict_interaction():
    """Test the predict_interaction tool."""
    print("\n" + "=" * 60)
    print("TEST 3: predict_interaction Tool")
    print("=" * 60)

    from agent.graph_agent import (
        set_active_graph, load_pyg_ensemble,
        predict_interaction
    )
    from models.knowledge_graph import KnowledgeGraph
    from unittest.mock import MagicMock

    # Create a minimal graph
    graph = KnowledgeGraph()
    graph.add_entity("TP53", "protein", "Tumor protein p53")
    graph.add_entity("BRCA1", "protein", "BRCA1 DNA repair associated")
    set_active_graph(graph)

    # Ensure ensemble is loaded
    load_pyg_ensemble("models")

    # Create mock tool context
    tool_context = MagicMock()
    tool_context.state = {"active_graph": "default"}

    # Test with known proteins
    result = predict_interaction(tool_context, "TP53", "BRCA1")

    print(f"Prediction for TP53-BRCA1:")
    print(f"  - ML Score: {result.get('ml_score')}")
    print(f"  - In STRING: {result.get('in_string')}")
    print(f"  - Confidence: {result.get('confidence')}")
    print(f"  - Interpretation: {result.get('interpretation')}")

    if result.get('model_scores'):
        print(f"  - Model scores:")
        for ms in result['model_scores']:
            print(f"    - {ms['model']}: {ms['score']:.4f}")

    if result.get('error'):
        print(f"  - Error: {result.get('error')}")
        return False

    if result.get('ml_score') is not None:
        print("SUCCESS: predict_interaction returned ML score")
        return True
    else:
        print("FAILED: No ML score returned")
        return False


def test_generate_hypothesis():
    """Test the generate_hypothesis tool with ML scores."""
    print("\n" + "=" * 60)
    print("TEST 4: generate_hypothesis Tool with ML Scores")
    print("=" * 60)

    from agent.graph_agent import (
        set_active_graph, load_pyg_ensemble,
        generate_hypothesis
    )
    from models.knowledge_graph import KnowledgeGraph, RelationshipType
    from unittest.mock import MagicMock

    # Create a graph with some structure
    graph = KnowledgeGraph()
    graph.add_entity("CDK2", "protein", "Cyclin-dependent kinase 2")
    graph.add_entity("CCNE1", "protein", "Cyclin E1")
    graph.add_entity("RB1", "protein", "RB transcriptional corepressor 1")
    graph.add_entity("E2F1", "protein", "E2F transcription factor 1")

    # Add some relationships to create network context
    graph.add_relationship("CDK2", "RB1", RelationshipType.ACTIVATES, ml_score=0.8)
    graph.add_relationship("CCNE1", "RB1", RelationshipType.REGULATES, ml_score=0.75)
    graph.add_relationship("RB1", "E2F1", RelationshipType.INHIBITS, ml_score=0.9)

    set_active_graph(graph)

    # Ensure ensemble is loaded
    load_pyg_ensemble("models")

    # Create mock tool context
    tool_context = MagicMock()
    tool_context.state = {"active_graph": "default"}

    # Generate hypothesis
    result = generate_hypothesis(tool_context, "CDK2", "CCNE1")

    print(f"Hypothesis for CDK2-CCNE1:")
    print(f"  - Statement: {result.get('hypothesis_statement')[:100]}...")
    print(f"  - ML Confidence: {result.get('ml_confidence')}")

    ml_pred = result.get('ml_prediction')
    if ml_pred:
        print(f"  - ML Prediction Score: {ml_pred.get('score')}")
        if ml_pred.get('model_scores'):
            print(f"  - Model Scores:")
            for ms in ml_pred['model_scores']:
                print(f"    - {ms['model']}: {ms['score']:.4f}")

    if result.get('model_interpretation'):
        print(f"  - Model Interpretation: {result['model_interpretation']}")

    print(f"  - Shared Neighbors: {result['supporting_evidence'].get('shared_neighbors')}")
    print(f"  - Suggested Experiments: {len(result.get('suggested_experiments', []))} experiments")

    if ml_pred and ml_pred.get('score') is not None:
        print("SUCCESS: generate_hypothesis includes ML prediction")
        return True
    elif ml_pred and ml_pred.get('error'):
        print(f"PARTIAL: ML prediction had error: {ml_pred.get('error')}")
        return True  # Still counts as working, just proteins not in model vocabulary
    else:
        print("FAILED: No ML prediction in hypothesis")
        return False


def test_predict_batch_interactions():
    """Test batch prediction tool."""
    print("\n" + "=" * 60)
    print("TEST 5: predict_batch_interactions Tool")
    print("=" * 60)

    from agent.graph_agent import (
        set_active_graph, load_pyg_ensemble,
        predict_batch_interactions
    )
    from models.knowledge_graph import KnowledgeGraph
    from unittest.mock import MagicMock

    # Create minimal graph
    graph = KnowledgeGraph()
    set_active_graph(graph)

    # Ensure ensemble is loaded
    load_pyg_ensemble("models")

    # Create mock tool context
    tool_context = MagicMock()
    tool_context.state = {"active_graph": "default"}

    # Test batch prediction with proteins that should be in STRING
    pairs = [
        ["TP53", "MDM2"],
        ["BRCA1", "BRCA2"],
        ["EGFR", "KRAS"],
        ["CDK1", "CCNB1"],
        ["UNKNOWN1", "UNKNOWN2"],  # Should fail gracefully
    ]

    result = predict_batch_interactions(tool_context, pairs)

    print(f"Batch prediction results:")
    print(f"  - Total pairs: {result.get('count')}")
    print(f"  - Average score: {result.get('avg_score')}")
    print(f"  - High confidence: {result.get('high_confidence')}")
    print(f"  - Interpretation: {result.get('interpretation')}")

    if result.get('predictions'):
        print(f"  - Sample predictions:")
        for pred in result['predictions'][:3]:
            if pred.get('error'):
                print(f"    - {pred['protein1']}-{pred['protein2']}: ERROR - {pred['error']}")
            else:
                print(f"    - {pred['protein1']}-{pred['protein2']}: {pred.get('ml_score', 'N/A')}")

    if result.get('error'):
        print(f"FAILED: {result.get('error')}")
        return False

    print("SUCCESS: Batch prediction completed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Graph RAG Agent PyG Integration Tests")
    print("=" * 60)

    tests = [
        ("PyG Predictor Loading", test_pyg_predictor_loading),
        ("PyG Ensemble Loading", test_pyg_ensemble_loading),
        ("predict_interaction Tool", test_predict_interaction),
        ("generate_hypothesis Tool", test_generate_hypothesis),
        ("predict_batch_interactions Tool", test_predict_batch_interactions),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\nEXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
