"""
Machine Learning module for protein interaction prediction.

This module provides:
- LinkPredictor: Single Node2Vec model for link prediction (gensim-based, CPU)
- PyGLinkPredictor: PyTorch Geometric Node2Vec with MPS/GPU support
- EnsembleLinkPredictor: Ensemble of multiple Node2Vec models
- estimate_graph_homophily: Analyze graph homophily to select optimal model
- analyze_node_properties: Analyze node-level structural properties
- HardNegativeSampler: Hard negative sampling strategies for rigorous evaluation
- evaluate_with_hard_negatives: Evaluate models with multiple negative sampling strategies
"""

from ml.link_predictor import LinkPredictor
from ml.ensemble_predictor import (
    EnsembleLinkPredictor,
    estimate_graph_homophily,
    analyze_node_properties,
)
from ml.hard_negative_sampling import (
    HardNegativeSampler,
    evaluate_with_hard_negatives,
)

# Conditionally import PyG predictor (requires torch)
try:
    from ml.pyg_link_predictor import PyGLinkPredictor
    _HAS_PYTORCH = True
except ImportError:
    PyGLinkPredictor = None
    _HAS_PYTORCH = False

__all__ = [
    "LinkPredictor",
    "PyGLinkPredictor",
    "EnsembleLinkPredictor",
    "estimate_graph_homophily",
    "analyze_node_properties",
    "HardNegativeSampler",
    "evaluate_with_hard_negatives",
]
