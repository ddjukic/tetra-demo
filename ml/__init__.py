"""
Machine Learning module for protein interaction prediction.

This module provides:
- LinkPredictor: Single Node2Vec model for link prediction
- EnsembleLinkPredictor: Ensemble of multiple Node2Vec models
- estimate_graph_homophily: Analyze graph homophily to select optimal model
- analyze_node_properties: Analyze node-level structural properties
"""

from ml.link_predictor import LinkPredictor
from ml.ensemble_predictor import (
    EnsembleLinkPredictor,
    estimate_graph_homophily,
    analyze_node_properties,
)

__all__ = [
    "LinkPredictor",
    "EnsembleLinkPredictor",
    "estimate_graph_homophily",
    "analyze_node_properties",
]
