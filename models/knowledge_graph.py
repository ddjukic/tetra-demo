"""
Knowledge Graph data structures for the Scientific Knowledge Graph Agent.

This module provides the core data structures for representing proteins,
their relationships, and evidence backing those relationships.

Graph Data Science algorithms included:
- Centrality metrics: PageRank, Betweenness, Degree, Closeness
- Community detection: Louvain algorithm
- Pathfinding: Shortest path, all simple paths
- Drug discovery: DIAMOnD, Network Proximity, Synergy, Robustness
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any
import json
import random

import networkx as nx
import numpy as np


class EvidenceSource(Enum):
    """Source types for relationship evidence."""
    LITERATURE = "literature"
    STRING = "string"
    ML_PREDICTED = "ml_predicted"


class RelationshipType(Enum):
    """Types of relationships between biological entities."""
    ACTIVATES = "activates"
    INHIBITS = "inhibits"
    ASSOCIATED_WITH = "associated_with"
    REGULATES = "regulates"
    BINDS_TO = "binds_to"
    INTERACTS_WITH = "interacts_with"
    COOCCURS_WITH = "cooccurs_with"
    HYPOTHESIZED = "hypothesized"


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    PROTEIN = "protein"
    GENE = "gene"
    DISEASE = "disease"
    CHEMICAL = "chemical"
    PATHWAY = "pathway"
    UNKNOWN = "unknown"


# =============================================================================
# Data Classes for Algorithm Results
# =============================================================================

@dataclass
class CentralityResult:
    """Result from centrality calculation."""
    method: str
    top_entities: list[tuple[str, float]]
    all_scores: dict[str, float]


@dataclass
class CommunityResult:
    """Result from community detection."""
    num_communities: int
    communities: list[dict[str, Any]]
    modularity: float


@dataclass
class PathResult:
    """Result from pathfinding."""
    source: str
    target: str
    shortest_path: list[dict[str, Any]]
    distance: int
    alternative_paths: list[list[str]]
    path_count: int


@dataclass
class DIAMOnDResult:
    """Result from DIAMOnD disease module detection."""
    seed_genes: list[str]
    module_genes: list[str]
    ranked_candidates: list[tuple[str, float]]
    module_size: int
    iterations_run: int


@dataclass
class NetworkProximityResult:
    """Result from network proximity calculation."""
    drug_targets: list[str]
    disease_genes: list[str]
    observed_distance: float
    expected_distance: float
    z_score: float
    p_value: float
    is_significant: bool
    interpretation: str


@dataclass
class SynergyResult:
    """Result from synergy prediction."""
    target1: str
    target2: str
    synergy_score: float
    disease_proximity_avg: float
    target_separation: int
    module_coverage: float
    complementarity: float
    interpretation: str


@dataclass
class RobustnessResult:
    """Result from network robustness analysis."""
    target: str
    disease_impact: float
    global_impact: float
    therapeutic_index: float
    compensatory_paths: int
    interpretation: str


class KnowledgeGraph:
    """
    Evidence-backed knowledge graph for biological entities and relationships.

    Uses NetworkX MultiDiGraph as the underlying structure to support:
    - Directed edges (A activates B != B activates A)
    - Multiple edges between same nodes (different relationship types)
    - Rich edge attributes (evidence, scores, etc.)

    Attributes:
        graph: NetworkX MultiDiGraph for the actual graph structure
        entities: Dictionary mapping entity_id to entity metadata
        relationships: Dictionary mapping (source, target, type) to relationship data
    """

    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.graph = nx.MultiDiGraph()
        self.entities: dict[str, dict[str, Any]] = {}  # id -> {type, name, aliases, metadata}
        self.relationships: dict[tuple[str, str, str], dict[str, Any]] = {}  # (src, tgt, type) -> {evidence, ml_score, ...}

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        **metadata: Any
    ) -> None:
        """
        Add an entity to the graph.

        If the entity already exists, metadata is merged (new values override).

        Args:
            entity_id: Unique identifier for the entity (e.g., gene symbol, UniProt ID)
            entity_type: Type of entity (protein, gene, disease, etc.)
            name: Human-readable name
            **metadata: Additional metadata (aliases, description, etc.)
        """
        if entity_id in self.entities:
            # Merge metadata
            self.entities[entity_id].update(metadata)
            self.entities[entity_id]["name"] = name
            self.entities[entity_id]["type"] = entity_type
        else:
            self.entities[entity_id] = {
                "type": entity_type,
                "name": name,
                "aliases": metadata.get("aliases", []),
                **metadata
            }

        # Add node to NetworkX graph
        self.graph.add_node(entity_id, **self.entities[entity_id])

    def add_relationship(
        self,
        source: str,
        target: str,
        rel_type: RelationshipType,
        evidence: Optional[list[dict[str, Any]]] = None,
        ml_score: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Add a relationship between two entities, merging evidence if the relationship exists.

        Args:
            source: Source entity ID
            target: Target entity ID
            rel_type: Type of relationship
            evidence: List of evidence dictionaries with keys:
                - source_type: EvidenceSource value
                - source_id: Identifier (e.g., PMID)
                - confidence: Float 0-1
                - text_snippet: Optional supporting text
            ml_score: ML-predicted probability (0-1)
            **kwargs: Additional relationship attributes (reasoning, inferred_type, etc.)
        """
        # Ensure both entities exist in the graph
        if source not in self.entities:
            self.add_entity(source, EntityType.UNKNOWN.value, source)
        if target not in self.entities:
            self.add_entity(target, EntityType.UNKNOWN.value, target)

        rel_key = (source, target, rel_type.value)

        if rel_key in self.relationships:
            # Merge evidence
            existing = self.relationships[rel_key]
            if evidence:
                existing_evidence = existing.get("evidence", [])
                # Deduplicate by source_id
                existing_ids = {e.get("source_id") for e in existing_evidence}
                for ev in evidence:
                    if ev.get("source_id") not in existing_ids:
                        existing_evidence.append(ev)
                existing["evidence"] = existing_evidence

            # Update ml_score if provided and higher
            if ml_score is not None:
                if existing.get("ml_score") is None or ml_score > existing.get("ml_score"):
                    existing["ml_score"] = ml_score

            # Update other attributes
            for key, value in kwargs.items():
                if value is not None:
                    existing[key] = value
        else:
            # Create new relationship
            self.relationships[rel_key] = {
                "source": source,
                "target": target,
                "relation_type": rel_type.value,
                "evidence": evidence or [],
                "ml_score": ml_score,
                **kwargs
            }

        # Add edge to NetworkX graph
        edge_data = self.relationships[rel_key].copy()
        self.graph.add_edge(source, target, key=rel_type.value, **edge_data)

    def get_relationship(self, source: str, target: str) -> Optional[dict[str, Any]]:
        """
        Get all relationship data between two entities.

        Args:
            source: Source entity ID
            target: Target entity ID

        Returns:
            Dictionary with all relationships between the entities, or None if none exist.
            Keys are relationship types, values are relationship data.
        """
        relationships = {}
        for (src, tgt, rel_type), data in self.relationships.items():
            if src == source and tgt == target:
                relationships[rel_type] = data

        return relationships if relationships else None

    def get_neighbors(
        self,
        node: str,
        max_neighbors: int = 10
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Get neighboring nodes with relationship data.

        Returns both incoming and outgoing neighbors.

        Args:
            node: Node ID to find neighbors for
            max_neighbors: Maximum number of neighbors to return

        Returns:
            List of tuples (neighbor_id, relationship_data)
        """
        if node not in self.graph:
            return []

        neighbors = []

        # Outgoing edges
        for _, target, data in self.graph.out_edges(node, data=True):
            neighbors.append((target, {
                "direction": "outgoing",
                **data
            }))

        # Incoming edges
        for source, _, data in self.graph.in_edges(node, data=True):
            neighbors.append((source, {
                "direction": "incoming",
                **data
            }))

        # Sort by evidence count and ml_score
        def sort_key(item: tuple[str, dict]) -> tuple[int, float]:
            data = item[1]
            evidence_count = len(data.get("evidence", []))
            ml_score = data.get("ml_score", 0.0) or 0.0
            return (evidence_count, ml_score)

        neighbors.sort(key=sort_key, reverse=True)

        return neighbors[:max_neighbors]

    def get_novel_predictions(self, min_ml_score: float = 0.7) -> list[dict[str, Any]]:
        """
        Get ML-predicted edges with no literature support.

        These are edges that only have ML evidence (no PubMed citations).

        Args:
            min_ml_score: Minimum ML prediction score to include

        Returns:
            List of relationship dictionaries for novel predictions
        """
        novel = []
        for rel_key, data in self.relationships.items():
            ml_score = data.get("ml_score")
            if ml_score is None or ml_score < min_ml_score:
                continue

            # Check if there's only ML evidence (no literature)
            evidence = data.get("evidence", [])
            has_literature = any(
                ev.get("source_type") == EvidenceSource.LITERATURE.value
                for ev in evidence
            )

            if not has_literature:
                novel.append({
                    "source": rel_key[0],
                    "target": rel_key[1],
                    "relation_type": rel_key[2],
                    "ml_score": ml_score,
                    **{k: v for k, v in data.items() if k not in ["source", "target", "relation_type", "ml_score"]}
                })

        # Sort by ml_score descending
        novel.sort(key=lambda x: x.get("ml_score", 0), reverse=True)

        return novel

    def to_summary(self) -> dict[str, Any]:
        """
        Return graph statistics.

        Returns:
            Dictionary with node count, edge count, entity types, relationship types, etc.
        """
        # Count entity types
        entity_type_counts: dict[str, int] = {}
        for entity_data in self.entities.values():
            etype = entity_data.get("type", "unknown")
            entity_type_counts[etype] = entity_type_counts.get(etype, 0) + 1

        # Count relationship types
        rel_type_counts: dict[str, int] = {}
        for (_, _, rel_type), _ in self.relationships.items():
            rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1

        # Count evidence sources
        evidence_source_counts: dict[str, int] = {}
        for data in self.relationships.values():
            for ev in data.get("evidence", []):
                source_type = ev.get("source_type", "unknown")
                evidence_source_counts[source_type] = evidence_source_counts.get(source_type, 0) + 1

        # Count edges with ML predictions
        ml_predicted_edges = sum(
            1 for data in self.relationships.values()
            if data.get("ml_score") is not None
        )

        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "relationship_count": len(self.relationships),
            "entity_types": entity_type_counts,
            "relationship_types": rel_type_counts,
            "evidence_sources": evidence_source_counts,
            "ml_predicted_edges": ml_predicted_edges,
            "novel_predictions": len(self.get_novel_predictions())
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Export graph as dict for JSON serialization.

        Returns:
            Dictionary with entities and relationships suitable for JSON export.
        """
        return {
            "entities": self.entities,
            "relationships": {
                f"{src}|{tgt}|{rel_type}": data
                for (src, tgt, rel_type), data in self.relationships.items()
            },
            "summary": self.to_summary()
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Export graph as JSON string.

        Args:
            indent: Indentation level for pretty printing

        Returns:
            JSON string representation of the graph
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        """
        Create a KnowledgeGraph from a dictionary.

        Args:
            data: Dictionary with entities and relationships

        Returns:
            New KnowledgeGraph instance
        """
        kg = cls()

        # Restore entities
        for entity_id, entity_data in data.get("entities", {}).items():
            entity_type = entity_data.pop("type", EntityType.UNKNOWN.value)
            name = entity_data.pop("name", entity_id)
            kg.add_entity(entity_id, entity_type, name, **entity_data)

        # Restore relationships
        for rel_key, rel_data in data.get("relationships", {}).items():
            parts = rel_key.split("|")
            if len(parts) == 3:
                source, target, rel_type_str = parts
                try:
                    rel_type = RelationshipType(rel_type_str)
                except ValueError:
                    rel_type = RelationshipType.ASSOCIATED_WITH

                evidence = rel_data.get("evidence", [])
                ml_score = rel_data.get("ml_score")
                other_data = {
                    k: v for k, v in rel_data.items()
                    if k not in ["source", "target", "relation_type", "evidence", "ml_score"]
                }
                kg.add_relationship(source, target, rel_type, evidence, ml_score, **other_data)

        return kg

    def get_entity_interactions_summary(self, entity_id: str) -> str:
        """
        Get a text summary of an entity's interactions for LLM context.

        Args:
            entity_id: Entity ID to summarize

        Returns:
            Formatted string describing the entity's interactions
        """
        if entity_id not in self.entities:
            return f"{entity_id}: No known interactions"

        neighbors = self.get_neighbors(entity_id, max_neighbors=10)
        if not neighbors:
            return f"{entity_id}: No known interactions"

        lines = [f"Known interactions for {entity_id}:"]
        for neighbor_id, rel_data in neighbors:
            direction = rel_data.get("direction", "unknown")
            rel_type = rel_data.get("relation_type", "interacts_with")
            evidence_count = len(rel_data.get("evidence", []))
            ml_score = rel_data.get("ml_score")

            if direction == "outgoing":
                line = f"  - {entity_id} {rel_type} {neighbor_id}"
            else:
                line = f"  - {neighbor_id} {rel_type} {entity_id}"

            if evidence_count > 0:
                line += f" (evidence: {evidence_count} sources)"
            if ml_score is not None:
                line += f" (ML score: {ml_score:.2f})"

            lines.append(line)

        return "\n".join(lines)

    # =========================================================================
    # Graph Data Science: Centrality Algorithms
    # =========================================================================

    def compute_centrality(self, method: str = "pagerank", top_k: int = 10) -> CentralityResult:
        """
        Compute centrality scores for all nodes.

        Args:
            method: Centrality method - "pagerank", "betweenness", "degree", "closeness"
            top_k: Number of top entities to return

        Returns:
            CentralityResult with top entities and all scores
        """
        if self.graph.number_of_nodes() == 0:
            return CentralityResult(method=method, top_entities=[], all_scores={})

        if method == "pagerank":
            scores = nx.pagerank(self.graph)
        elif method == "betweenness":
            scores = nx.betweenness_centrality(self.graph)
        elif method == "degree":
            scores = nx.degree_centrality(self.graph)
        elif method == "closeness":
            scores = nx.closeness_centrality(self.graph)
        else:
            scores = nx.pagerank(self.graph)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return CentralityResult(
            method=method,
            top_entities=ranked[:top_k],
            all_scores=scores
        )

    # =========================================================================
    # Graph Data Science: Community Detection
    # =========================================================================

    def detect_communities(self) -> CommunityResult:
        """
        Detect communities using the Louvain algorithm.

        Returns:
            CommunityResult with communities and modularity score
        """
        if self.graph.number_of_nodes() == 0:
            return CommunityResult(num_communities=0, communities=[], modularity=0.0)

        # Convert to undirected for community detection
        G_undirected = self.graph.to_undirected()

        try:
            from networkx.algorithms import community
            communities = community.louvain_communities(G_undirected)
            modularity = community.modularity(G_undirected, communities)
        except Exception:
            return CommunityResult(num_communities=0, communities=[], modularity=0.0)

        labeled_communities = []
        for i, comm in enumerate(communities):
            entity_types = [
                self.entities.get(node, {}).get("type", "unknown")
                for node in comm
            ]
            dominant_type = max(set(entity_types), key=entity_types.count) if entity_types else "unknown"

            labeled_communities.append({
                "id": i,
                "size": len(comm),
                "nodes": list(comm)[:20],
                "dominant_type": dominant_type
            })

        return CommunityResult(
            num_communities=len(communities),
            communities=labeled_communities,
            modularity=modularity
        )

    # =========================================================================
    # Graph Data Science: Pathfinding
    # =========================================================================

    def find_shortest_path(
        self,
        source: str,
        target: str,
        max_alternatives: int = 5
    ) -> PathResult:
        """
        Find the shortest path between two entities.

        Args:
            source: Source entity ID
            target: Target entity ID
            max_alternatives: Maximum alternative paths to return

        Returns:
            PathResult with path details
        """
        if source not in self.graph or target not in self.graph:
            return PathResult(
                source=source,
                target=target,
                shortest_path=[],
                distance=-1,
                alternative_paths=[],
                path_count=0
            )

        try:
            path = nx.shortest_path(self.graph, source, target)
            all_paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=len(path) + 2
            ))

            annotated_path = self._annotate_path(path)

            return PathResult(
                source=source,
                target=target,
                shortest_path=annotated_path,
                distance=len(path) - 1,
                alternative_paths=all_paths[:max_alternatives],
                path_count=len(all_paths)
            )

        except nx.NetworkXNoPath:
            return PathResult(
                source=source,
                target=target,
                shortest_path=[],
                distance=-1,
                alternative_paths=[],
                path_count=0
            )

    def _annotate_path(self, path: list[str]) -> list[dict[str, Any]]:
        """Add relationship info to a path."""
        annotated = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            edge_data = self.graph.get_edge_data(source, target)
            first_edge = list(edge_data.values())[0] if edge_data else {}

            annotated.append({
                "source": source,
                "target": target,
                "relationship": first_edge.get("relation_type", "unknown"),
                "evidence_count": len(first_edge.get("evidence", [])),
                "ml_score": first_edge.get("ml_score")
            })

        return annotated

    # =========================================================================
    # Drug Discovery: DIAMOnD Algorithm
    # Based on: Ghiassian et al. (2015) PLoS Computational Biology
    # =========================================================================

    def run_diamond(
        self,
        seed_genes: list[str],
        max_iterations: int = 200,
        alpha: float = 1.0
    ) -> DIAMOnDResult:
        """
        DIAMOnD: Disease Module Detection Algorithm.

        Iteratively expands a disease module from seed genes by adding genes
        with highest connectivity significance to the current module.

        Args:
            seed_genes: Known disease-associated genes
            max_iterations: Maximum genes to add to module
            alpha: Weight parameter for connectivity significance

        Returns:
            DIAMOnDResult with module genes and ranked candidates
        """
        if self.graph.number_of_nodes() == 0:
            return DIAMOnDResult(
                seed_genes=seed_genes,
                module_genes=[],
                ranked_candidates=[],
                module_size=0,
                iterations_run=0
            )

        G = self.graph.to_undirected()

        seed_set = {g for g in seed_genes if g in G}
        if not seed_set:
            return DIAMOnDResult(
                seed_genes=seed_genes,
                module_genes=[],
                ranked_candidates=[],
                module_size=0,
                iterations_run=0
            )

        module = set(seed_set)
        ranked_additions = []

        for _ in range(max_iterations):
            candidates = set()
            for node in module:
                candidates.update(G.neighbors(node))
            candidates -= module

            if not candidates:
                break

            candidate_scores = []
            for candidate in candidates:
                k_s = sum(1 for neighbor in G.neighbors(candidate) if neighbor in module)
                k = G.degree(candidate)
                s = len(module)
                N = G.number_of_nodes()

                if k > 0 and s > 0:
                    expected = (k * s) / N
                    variance = expected * (1 - s/N) * (N - k) / (N - 1) if N > 1 else 1
                    std = np.sqrt(variance) if variance > 0 else 1
                    z_score = (k_s - expected) / std if std > 0 else 0
                    score = z_score * alpha
                else:
                    score = 0

                candidate_scores.append((candidate, score))

            candidate_scores.sort(key=lambda x: x[1], reverse=True)

            if not candidate_scores or candidate_scores[0][1] <= 0:
                break

            best_candidate, best_score = candidate_scores[0]
            module.add(best_candidate)
            ranked_additions.append((best_candidate, best_score))

        return DIAMOnDResult(
            seed_genes=list(seed_set),
            module_genes=list(module - seed_set),
            ranked_candidates=ranked_additions,
            module_size=len(module),
            iterations_run=len(ranked_additions)
        )

    # =========================================================================
    # Drug Discovery: Network Proximity
    # Based on: Guney et al. (2016) Nature Communications
    # =========================================================================

    def calculate_network_proximity(
        self,
        drug_targets: list[str],
        disease_genes: list[str],
        n_random: int = 1000,
        method: str = "closest"
    ) -> NetworkProximityResult:
        """
        Calculate network proximity between drug targets and disease genes.

        Args:
            drug_targets: List of drug target genes/proteins
            disease_genes: List of disease-associated genes
            n_random: Number of random permutations for z-score
            method: Distance method - "closest" or "shortest"

        Returns:
            NetworkProximityResult with z-score and significance
        """
        if self.graph.number_of_nodes() == 0:
            return NetworkProximityResult(
                drug_targets=drug_targets,
                disease_genes=disease_genes,
                observed_distance=float('inf'),
                expected_distance=float('inf'),
                z_score=0.0,
                p_value=1.0,
                is_significant=False,
                interpretation="Graph is empty"
            )

        G = self.graph.to_undirected()

        targets_in_graph = [t for t in drug_targets if t in G]
        diseases_in_graph = [d for d in disease_genes if d in G]

        if not targets_in_graph or not diseases_in_graph:
            return NetworkProximityResult(
                drug_targets=drug_targets,
                disease_genes=disease_genes,
                observed_distance=float('inf'),
                expected_distance=float('inf'),
                z_score=0.0,
                p_value=1.0,
                is_significant=False,
                interpretation="Drug targets or disease genes not found in graph"
            )

        observed = self._calculate_set_distance(G, targets_in_graph, diseases_in_graph, method)

        if observed == float('inf'):
            return NetworkProximityResult(
                drug_targets=drug_targets,
                disease_genes=disease_genes,
                observed_distance=float('inf'),
                expected_distance=float('inf'),
                z_score=0.0,
                p_value=1.0,
                is_significant=False,
                interpretation="No path exists between drug targets and disease genes"
            )

        all_nodes = list(G.nodes())
        random_distances = []

        for _ in range(n_random):
            random_targets = random.sample(all_nodes, min(len(targets_in_graph), len(all_nodes)))
            random_diseases = random.sample(all_nodes, min(len(diseases_in_graph), len(all_nodes)))
            rand_dist = self._calculate_set_distance(G, random_targets, random_diseases, method)
            if rand_dist != float('inf'):
                random_distances.append(rand_dist)

        if not random_distances:
            return NetworkProximityResult(
                drug_targets=drug_targets,
                disease_genes=disease_genes,
                observed_distance=observed,
                expected_distance=float('inf'),
                z_score=0.0,
                p_value=1.0,
                is_significant=False,
                interpretation="Could not generate random reference distribution"
            )

        expected = float(np.mean(random_distances))
        std = float(np.std(random_distances))
        z_score = (observed - expected) / std if std > 0 else 0.0
        p_value = sum(1 for d in random_distances if d <= observed) / len(random_distances)
        is_significant = z_score < -2.0 and p_value < 0.05

        if is_significant:
            interpretation = f"SIGNIFICANT proximity (z={z_score:.2f}). Drug targets are closer to disease genes than random."
        elif z_score < 0:
            interpretation = f"Moderate proximity (z={z_score:.2f}). Some relationship exists but not significant."
        else:
            interpretation = f"No proximity (z={z_score:.2f}). Drug targets are not closer than random."

        return NetworkProximityResult(
            drug_targets=drug_targets,
            disease_genes=disease_genes,
            observed_distance=observed,
            expected_distance=expected,
            z_score=z_score,
            p_value=p_value,
            is_significant=is_significant,
            interpretation=interpretation
        )

    def _calculate_set_distance(
        self,
        G: nx.Graph,
        set_a: list[str],
        set_b: list[str],
        method: str = "closest"
    ) -> float:
        """Calculate distance between two gene sets."""
        if method == "closest":
            min_dist = float('inf')
            for a in set_a:
                for b in set_b:
                    try:
                        dist = nx.shortest_path_length(G, a, b)
                        min_dist = min(min_dist, dist)
                    except nx.NetworkXNoPath:
                        pass
            return min_dist
        else:
            distances = []
            for a in set_a:
                min_to_b = float('inf')
                for b in set_b:
                    try:
                        dist = nx.shortest_path_length(G, a, b)
                        min_to_b = min(min_to_b, dist)
                    except nx.NetworkXNoPath:
                        pass
                if min_to_b != float('inf'):
                    distances.append(min_to_b)
            return float(np.mean(distances)) if distances else float('inf')

    # =========================================================================
    # Drug Discovery: Synergy Prediction
    # Based on: Cheng et al. (2019) Nature Communications
    # =========================================================================

    def predict_synergy(
        self,
        target1: str,
        target2: str,
        disease_genes: list[str]
    ) -> SynergyResult:
        """
        Predict synergy potential for two drug targets in combination therapy.

        Args:
            target1: First drug target
            target2: Second drug target
            disease_genes: Disease-associated genes for context

        Returns:
            SynergyResult with synergy scores and interpretation
        """
        if self.graph.number_of_nodes() == 0:
            return SynergyResult(
                target1=target1, target2=target2, synergy_score=0.0,
                disease_proximity_avg=-1, target_separation=-1,
                module_coverage=0.0, complementarity=0.0,
                interpretation="Graph is empty"
            )

        G = self.graph.to_undirected()

        if target1 not in G or target2 not in G:
            return SynergyResult(
                target1=target1, target2=target2, synergy_score=0.0,
                disease_proximity_avg=-1, target_separation=-1,
                module_coverage=0.0, complementarity=0.0,
                interpretation="One or both targets not found in graph"
            )

        disease_ids = [d for d in disease_genes if d in G]

        try:
            separation = nx.shortest_path_length(G, target1, target2)
        except nx.NetworkXNoPath:
            separation = -1

        if separation == -1:
            separation_score = 0.0
        elif 2 <= separation <= 4:
            separation_score = 1.0
        elif separation == 1:
            separation_score = 0.3
        elif separation > 4:
            separation_score = max(0, 1 - (separation - 4) * 0.2)
        else:
            separation_score = 0.5

        t1_neighbors = set(G.neighbors(target1)) | {target1}
        t2_neighbors = set(G.neighbors(target2)) | {target2}
        combined_reach = t1_neighbors | t2_neighbors

        if disease_ids:
            disease_set = set(disease_ids)
            coverage = len(combined_reach & disease_set) / len(disease_set)
        else:
            coverage = 0.0

        overlap = t1_neighbors & t2_neighbors
        union = t1_neighbors | t2_neighbors
        complementarity = 1 - (len(overlap) / len(union)) if union else 0

        if disease_ids:
            distances = []
            for disease_id in disease_ids:
                try:
                    d1 = nx.shortest_path_length(G, target1, disease_id)
                    d2 = nx.shortest_path_length(G, target2, disease_id)
                    distances.append(min(d1, d2))
                except nx.NetworkXNoPath:
                    pass
            avg_proximity = float(np.mean(distances)) if distances else -1
        else:
            avg_proximity = -1

        proximity_score = 1 / (1 + avg_proximity) if avg_proximity >= 0 else 0.0

        synergy_score = (
            0.3 * separation_score +
            0.3 * coverage +
            0.2 * complementarity +
            0.2 * proximity_score
        )

        if synergy_score > 0.7:
            interpretation = f"HIGH synergy potential. Optimal separation ({separation}), good complementarity ({complementarity:.2f})."
        elif synergy_score > 0.4:
            interpretation = f"MODERATE synergy potential. Separation: {separation}, Coverage: {coverage:.2f}."
        else:
            interpretation = f"LOW synergy potential. Targets may be redundant or lack disease relevance."

        return SynergyResult(
            target1=target1,
            target2=target2,
            synergy_score=synergy_score,
            disease_proximity_avg=avg_proximity,
            target_separation=separation,
            module_coverage=coverage,
            complementarity=complementarity,
            interpretation=interpretation
        )

    # =========================================================================
    # Drug Discovery: Network Robustness Analysis
    # =========================================================================

    def analyze_robustness(
        self,
        target: str,
        disease_genes: list[str]
    ) -> RobustnessResult:
        """
        Analyze network robustness impact of targeting a specific node.

        Evaluates therapeutic potential by measuring disease impact vs
        global network disruption.

        Args:
            target: Drug target to analyze
            disease_genes: Disease-associated genes

        Returns:
            RobustnessResult with therapeutic index
        """
        if self.graph.number_of_nodes() == 0 or target not in self.graph:
            return RobustnessResult(
                target=target, disease_impact=0.0, global_impact=0.0,
                therapeutic_index=0.0, compensatory_paths=0,
                interpretation="Target not found in graph"
            )

        G = self.graph.to_undirected()
        disease_ids = [d for d in disease_genes if d in G]

        original_components = nx.number_connected_components(G)

        G_removed = G.copy()
        G_removed.remove_node(target)

        new_components = nx.number_connected_components(G_removed)
        component_increase = new_components - original_components
        global_impact = min(1.0, component_increase * 0.5)

        if disease_ids:
            paths_broken = 0
            compensatory = 0
            for disease_id in disease_ids:
                if disease_id == target:
                    paths_broken += 1
                    continue
                try:
                    nx.shortest_path(G, target, disease_id)
                    paths_broken += 1
                    if disease_id in G_removed:
                        neighbors = list(G.neighbors(target))
                        for neighbor in neighbors:
                            if neighbor in G_removed and neighbor != disease_id:
                                try:
                                    nx.shortest_path(G_removed, neighbor, disease_id)
                                    compensatory += 1
                                    break
                                except nx.NetworkXNoPath:
                                    pass
                except nx.NetworkXNoPath:
                    pass

            disease_impact = paths_broken / len(disease_ids) if disease_ids else 0.0
        else:
            disease_impact = 0.0
            compensatory = 0

        if global_impact > 0:
            therapeutic_index = disease_impact / global_impact
        else:
            therapeutic_index = disease_impact * 10

        if therapeutic_index > 2.0 and disease_impact > 0.3:
            interpretation = f"EXCELLENT target. High disease disruption ({disease_impact:.2f}) with minimal global impact."
        elif therapeutic_index > 1.0:
            interpretation = f"GOOD target. Moderate therapeutic index ({therapeutic_index:.2f})."
        elif disease_impact > 0.5:
            interpretation = f"RISKY target. High disease impact but also high global impact."
        else:
            interpretation = f"POOR target. Low disease impact ({disease_impact:.2f})."

        return RobustnessResult(
            target=target,
            disease_impact=disease_impact,
            global_impact=global_impact,
            therapeutic_index=therapeutic_index,
            compensatory_paths=compensatory,
            interpretation=interpretation
        )
