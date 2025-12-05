"""
Knowledge Graph data structures for the Scientific Knowledge Graph Agent.

This module provides the core data structures for representing proteins,
their relationships, and evidence backing those relationships.

Graph Data Science algorithms included:
- Centrality metrics: PageRank, Betweenness, Degree, Closeness
- Community detection: Louvain algorithm
- Pathfinding: Shortest path, all simple paths
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

    def get_novel_predictions(self, min_ml_score: float = 0.5) -> list[dict[str, Any]]:
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
            if ml_score is None:
                continue
            # Handle string ml_score (from GraphML load)
            if isinstance(ml_score, str):
                # Skip 'None' strings
                if ml_score == 'None' or ml_score == 'null':
                    continue
                try:
                    ml_score = float(ml_score)
                except ValueError:
                    continue
            if ml_score < min_ml_score:
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

    def get_all_ml_predictions(self, min_ml_score: float = 0.5) -> list[dict[str, Any]]:
        """
        Get ALL ML predictions above threshold, regardless of literature support.

        Unlike get_novel_predictions(), this returns predictions that may also
        have literature evidence. Useful for seeing all ML-scored relationships.

        Args:
            min_ml_score: Minimum ML prediction score to include (default: 0.5)

        Returns:
            List of relationship dictionaries for all ML predictions, sorted by ml_score descending.
        """
        predictions = []
        for rel_key, data in self.relationships.items():
            ml_score = data.get("ml_score")
            if ml_score is None:
                continue
            # Handle string ml_score (from GraphML load)
            if isinstance(ml_score, str):
                # Skip 'None' strings
                if ml_score == 'None' or ml_score == 'null':
                    continue
                try:
                    ml_score = float(ml_score)
                except ValueError:
                    continue
            if ml_score < min_ml_score:
                continue

            predictions.append({
                "source": rel_key[0],
                "target": rel_key[1],
                "relation_type": rel_key[2],
                "ml_score": ml_score,
                "evidence": data.get("evidence", []),
                **{k: v for k, v in data.items() if k not in ["source", "target", "relation_type", "ml_score", "evidence"]}
            })

        # Sort by ml_score descending
        predictions.sort(key=lambda x: x.get("ml_score", 0), reverse=True)

        return predictions

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

    def save(self, path: str, format: str = "graphml") -> str:
        """
        Save the knowledge graph to disk.

        Supports multiple formats:
        - graphml: XML-based format, readable by Cytoscape, Gephi, etc.
        - pickle: Python pickle (includes all attributes, fast)
        - json: JSON export (custom format with entities/relationships)

        Args:
            path: Base path for the output file (extension added based on format)
            format: Output format - "graphml", "pickle", or "json"

        Returns:
            Path to the saved file
        """
        from pathlib import Path
        import pickle

        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "graphml":
            # GraphML is widely compatible (Cytoscape, Gephi, etc.)
            file_path = base_path.with_suffix(".graphml")
            # Convert edge keys to strings for GraphML compatibility
            G_copy = nx.MultiDiGraph()
            for node, attrs in self.graph.nodes(data=True):
                # Filter to serializable attributes
                safe_attrs = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                              for k, v in attrs.items()}
                G_copy.add_node(node, **safe_attrs)
            for u, v, key, attrs in self.graph.edges(keys=True, data=True):
                safe_attrs = {}
                for k, val in attrs.items():
                    if isinstance(val, (str, int, float, bool)):
                        safe_attrs[k] = val
                    elif isinstance(val, list):
                        safe_attrs[k] = json.dumps(val)
                    else:
                        safe_attrs[k] = str(val)
                G_copy.add_edge(u, v, key=str(key), **safe_attrs)
            nx.write_graphml(G_copy, str(file_path))

        elif format == "pickle":
            # Pickle preserves everything but is Python-only
            file_path = base_path.with_suffix(".pkl")
            with open(file_path, "wb") as f:
                pickle.dump({
                    "graph": self.graph,
                    "entities": self.entities,
                    "relationships": self.relationships,
                }, f)

        elif format == "json":
            # JSON is portable but may lose some type info
            file_path = base_path.with_suffix(".json")
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'graphml', 'pickle', or 'json'.")

        return str(file_path)

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """
        Load a knowledge graph from disk.

        Detects format from file extension:
        - .graphml: GraphML format
        - .pkl: Python pickle
        - .json: JSON format

        Args:
            path: Path to the graph file

        Returns:
            Loaded KnowledgeGraph instance
        """
        from pathlib import Path
        import pickle

        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")

        suffix = file_path.suffix.lower()

        if suffix == ".graphml":
            # Load from GraphML
            G = nx.read_graphml(str(file_path))
            kg = cls()
            kg.graph = nx.MultiDiGraph(G)

            # Reconstruct entities from nodes
            for node, attrs in kg.graph.nodes(data=True):
                kg.entities[node] = dict(attrs)

            # Reconstruct relationships from edges
            for u, v, key, attrs in kg.graph.edges(keys=True, data=True):
                # Parse JSON strings back to lists and convert numeric strings
                parsed_attrs = {}
                for k, val in attrs.items():
                    if isinstance(val, str):
                        # Handle 'None' string -> Python None
                        if val == 'None' or val == 'null':
                            parsed_attrs[k] = None
                            continue
                        # Try parsing as JSON array first
                        if val.startswith("["):
                            try:
                                parsed_attrs[k] = json.loads(val)
                                continue
                            except json.JSONDecodeError:
                                pass
                        # Try converting to float (handles scores, weights, etc.)
                        try:
                            parsed_attrs[k] = float(val)
                            # Convert to int if it's a whole number
                            if parsed_attrs[k] == int(parsed_attrs[k]):
                                parsed_attrs[k] = int(parsed_attrs[k])
                            continue
                        except ValueError:
                            pass
                        parsed_attrs[k] = val
                    else:
                        parsed_attrs[k] = val
                kg.relationships[(u, v, key)] = parsed_attrs
                # Update edge attributes in graph too
                kg.graph.edges[u, v, key].update(parsed_attrs)

            return kg

        elif suffix == ".pkl":
            # Load from pickle
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            kg = cls()
            kg.graph = data.get("graph", nx.MultiDiGraph())
            kg.entities = data.get("entities", {})
            kg.relationships = data.get("relationships", {})
            return kg

        elif suffix == ".json":
            # Load from JSON
            with open(file_path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)

        else:
            raise ValueError(f"Unsupported file extension: {suffix}. Use '.graphml', '.pkl', or '.json'.")

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
    # Entity Grounding: INDRA/Gilda Integration
    # =========================================================================

    def get_groundable_entities(self) -> list[str]:
        """Get list of entity IDs that can be grounded (proteins/genes).

        Returns:
            List of entity IDs with type 'protein' or 'gene'.
        """
        groundable_types = {EntityType.PROTEIN.value, EntityType.GENE.value}
        return [
            entity_id for entity_id, data in self.entities.items()
            if data.get("type") in groundable_types
        ]

    async def ground_entities(
        self,
        gilda_client: Any,
        entity_types: list[str] | None = None,
        min_score: float = 0.5,
    ) -> dict[str, Any]:
        """Ground all protein/gene/disease entities using INDRA/Gilda.

        Batch-grounds entities and updates their metadata with grounding info.
        Supports HGNC for genes/proteins, DOID/MESH for diseases.

        Args:
            gilda_client: GildaClient instance for API calls.
            entity_types: Types to ground (default: ['protein', 'gene', 'disease']).
            min_score: Minimum grounding score to accept.

        Returns:
            Dictionary with grounding statistics:
                - total_entities: Number of entities processed
                - grounded: Number successfully grounded
                - ungrounded: Number that failed grounding
                - hgnc_grounded: Number grounded to HGNC specifically
                - doid_grounded: Number grounded to DOID specifically
                - mesh_grounded: Number grounded to MESH specifically
                - grounding_map: Dict mapping entity_id to grounding result
        """
        if entity_types is None:
            entity_types = [
                EntityType.PROTEIN.value,
                EntityType.GENE.value,
                EntityType.DISEASE.value,
            ]

        # Collect entities to ground
        entities_to_ground = [
            entity_id for entity_id, data in self.entities.items()
            if data.get("type") in entity_types
        ]

        if not entities_to_ground:
            return {
                "total_entities": 0,
                "grounded": 0,
                "ungrounded": 0,
                "hgnc_grounded": 0,
                "grounding_map": {},
            }

        # Get entity names for grounding (use 'name' field, fall back to ID)
        texts_to_ground = [
            self.entities[eid].get("name", eid) for eid in entities_to_ground
        ]

        # Batch ground
        results = await gilda_client.ground_batch(texts_to_ground)

        # Update entities with grounding info
        grounded_count = 0
        hgnc_count = 0
        doid_count = 0
        mesh_count = 0
        grounding_map: dict[str, dict[str, Any] | None] = {}

        for entity_id, text in zip(entities_to_ground, texts_to_ground):
            grounding = results.get(text)

            if grounding and grounding.score >= min_score:
                # Store grounding info in entity
                self.entities[entity_id]["grounded_db"] = grounding.db
                self.entities[entity_id]["grounded_id"] = grounding.id
                self.entities[entity_id]["grounded_entry_name"] = grounding.entry_name
                self.entities[entity_id]["grounding_score"] = grounding.score

                # Set hgnc_id if grounded to HGNC (for genes/proteins)
                if grounding.db == "HGNC":
                    self.entities[entity_id]["hgnc_id"] = grounding.full_id
                    hgnc_count += 1
                # Set doid_id if grounded to DOID (for diseases)
                elif grounding.db == "DOID":
                    self.entities[entity_id]["doid_id"] = grounding.full_id
                    doid_count += 1
                # Set mesh_id if grounded to MESH (alternative for diseases)
                elif grounding.db == "MESH":
                    self.entities[entity_id]["mesh_id"] = grounding.full_id
                    mesh_count += 1

                # Update graph node attributes
                self.graph.nodes[entity_id].update({
                    "grounded_db": grounding.db,
                    "grounded_id": grounding.id,
                    "grounded_entry_name": grounding.entry_name,
                    "grounding_score": grounding.score,
                })
                if grounding.db == "HGNC":
                    self.graph.nodes[entity_id]["hgnc_id"] = grounding.full_id
                elif grounding.db == "DOID":
                    self.graph.nodes[entity_id]["doid_id"] = grounding.full_id
                elif grounding.db == "MESH":
                    self.graph.nodes[entity_id]["mesh_id"] = grounding.full_id

                grounded_count += 1
                grounding_map[entity_id] = {
                    "db": grounding.db,
                    "id": grounding.id,
                    "entry_name": grounding.entry_name,
                    "score": grounding.score,
                    "full_id": grounding.full_id,
                }
            else:
                grounding_map[entity_id] = None

        return {
            "total_entities": len(entities_to_ground),
            "grounded": grounded_count,
            "ungrounded": len(entities_to_ground) - grounded_count,
            "hgnc_grounded": hgnc_count,
            "doid_grounded": doid_count,
            "mesh_grounded": mesh_count,
            "grounding_map": grounding_map,
        }

    def deduplicate_by_hgnc(self) -> dict[str, Any]:
        """Merge entities that share the same HGNC ID.

        Entities with identical HGNC IDs are considered synonyms and merged.
        The entity with the canonical HGNC entry_name becomes the primary.
        Other entities become aliases.

        Returns:
            Dictionary with deduplication statistics:
                - original_count: Entity count before deduplication
                - final_count: Entity count after deduplication
                - merged_groups: List of merged entity groups
                - edges_remapped: Number of edges remapped to canonical IDs
        """
        # Group entities by HGNC ID
        hgnc_groups: dict[str, list[str]] = {}
        for entity_id, data in self.entities.items():
            hgnc_id = data.get("hgnc_id")
            if hgnc_id:
                if hgnc_id not in hgnc_groups:
                    hgnc_groups[hgnc_id] = []
                hgnc_groups[hgnc_id].append(entity_id)

        original_count = len(self.entities)
        merged_groups: list[dict[str, Any]] = []
        edges_remapped = 0

        # Process each group that has more than one entity
        for hgnc_id, entity_ids in hgnc_groups.items():
            if len(entity_ids) <= 1:
                continue

            # Find canonical entity (prefer HGNC entry_name match)
            canonical_id = None
            canonical_entry = None

            for eid in entity_ids:
                entry_name = self.entities[eid].get("grounded_entry_name", "")
                if entry_name:
                    canonical_entry = entry_name
                    # If entity_id matches entry_name, it's canonical
                    if eid.upper() == entry_name.upper():
                        canonical_id = eid
                        break

            # If no exact match, prefer the entry_name as new canonical
            if canonical_id is None and canonical_entry:
                # Check if entry_name exists as an entity
                if canonical_entry in self.entities:
                    canonical_id = canonical_entry
                else:
                    # Pick first entity alphabetically
                    canonical_id = sorted(entity_ids)[0]
            elif canonical_id is None:
                canonical_id = sorted(entity_ids)[0]

            aliases_to_merge = [eid for eid in entity_ids if eid != canonical_id]

            if not aliases_to_merge:
                continue

            merged_groups.append({
                "hgnc_id": hgnc_id,
                "canonical": canonical_id,
                "merged": aliases_to_merge,
            })

            # Merge aliases into canonical entity
            canonical_data = self.entities[canonical_id]
            existing_aliases = set(canonical_data.get("aliases", []))

            for alias_id in aliases_to_merge:
                # Add alias ID and its aliases to canonical
                existing_aliases.add(alias_id)
                alias_data = self.entities.get(alias_id, {})
                for a in alias_data.get("aliases", []):
                    existing_aliases.add(a)

                # Remap edges involving this alias
                edges_remapped += self._remap_edges(alias_id, canonical_id)

                # Remove alias entity
                del self.entities[alias_id]
                if alias_id in self.graph:
                    self.graph.remove_node(alias_id)

            # Update canonical with merged aliases
            canonical_data["aliases"] = list(existing_aliases)
            self.graph.nodes[canonical_id]["aliases"] = str(list(existing_aliases))

        return {
            "original_count": original_count,
            "final_count": len(self.entities),
            "merged_groups": merged_groups,
            "edges_remapped": edges_remapped,
        }

    def deduplicate_by_doid(self) -> dict[str, Any]:
        """Merge disease entities that share the same DOID ID.

        Entities with identical DOID IDs are considered synonyms and merged.
        For example, "Alzheimer disease" and "Alzheimer's disease" with the
        same DOID:10652 will be merged into a single canonical entity.

        Returns:
            Dictionary with deduplication statistics:
                - original_count: Entity count before deduplication
                - final_count: Entity count after deduplication
                - merged_groups: List of merged entity groups
                - edges_remapped: Number of edges remapped to canonical IDs
        """
        # Group entities by DOID ID
        doid_groups: dict[str, list[str]] = {}
        for entity_id, data in self.entities.items():
            doid_id = data.get("doid_id")
            if doid_id:
                if doid_id not in doid_groups:
                    doid_groups[doid_id] = []
                doid_groups[doid_id].append(entity_id)

        original_count = len(self.entities)
        merged_groups: list[dict[str, Any]] = []
        edges_remapped = 0

        # Process each group that has more than one entity
        for doid_id, entity_ids in doid_groups.items():
            if len(entity_ids) <= 1:
                continue

            # Find canonical entity (prefer DOID entry_name match)
            canonical_id = None
            canonical_entry = None

            for eid in entity_ids:
                entry_name = self.entities[eid].get("grounded_entry_name", "")
                if entry_name:
                    canonical_entry = entry_name
                    # If entity_id matches entry_name (case-insensitive), it's canonical
                    if eid.lower().replace("'", "").replace(" ", "_") == entry_name.lower().replace("'", "").replace(" ", "_"):
                        canonical_id = eid
                        break

            # If no exact match, prefer the entry_name as new canonical
            if canonical_id is None and canonical_entry:
                # Check if entry_name exists as an entity
                if canonical_entry in self.entities:
                    canonical_id = canonical_entry
                else:
                    # Pick first entity alphabetically
                    canonical_id = sorted(entity_ids)[0]
            elif canonical_id is None:
                canonical_id = sorted(entity_ids)[0]

            aliases_to_merge = [eid for eid in entity_ids if eid != canonical_id]

            if not aliases_to_merge:
                continue

            merged_groups.append({
                "doid_id": doid_id,
                "canonical": canonical_id,
                "merged": aliases_to_merge,
            })

            # Merge aliases into canonical entity
            canonical_data = self.entities[canonical_id]
            existing_aliases = set(canonical_data.get("aliases", []))

            for alias_id in aliases_to_merge:
                # Add alias ID and its aliases to canonical
                existing_aliases.add(alias_id)
                alias_data = self.entities.get(alias_id, {})
                for a in alias_data.get("aliases", []):
                    existing_aliases.add(a)

                # Remap edges involving this alias
                edges_remapped += self._remap_edges(alias_id, canonical_id)

                # Remove alias entity
                del self.entities[alias_id]
                if alias_id in self.graph:
                    self.graph.remove_node(alias_id)

            # Update canonical with merged aliases
            canonical_data["aliases"] = list(existing_aliases)
            self.graph.nodes[canonical_id]["aliases"] = str(list(existing_aliases))

        return {
            "original_count": original_count,
            "final_count": len(self.entities),
            "merged_groups": merged_groups,
            "edges_remapped": edges_remapped,
        }

    def deduplicate_by_ncbi_id(self) -> dict[str, Any]:
        """Merge entities that share the same NCBI ID (entity_ncbi_id).

        This is particularly useful for genes/proteins where different names
        refer to the same entity. For example:
        - "amyloid beta", "APP", "amyloid precursor protein" all have NCBI ID 351
        - "Tau", "tau", "MAPT" all have NCBI ID 4137

        The canonical entity is chosen by:
        1. Prefer official gene symbols (uppercase, short names like "APP", "MAPT")
        2. Otherwise pick the shortest name

        Returns:
            Dictionary with deduplication statistics:
                - original_count: Entity count before deduplication
                - final_count: Entity count after deduplication
                - merged_groups: List of merged entity groups
                - edges_remapped: Number of edges remapped to canonical IDs
        """
        # Group entities by NCBI ID (skip empty/invalid IDs)
        ncbi_groups: dict[str, list[str]] = {}
        invalid_ids = {"", "-", "None", "null", None}

        for entity_id, data in self.entities.items():
            ncbi_id = data.get("entity_ncbi_id")
            if ncbi_id and str(ncbi_id) not in invalid_ids:
                ncbi_key = str(ncbi_id)
                if ncbi_key not in ncbi_groups:
                    ncbi_groups[ncbi_key] = []
                ncbi_groups[ncbi_key].append(entity_id)

        original_count = len(self.entities)
        merged_groups: list[dict[str, Any]] = []
        edges_remapped = 0

        # Process each group that has more than one entity
        for ncbi_id, entity_ids in ncbi_groups.items():
            if len(entity_ids) <= 1:
                continue

            # Find canonical entity
            # Priority: official gene symbols (uppercase, 2-6 chars) > shortest name
            def score_entity(eid: str) -> tuple[int, int, str]:
                # Lower score = better canonical candidate
                is_official = eid.isupper() and 2 <= len(eid) <= 6
                return (0 if is_official else 1, len(eid), eid.lower())

            sorted_entities = sorted(entity_ids, key=score_entity)
            canonical_id = sorted_entities[0]
            aliases_to_merge = sorted_entities[1:]

            merged_groups.append({
                "ncbi_id": ncbi_id,
                "canonical": canonical_id,
                "merged": aliases_to_merge,
            })

            # Merge aliases into canonical entity
            canonical_data = self.entities[canonical_id]
            existing_aliases = set(canonical_data.get("aliases", []))
            if isinstance(existing_aliases, str):
                try:
                    import json
                    existing_aliases = set(json.loads(existing_aliases))
                except Exception:
                    existing_aliases = set()

            for alias_id in aliases_to_merge:
                # Add alias ID and its aliases to canonical
                existing_aliases.add(alias_id)
                alias_data = self.entities.get(alias_id, {})
                alias_aliases = alias_data.get("aliases", [])
                if isinstance(alias_aliases, str):
                    try:
                        import json
                        alias_aliases = json.loads(alias_aliases)
                    except Exception:
                        alias_aliases = []
                for a in alias_aliases:
                    existing_aliases.add(a)

                # Remap edges involving this alias
                edges_remapped += self._remap_edges(alias_id, canonical_id)

                # Remove alias entity
                if alias_id in self.entities:
                    del self.entities[alias_id]
                if alias_id in self.graph:
                    self.graph.remove_node(alias_id)

            # Update canonical with merged aliases
            canonical_data["aliases"] = list(existing_aliases)
            if canonical_id in self.graph.nodes:
                self.graph.nodes[canonical_id]["aliases"] = str(list(existing_aliases))

        return {
            "original_count": original_count,
            "final_count": len(self.entities),
            "merged_groups": merged_groups,
            "edges_remapped": edges_remapped,
        }

    def deduplicate_by_case_insensitive(self) -> dict[str, Any]:
        """Merge entities that differ only by case or minor punctuation.

        Handles cases like:
        - "cholesterol" vs "Cholesterol"
        - "Alzheimer disease" vs "Alzheimer's disease"

        The canonical entity is chosen by preferring:
        1. Title case or proper capitalization
        2. The more common form (by edge count)

        Returns:
            Dictionary with deduplication statistics.
        """
        # Build normalized -> original mapping
        normalized_groups: dict[str, list[str]] = {}

        def normalize(s: str) -> str:
            # Normalize: lowercase, remove apostrophes, normalize spaces
            return s.lower().replace("'", "").replace("-", " ").strip()

        for entity_id in self.entities.keys():
            norm = normalize(entity_id)
            if norm not in normalized_groups:
                normalized_groups[norm] = []
            normalized_groups[norm].append(entity_id)

        original_count = len(self.entities)
        merged_groups: list[dict[str, Any]] = []
        edges_remapped = 0

        for norm_key, entity_ids in normalized_groups.items():
            if len(entity_ids) <= 1:
                continue

            # Choose canonical: prefer title case, then by edge count
            def score_entity(eid: str) -> tuple[int, int, str]:
                # Count edges for this entity
                edge_count = sum(
                    1 for (s, t, _) in self.relationships.keys()
                    if s == eid or t == eid
                )
                is_title = eid[0].isupper() if eid else False
                # Lower score = better (prefer title case, more edges)
                return (0 if is_title else 1, -edge_count, eid)

            sorted_entities = sorted(entity_ids, key=score_entity)
            canonical_id = sorted_entities[0]
            aliases_to_merge = sorted_entities[1:]

            merged_groups.append({
                "normalized": norm_key,
                "canonical": canonical_id,
                "merged": aliases_to_merge,
            })

            # Merge aliases into canonical entity
            canonical_data = self.entities[canonical_id]
            existing_aliases = set(canonical_data.get("aliases", []))
            if isinstance(existing_aliases, str):
                try:
                    import json
                    existing_aliases = set(json.loads(existing_aliases))
                except Exception:
                    existing_aliases = set()

            for alias_id in aliases_to_merge:
                existing_aliases.add(alias_id)
                alias_data = self.entities.get(alias_id, {})
                alias_aliases = alias_data.get("aliases", [])
                if isinstance(alias_aliases, str):
                    try:
                        import json
                        alias_aliases = json.loads(alias_aliases)
                    except Exception:
                        alias_aliases = []
                for a in alias_aliases:
                    existing_aliases.add(a)

                edges_remapped += self._remap_edges(alias_id, canonical_id)

                if alias_id in self.entities:
                    del self.entities[alias_id]
                if alias_id in self.graph:
                    self.graph.remove_node(alias_id)

            canonical_data["aliases"] = list(existing_aliases)
            if canonical_id in self.graph.nodes:
                self.graph.nodes[canonical_id]["aliases"] = str(list(existing_aliases))

        return {
            "original_count": original_count,
            "final_count": len(self.entities),
            "merged_groups": merged_groups,
            "edges_remapped": edges_remapped,
        }

    def _remap_edges(self, old_id: str, new_id: str) -> int:
        """Remap all edges from old_id to new_id.

        Args:
            old_id: Entity ID to replace.
            new_id: Entity ID to use instead.

        Returns:
            Number of edges remapped.
        """
        remapped = 0
        relationships_to_update: list[tuple[tuple[str, str, str], dict[str, Any]]] = []
        relationships_to_remove: list[tuple[str, str, str]] = []

        for rel_key, data in list(self.relationships.items()):
            src, tgt, rel_type = rel_key
            new_src, new_tgt = src, tgt

            if src == old_id:
                new_src = new_id
            if tgt == old_id:
                new_tgt = new_id

            if new_src != src or new_tgt != tgt:
                relationships_to_remove.append(rel_key)
                new_key = (new_src, new_tgt, rel_type)

                # Update data
                new_data = data.copy()
                new_data["source"] = new_src
                new_data["target"] = new_tgt
                relationships_to_update.append((new_key, new_data))
                remapped += 1

        # Apply updates
        for old_key in relationships_to_remove:
            del self.relationships[old_key]

        for new_key, new_data in relationships_to_update:
            if new_key in self.relationships:
                # Merge with existing relationship
                existing = self.relationships[new_key]
                existing_evidence = existing.get("evidence", [])
                new_evidence = new_data.get("evidence", [])
                existing_ids = {e.get("source_id") for e in existing_evidence}
                for ev in new_evidence:
                    if ev.get("source_id") not in existing_ids:
                        existing_evidence.append(ev)
                existing["evidence"] = existing_evidence
            else:
                self.relationships[new_key] = new_data

        # Update NetworkX graph edges (node removal handles this)
        return remapped

    def get_grounding_summary(self) -> dict[str, Any]:
        """Get summary of entity grounding status.

        Returns:
            Dictionary with grounding statistics.
        """
        total = 0
        grounded = 0
        hgnc_grounded = 0
        doid_grounded = 0
        mesh_grounded = 0
        by_db: dict[str, int] = {}

        # Include proteins, genes, and diseases for grounding
        groundable_types = {
            EntityType.PROTEIN.value,
            EntityType.GENE.value,
            EntityType.DISEASE.value,
        }

        for entity_id, data in self.entities.items():
            if data.get("type") not in groundable_types:
                continue

            total += 1
            db = data.get("grounded_db")
            if db:
                grounded += 1
                by_db[db] = by_db.get(db, 0) + 1
                if db == "HGNC":
                    hgnc_grounded += 1
                elif db == "DOID":
                    doid_grounded += 1
                elif db == "MESH":
                    mesh_grounded += 1

        return {
            "total_groundable": total,
            "grounded": grounded,
            "ungrounded": total - grounded,
            "hgnc_grounded": hgnc_grounded,
            "doid_grounded": doid_grounded,
            "mesh_grounded": mesh_grounded,
            "by_database": by_db,
        }
