"""
Pipeline data models for the hybrid KG architecture.

This module defines the data structures used to pass data between
the DataFetchAgent (LLM-driven) and KGPipeline (deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineInput:
    """
    Input data for the deterministic KG pipeline.

    Collected by the DataFetchAgent using LLM reasoning for query construction,
    then passed to the KGPipeline for deterministic processing.

    Attributes:
        articles: List of article dicts from PubMed with keys:
            - pmid: PubMed ID
            - title: Article title
            - abstract: Article abstract text
            - year: Publication year (optional)
            - journal: Journal name (optional)
            - authors: List of author names (optional)

        annotations: Dict mapping PMID to list of NER annotations from PubTator.
            Each annotation has keys:
            - entity_id: Normalized entity identifier
            - entity_text: Entity text as mentioned in article
            - entity_type: Entity type (Gene, Disease, Chemical, etc.)
            - pmid: PubMed ID

        string_interactions: List of protein-protein interactions from STRING.
            Each interaction has keys:
            - stringId_A, stringId_B: STRING identifiers
            - preferredName_A, preferredName_B: Common protein names
            - score: Combined interaction score (0-1)
            - nscore, fscore, pscore, ascore, escore, dscore, tscore: Individual scores

        seed_proteins: Original seed proteins used for STRING query.

        pubmed_query: The PubMed query used to search for articles.

        metadata: Additional metadata about the data collection.
    """

    articles: list[dict[str, Any]] = field(default_factory=list)
    annotations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    string_interactions: list[dict[str, Any]] = field(default_factory=list)
    seed_proteins: list[str] = field(default_factory=list)
    pubmed_query: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def article_count(self) -> int:
        """Number of articles in the input."""
        return len(self.articles)

    @property
    def interaction_count(self) -> int:
        """Number of STRING interactions in the input."""
        return len(self.string_interactions)

    @property
    def annotation_count(self) -> int:
        """Total number of NER annotations across all articles."""
        return sum(len(anns) for anns in self.annotations.values())

    @property
    def unique_proteins(self) -> set[str]:
        """Set of unique protein names from STRING interactions."""
        proteins = set()
        for interaction in self.string_interactions:
            proteins.add(interaction.get("preferredName_A", ""))
            proteins.add(interaction.get("preferredName_B", ""))
        proteins.discard("")
        return proteins

    @property
    def is_empty(self) -> bool:
        """Check if the input has no data."""
        return (
            len(self.articles) == 0 and
            len(self.string_interactions) == 0 and
            len(self.annotations) == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "articles": self.articles,
            "annotations": self.annotations,
            "string_interactions": self.string_interactions,
            "seed_proteins": self.seed_proteins,
            "pubmed_query": self.pubmed_query,
            "metadata": self.metadata,
            "summary": {
                "article_count": self.article_count,
                "interaction_count": self.interaction_count,
                "annotation_count": self.annotation_count,
                "unique_proteins": len(self.unique_proteins),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineInput":
        """Create from dictionary."""
        return cls(
            articles=data.get("articles", []),
            annotations=data.get("annotations", {}),
            string_interactions=data.get("string_interactions", []),
            seed_proteins=data.get("seed_proteins", []),
            pubmed_query=data.get("pubmed_query", ""),
            metadata=data.get("metadata", {}),
        )

    def merge(self, other: "PipelineInput") -> "PipelineInput":
        """
        Merge another PipelineInput into this one.

        Used for graph expansion - adds new data while avoiding duplicates.

        Args:
            other: Another PipelineInput to merge.

        Returns:
            New PipelineInput with merged data.
        """
        # Merge articles by PMID (avoid duplicates)
        existing_pmids = {a.get("pmid") for a in self.articles}
        merged_articles = self.articles.copy()
        for article in other.articles:
            if article.get("pmid") not in existing_pmids:
                merged_articles.append(article)

        # Merge annotations (add new ones per PMID)
        merged_annotations = {**self.annotations}
        for pmid, anns in other.annotations.items():
            if pmid not in merged_annotations:
                merged_annotations[pmid] = anns
            else:
                # Add new annotations for existing PMID
                existing_texts = {a.get("entity_text") for a in merged_annotations[pmid]}
                for ann in anns:
                    if ann.get("entity_text") not in existing_texts:
                        merged_annotations[pmid].append(ann)

        # Merge STRING interactions (by protein pair)
        existing_pairs = {
            (i.get("preferredName_A", ""), i.get("preferredName_B", ""))
            for i in self.string_interactions
        }
        merged_interactions = self.string_interactions.copy()
        for interaction in other.string_interactions:
            pair = (
                interaction.get("preferredName_A", ""),
                interaction.get("preferredName_B", "")
            )
            reverse_pair = (pair[1], pair[0])
            if pair not in existing_pairs and reverse_pair not in existing_pairs:
                merged_interactions.append(interaction)

        # Merge seed proteins
        merged_seeds = list(set(self.seed_proteins) | set(other.seed_proteins))

        # Combine pubmed queries
        merged_query = self.pubmed_query
        if other.pubmed_query and other.pubmed_query != self.pubmed_query:
            merged_query = f"({self.pubmed_query}) OR ({other.pubmed_query})"

        # Merge metadata
        merged_metadata = {**self.metadata, **other.metadata}

        return PipelineInput(
            articles=merged_articles,
            annotations=merged_annotations,
            string_interactions=merged_interactions,
            seed_proteins=merged_seeds,
            pubmed_query=merged_query,
            metadata=merged_metadata,
        )


@dataclass
class GraphStatistics:
    """Statistics for a knowledge graph (before/after pruning)."""

    node_count: int = 0
    edge_count: int = 0
    component_count: int = 0
    relationship_types: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "component_count": self.component_count,
            "relationship_types": self.relationship_types,
        }


@dataclass
class PipelineResult:
    """
    Result from the KGPipeline processing.

    Contains the built knowledge graph and statistics about the processing.
    """

    graph: Any  # KnowledgeGraph - avoid circular import
    relationships_extracted: int = 0
    relationships_valid: int = 0
    entities_found: int = 0
    processing_time_ms: float = 0.0
    mining_statistics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    # Before/after pruning statistics
    stats_before_pruning: GraphStatistics = field(default_factory=GraphStatistics)
    stats_after_pruning: GraphStatistics = field(default_factory=GraphStatistics)
    nodes_pruned: int = 0
    edges_pruned: int = 0

    @property
    def success(self) -> bool:
        """Check if processing was successful."""
        return self.graph is not None and len(self.errors) == 0

    def to_summary(self) -> dict[str, Any]:
        """Get a summary of the result."""
        graph_summary = self.graph.to_summary() if self.graph else {}
        return {
            "success": self.success,
            "graph_summary": graph_summary,
            "relationships_extracted": self.relationships_extracted,
            "relationships_valid": self.relationships_valid,
            "entities_found": self.entities_found,
            "processing_time_ms": self.processing_time_ms,
            "mining_statistics": self.mining_statistics,
            "errors": self.errors,
            # Pruning statistics
            "before_pruning": self.stats_before_pruning.to_dict(),
            "after_pruning": self.stats_after_pruning.to_dict(),
            "nodes_pruned": self.nodes_pruned,
            "edges_pruned": self.edges_pruned,
        }
