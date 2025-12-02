"""
Graph merge module for unifying all data sources into a knowledge graph.

This module is Phase 5 of the knowledge graph pipeline. It takes outputs from:
- Phase 1: STRING database expansion (protein interactions)
- Phase 4A: Co-occurrence extraction (entity co-mentions)
- Phase 4B: LLM relationship extraction (semantic relationships)

And merges them into a unified KnowledgeGraph with:
- Deduplicated edges with aggregated evidence
- Confidence scores based on source reliability and evidence count
- Proper entity typing and relationship classification

Example:
    >>> from pipeline import PipelineConfig, PipelineReport
    >>> from pipeline.merge import merge_to_knowledge_graph
    >>>
    >>> result = await merge_to_knowledge_graph(
    ...     string_result=string_expansion_result,
    ...     extraction_result=extraction_result,
    ...     pubmed_articles=articles,
    ...     config=config,
    ...     report=report,
    ... )
    >>> print(f"Created graph with {result.nodes_created} nodes and {result.edges_created} edges")
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from models.knowledge_graph import (
    EvidenceSource,
    KnowledgeGraph,
    RelationshipType,
)
from pipeline.config import PipelineConfig
from pipeline.metrics import PhaseMetrics, PipelineReport
from pipeline.parallel_extraction import CoOccurrenceEdge, ExtractionResult
from pipeline.string_expansion import STRINGExpansionResult

logger = logging.getLogger(__name__)


# =============================================================================
# Constants for Confidence Scoring
# =============================================================================

# Base confidence weights by source type
SOURCE_WEIGHTS: dict[str, float] = {
    "STRING": 0.9,       # High confidence - curated database
    "llm": 0.7,          # Medium-high - semantic extraction with evidence
    "co-occurrence": 0.5,  # Medium - statistical association
}

# Minimum STRING score (0-1) to consider high confidence
STRING_HIGH_CONFIDENCE_THRESHOLD: float = 0.7

# Minimum PMI score to consider significant co-occurrence
PMI_SIGNIFICANCE_THRESHOLD: float = 2.0

# Boost factor for multi-source evidence
MULTI_SOURCE_BOOST: float = 0.15

# Maximum confidence score
MAX_CONFIDENCE: float = 1.0


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MergeResult:
    """
    Result of merging all data sources into a knowledge graph.

    Contains the unified graph along with statistics about
    the merge operation including node/edge counts and deduplication.

    Attributes:
        graph: The merged KnowledgeGraph instance.
        nodes_created: Total number of nodes created.
        edges_created: Total number of unique edges created.
        edges_deduplicated: Number of duplicate edges that were merged.
        evidence_aggregated: Total pieces of evidence aggregated.
        phase_metrics: Optional phase metrics for this operation.
    """

    graph: KnowledgeGraph
    nodes_created: int
    edges_created: int
    edges_deduplicated: int
    evidence_aggregated: int
    phase_metrics: PhaseMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nodes_created": self.nodes_created,
            "edges_created": self.edges_created,
            "edges_deduplicated": self.edges_deduplicated,
            "evidence_aggregated": self.evidence_aggregated,
            "graph_summary": self.graph.to_summary(),
            "phase_metrics": (
                self.phase_metrics.to_dict() if self.phase_metrics else None
            ),
        }


@dataclass
class EdgeCandidate:
    """
    Intermediate representation of an edge before deduplication.

    Used to collect all edges from different sources before
    merging duplicates and calculating final confidence scores.

    Attributes:
        source: Source node ID.
        target: Target node ID.
        relation_type: Type of relationship.
        data_source: Origin of this edge ("STRING", "llm", "co-occurrence").
        confidence: Initial confidence score from source.
        evidence_texts: List of evidence text snippets.
        pmids: List of PubMed IDs supporting this edge.
        string_score: STRING confidence score if applicable.
        pmi_score: PMI score if from co-occurrence.
        raw_count: Raw co-occurrence count if applicable.
        metadata: Additional source-specific metadata.
    """

    source: str
    target: str
    relation_type: str
    data_source: str
    confidence: float = 0.0
    evidence_texts: list[str] = field(default_factory=list)
    pmids: list[str] = field(default_factory=list)
    string_score: float | None = None
    pmi_score: float | None = None
    raw_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Confidence Scoring
# =============================================================================


def calculate_edge_confidence(
    sources: list[str],
    string_score: float | None = None,
    pmi_score: float | None = None,
    evidence_count: int = 0,
) -> float:
    """
    Calculate confidence score (0-1) for an edge based on evidence.

    The confidence score is calculated using a weighted combination of:
    - Source reliability (STRING > LLM > co-occurrence)
    - STRING confidence score (for STRING edges)
    - PMI significance (for co-occurrence edges)
    - Evidence count (more evidence = higher confidence)
    - Multi-source boost (same edge from multiple sources)

    Weights:
    - STRING with high confidence (>0.7): 0.9 base
    - LLM extraction with evidence: 0.7 base
    - Co-occurrence with PMI > 2: 0.5 base
    - Multiple sources add 0.15 boost

    Args:
        sources: List of data sources for this edge (e.g., ["STRING", "llm"]).
        string_score: STRING confidence score (0-1 scale) if applicable.
        pmi_score: Pointwise Mutual Information score if from co-occurrence.
        evidence_count: Number of evidence items supporting this edge.

    Returns:
        Confidence score between 0 and 1.

    Examples:
        >>> calculate_edge_confidence(["STRING"], string_score=0.9)
        0.9
        >>> calculate_edge_confidence(["STRING", "llm"], string_score=0.8, evidence_count=3)
        0.96  # Boosted for multi-source
    """
    if not sources:
        return 0.0

    # Calculate base confidence from best source
    base_scores: list[float] = []

    for source in sources:
        if source == "STRING":
            if string_score is not None:
                # Scale STRING score (already 0-1)
                score = SOURCE_WEIGHTS["STRING"] * min(string_score, 1.0)
            else:
                score = SOURCE_WEIGHTS["STRING"] * 0.7  # Default moderate confidence
            base_scores.append(score)

        elif source == "llm":
            # LLM confidence is already factored in, use base weight
            base_scores.append(SOURCE_WEIGHTS["llm"])

        elif source == "co-occurrence":
            if pmi_score is not None and pmi_score > PMI_SIGNIFICANCE_THRESHOLD:
                # PMI > 2 indicates significant association
                pmi_factor = min(pmi_score / 5.0, 1.0)  # Normalize to ~0-1
                score = SOURCE_WEIGHTS["co-occurrence"] * (0.5 + 0.5 * pmi_factor)
            else:
                score = SOURCE_WEIGHTS["co-occurrence"] * 0.5  # Low base for weak PMI
            base_scores.append(score)

    if not base_scores:
        return 0.0

    # Use maximum base score (best evidence)
    confidence = max(base_scores)

    # Apply multi-source boost if evidence from multiple sources
    unique_sources = set(sources)
    if len(unique_sources) > 1:
        confidence += MULTI_SOURCE_BOOST * (len(unique_sources) - 1)

    # Apply evidence count boost (diminishing returns)
    if evidence_count > 1:
        evidence_boost = min(0.1 * (evidence_count - 1), 0.2)
        confidence += evidence_boost

    # Clamp to [0, 1]
    return min(confidence, MAX_CONFIDENCE)


# =============================================================================
# Edge Deduplication
# =============================================================================


def _normalize_entity_id(entity_id: str) -> str:
    """
    Normalize an entity identifier for consistent deduplication.

    Args:
        entity_id: Raw entity identifier.

    Returns:
        Normalized identifier (uppercase, stripped).
    """
    return entity_id.strip().upper()


def _create_edge_key(source: str, target: str, relation_type: str) -> tuple[str, str, str]:
    """
    Create a canonical key for edge deduplication.

    Ensures consistent ordering for undirected relationships while
    preserving directionality for directed relationships.

    Args:
        source: Source node ID.
        target: Target node ID.
        relation_type: Relationship type.

    Returns:
        Tuple key (normalized_source, normalized_target, relation_type).
    """
    norm_source = _normalize_entity_id(source)
    norm_target = _normalize_entity_id(target)

    # For symmetric relationships, sort to ensure consistent key
    symmetric_relations = {
        "interacts_with",
        "cooccurs_with",
        "associated_with",
        "binds_to",
    }

    if relation_type in symmetric_relations:
        if norm_source > norm_target:
            norm_source, norm_target = norm_target, norm_source

    return (norm_source, norm_target, relation_type)


def deduplicate_edges(
    edges: list[EdgeCandidate],
) -> list[dict[str, Any]]:
    """
    Deduplicate edges by (source, target, relation_type).

    When duplicates are found:
    - Aggregates all evidence texts
    - Keeps all unique PMIDs
    - Collects all data sources
    - Recalculates confidence based on combined evidence
    - Preserves best STRING/PMI scores

    Args:
        edges: List of EdgeCandidate objects to deduplicate.

    Returns:
        List of deduplicated edge dictionaries ready for graph insertion.

    Example:
        >>> edges = [
        ...     EdgeCandidate("A", "B", "interacts_with", "STRING", string_score=0.9),
        ...     EdgeCandidate("A", "B", "interacts_with", "llm", evidence_texts=["A binds B"]),
        ... ]
        >>> result = deduplicate_edges(edges)
        >>> len(result)
        1
        >>> result[0]["data_sources"]
        ["STRING", "llm"]
    """
    # Group edges by canonical key
    edge_groups: dict[tuple[str, str, str], list[EdgeCandidate]] = defaultdict(list)

    for edge in edges:
        key = _create_edge_key(edge.source, edge.target, edge.relation_type)
        edge_groups[key].append(edge)

    # Merge each group
    deduplicated: list[dict[str, Any]] = []
    total_duplicates = 0

    for (norm_source, norm_target, rel_type), group in edge_groups.items():
        if len(group) > 1:
            total_duplicates += len(group) - 1

        # Aggregate evidence
        all_sources: list[str] = []
        all_evidence: list[str] = []
        all_pmids: set[str] = set()
        best_string_score: float | None = None
        best_pmi_score: float | None = None
        best_raw_count: int | None = None

        for edge in group:
            all_sources.append(edge.data_source)
            all_evidence.extend(edge.evidence_texts)
            all_pmids.update(edge.pmids)

            if edge.string_score is not None:
                if best_string_score is None or edge.string_score > best_string_score:
                    best_string_score = edge.string_score

            if edge.pmi_score is not None:
                if best_pmi_score is None or edge.pmi_score > best_pmi_score:
                    best_pmi_score = edge.pmi_score

            if edge.raw_count is not None:
                if best_raw_count is None or edge.raw_count > best_raw_count:
                    best_raw_count = edge.raw_count

        # Remove duplicate evidence texts
        unique_evidence = list(dict.fromkeys(all_evidence))
        unique_sources = list(dict.fromkeys(all_sources))

        # Calculate combined confidence
        confidence = calculate_edge_confidence(
            sources=all_sources,
            string_score=best_string_score,
            pmi_score=best_pmi_score,
            evidence_count=len(unique_evidence),
        )

        # Use original case from first edge
        original_source = group[0].source
        original_target = group[0].target

        merged_edge = {
            "source": original_source,
            "target": original_target,
            "relation_type": rel_type,
            "confidence": confidence,
            "data_sources": unique_sources,
            "evidence": unique_evidence,
            "pmids": sorted(all_pmids),
            "string_score": best_string_score,
            "pmi_score": best_pmi_score,
        }

        if best_raw_count is not None:
            merged_edge["raw_count"] = best_raw_count

        # Mark as multi-source if from different data sources
        if len(unique_sources) > 1:
            merged_edge["is_multi_source"] = True

        deduplicated.append(merged_edge)

    logger.debug(
        "Deduplicated %d edges into %d unique edges (%d duplicates merged)",
        len(edges),
        len(deduplicated),
        total_duplicates,
    )

    return deduplicated


# =============================================================================
# Relationship Type Mapping
# =============================================================================


def _map_relationship_type(rel_string: str) -> RelationshipType:
    """
    Map a relationship string to RelationshipType enum.

    Handles various naming conventions and synonyms.

    Args:
        rel_string: Relationship type string from LLM or other source.

    Returns:
        Corresponding RelationshipType enum value.
    """
    normalized = rel_string.lower().strip().replace("-", "_").replace(" ", "_")

    mapping: dict[str, RelationshipType] = {
        "activates": RelationshipType.ACTIVATES,
        "activate": RelationshipType.ACTIVATES,
        "activation": RelationshipType.ACTIVATES,
        "stimulates": RelationshipType.ACTIVATES,
        "upregulates": RelationshipType.ACTIVATES,
        "up_regulates": RelationshipType.ACTIVATES,
        "increases": RelationshipType.ACTIVATES,
        "inhibits": RelationshipType.INHIBITS,
        "inhibit": RelationshipType.INHIBITS,
        "inhibition": RelationshipType.INHIBITS,
        "suppresses": RelationshipType.INHIBITS,
        "downregulates": RelationshipType.INHIBITS,
        "down_regulates": RelationshipType.INHIBITS,
        "decreases": RelationshipType.INHIBITS,
        "blocks": RelationshipType.INHIBITS,
        "associated_with": RelationshipType.ASSOCIATED_WITH,
        "associates_with": RelationshipType.ASSOCIATED_WITH,
        "association": RelationshipType.ASSOCIATED_WITH,
        "related_to": RelationshipType.ASSOCIATED_WITH,
        "linked_to": RelationshipType.ASSOCIATED_WITH,
        "regulates": RelationshipType.REGULATES,
        "regulate": RelationshipType.REGULATES,
        "regulation": RelationshipType.REGULATES,
        "modulates": RelationshipType.REGULATES,
        "controls": RelationshipType.REGULATES,
        "binds_to": RelationshipType.BINDS_TO,
        "binds": RelationshipType.BINDS_TO,
        "binding": RelationshipType.BINDS_TO,
        "interacts_with": RelationshipType.INTERACTS_WITH,
        "interacts": RelationshipType.INTERACTS_WITH,
        "interaction": RelationshipType.INTERACTS_WITH,
        "cooccurs_with": RelationshipType.COOCCURS_WITH,
        "co_occurs_with": RelationshipType.COOCCURS_WITH,
        "cooccurrence": RelationshipType.COOCCURS_WITH,
        "hypothesized": RelationshipType.HYPOTHESIZED,
        "predicted": RelationshipType.HYPOTHESIZED,
    }

    return mapping.get(normalized, RelationshipType.ASSOCIATED_WITH)


def _map_entity_type(type_string: str) -> str:
    """
    Normalize entity type string to standard format.

    Args:
        type_string: Raw entity type from annotations.

    Returns:
        Normalized entity type string.
    """
    normalized = type_string.lower().strip()

    mapping: dict[str, str] = {
        "gene": "gene",
        "protein": "protein",
        "disease": "disease",
        "chemical": "chemical",
        "drug": "chemical",
        "compound": "chemical",
        "species": "species",
        "organism": "species",
        "cellline": "cell_line",
        "cell_line": "cell_line",
        "mutation": "mutation",
        "variant": "mutation",
        "snp": "mutation",
        "pathway": "pathway",
        "biological_process": "pathway",
        "molecular_function": "pathway",
        "cellular_component": "pathway",
    }

    return mapping.get(normalized, normalized or "unknown")


# =============================================================================
# Edge Extraction from Sources
# =============================================================================


def _extract_string_edges(
    string_result: STRINGExpansionResult,
) -> list[EdgeCandidate]:
    """
    Extract edge candidates from STRING interactions.

    Args:
        string_result: STRING expansion result with interactions.

    Returns:
        List of EdgeCandidate objects from STRING data.
    """
    edges: list[EdgeCandidate] = []

    for interaction in string_result.interactions:
        name_a = interaction.get("preferredName_A", "")
        name_b = interaction.get("preferredName_B", "")
        score = interaction.get("score", 0.0)

        if not name_a or not name_b:
            continue

        # STRING scores are already 0-1 (converted from 0-1000 by client)
        # If raw score is > 1, normalize it
        if score > 1:
            score = score / 1000.0

        edge = EdgeCandidate(
            source=name_a,
            target=name_b,
            relation_type="interacts_with",
            data_source="STRING",
            confidence=SOURCE_WEIGHTS["STRING"] * score,
            string_score=score,
        )
        edges.append(edge)

    logger.debug("Extracted %d edges from STRING interactions", len(edges))
    return edges


def _extract_cooccurrence_edges(
    cooccurrence_edges: list[CoOccurrenceEdge],
) -> list[EdgeCandidate]:
    """
    Extract edge candidates from co-occurrence data.

    Args:
        cooccurrence_edges: List of CoOccurrenceEdge objects.

    Returns:
        List of EdgeCandidate objects from co-occurrence data.
    """
    edges: list[EdgeCandidate] = []

    for cooc in cooccurrence_edges:
        edge = EdgeCandidate(
            source=cooc.source,
            target=cooc.target,
            relation_type="cooccurs_with",
            data_source="co-occurrence",
            pmids=list(cooc.pmids) if cooc.pmids else [],
            pmi_score=cooc.pmi_score,
            raw_count=cooc.raw_count,
            metadata={
                "source_type": cooc.source_type,
                "target_type": cooc.target_type,
                "weight": cooc.weight,
            },
        )
        edges.append(edge)

    logger.debug("Extracted %d edges from co-occurrence data", len(edges))
    return edges


def _extract_llm_edges(
    llm_relationships: list[dict[str, Any]],
) -> list[EdgeCandidate]:
    """
    Extract edge candidates from LLM-extracted relationships.

    Args:
        llm_relationships: List of relationship dicts from LLM extraction.

    Returns:
        List of EdgeCandidate objects from LLM data.
    """
    edges: list[EdgeCandidate] = []

    for rel in llm_relationships:
        entity1 = rel.get("entity1", "")
        entity2 = rel.get("entity2", "")
        relationship = rel.get("relationship", "associated_with")
        confidence = rel.get("confidence", 0.5)
        pmid = rel.get("pmid", "")
        evidence_text = rel.get("evidence", "") or rel.get("sentence", "")

        if not entity1 or not entity2:
            continue

        edge = EdgeCandidate(
            source=entity1,
            target=entity2,
            relation_type=relationship,
            data_source="llm",
            confidence=SOURCE_WEIGHTS["llm"] * confidence,
            evidence_texts=[evidence_text] if evidence_text else [],
            pmids=[pmid] if pmid else [],
            metadata={
                "original_confidence": confidence,
            },
        )
        edges.append(edge)

    logger.debug("Extracted %d edges from LLM relationships", len(edges))
    return edges


# =============================================================================
# Node Extraction
# =============================================================================


def _extract_all_entities(
    string_result: STRINGExpansionResult,
    extraction_result: ExtractionResult,
    pubmed_annotations: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """
    Extract all unique entities from all data sources.

    Args:
        string_result: STRING expansion result.
        extraction_result: Parallel extraction result.
        pubmed_annotations: PubTator annotations keyed by PMID.

    Returns:
        Dictionary mapping entity_id to entity metadata.
    """
    entities: dict[str, dict[str, Any]] = {}

    # Add STRING proteins
    for protein in string_result.expanded_proteins:
        norm_id = _normalize_entity_id(protein)
        if norm_id not in entities:
            entities[norm_id] = {
                "name": protein,
                "type": "protein",
                "sources": ["STRING"],
            }
        else:
            if "STRING" not in entities[norm_id]["sources"]:
                entities[norm_id]["sources"].append("STRING")

    # Add entities from annotations
    for pmid, annotations in pubmed_annotations.items():
        for annot in annotations:
            entity_text = annot.get("entity_text", "")
            entity_id = annot.get("entity_id", entity_text)
            entity_type = annot.get("entity_type", "unknown")

            if not entity_text:
                continue

            norm_id = _normalize_entity_id(entity_id or entity_text)

            if norm_id not in entities:
                entities[norm_id] = {
                    "name": entity_text,
                    "type": _map_entity_type(entity_type),
                    "sources": ["PubTator"],
                }
            else:
                if "PubTator" not in entities[norm_id]["sources"]:
                    entities[norm_id]["sources"].append("PubTator")

    # Add entities from co-occurrence edges
    for edge in extraction_result.cooccurrence_edges:
        for entity_id, entity_type in [
            (edge.source, edge.source_type),
            (edge.target, edge.target_type),
        ]:
            norm_id = _normalize_entity_id(entity_id)
            if norm_id not in entities:
                entities[norm_id] = {
                    "name": entity_id,
                    "type": _map_entity_type(entity_type) if entity_type else "unknown",
                    "sources": ["co-occurrence"],
                }

    # Add entities from LLM relationships
    for rel in extraction_result.llm_relationships:
        for entity_key in ["entity1", "entity2"]:
            entity_text = rel.get(entity_key, "")
            if entity_text:
                norm_id = _normalize_entity_id(entity_text)
                if norm_id not in entities:
                    entities[norm_id] = {
                        "name": entity_text,
                        "type": "unknown",
                        "sources": ["llm"],
                    }
                else:
                    if "llm" not in entities[norm_id]["sources"]:
                        entities[norm_id]["sources"].append("llm")

    logger.debug("Extracted %d unique entities from all sources", len(entities))
    return entities


# =============================================================================
# Main Merge Function
# =============================================================================


async def merge_to_knowledge_graph(
    string_result: STRINGExpansionResult,
    extraction_result: ExtractionResult,
    pubmed_articles: list[dict[str, Any]],
    config: PipelineConfig,
    report: PipelineReport,
    annotations: dict[str, list[dict[str, Any]]] | None = None,
) -> MergeResult:
    """
    Merge all data sources into a unified knowledge graph.

    This is Phase 5 of the pipeline. It combines:
    - STRING protein interactions (high-confidence database edges)
    - Co-occurrence edges (statistical associations from abstracts)
    - LLM-extracted relationships (semantic relationships with evidence)

    The merge process:
    1. Extract all unique entities from all sources
    2. Add STRING entities as PROTEIN nodes
    3. Add STRING interactions as edges (source: "STRING")
    4. Add PubTator entities from annotations
    5. Add co-occurrence edges (source: "co-occurrence")
    6. Add LLM-extracted relationships (source: "llm")
    7. Deduplicate edges by (source, target, type), aggregating evidence
    8. Calculate confidence scores based on evidence and source reliability

    Args:
        string_result: Result from STRING network expansion (Phase 1).
        extraction_result: Result from parallel extraction (Phase 4A/4B).
        pubmed_articles: List of PubMed article dicts for evidence linking.
        config: Pipeline configuration.
        report: Pipeline report for metrics tracking.
        annotations: Optional PubTator annotations keyed by PMID.

    Returns:
        MergeResult with the unified knowledge graph and merge statistics.

    Example:
        >>> result = await merge_to_knowledge_graph(
        ...     string_result=expansion,
        ...     extraction_result=extraction,
        ...     pubmed_articles=articles,
        ...     config=PipelineConfig(),
        ...     report=report,
        ... )
        >>> print(f"Graph has {result.graph.graph.number_of_nodes()} nodes")
    """
    # Start phase tracking
    phase_name = "graph_merge"
    metrics = report.start_phase(phase_name)
    start_time = time.time()

    logger.info(
        "Starting graph merge: %d STRING proteins, %d interactions, "
        "%d co-occurrence edges, %d LLM relationships",
        len(string_result.expanded_proteins),
        len(string_result.interactions),
        len(extraction_result.cooccurrence_edges),
        len(extraction_result.llm_relationships),
    )

    # Initialize knowledge graph
    graph = KnowledgeGraph()

    # Build annotations dict from articles if not provided
    if annotations is None:
        annotations = {}

    # -------------------------------------------------------------------------
    # Step 1: Extract all entities
    # -------------------------------------------------------------------------
    all_entities = _extract_all_entities(
        string_result=string_result,
        extraction_result=extraction_result,
        pubmed_annotations=annotations,
    )

    # -------------------------------------------------------------------------
    # Step 2: Add entities as nodes
    # -------------------------------------------------------------------------
    nodes_created = 0
    for entity_id, entity_data in all_entities.items():
        graph.add_entity(
            entity_id=entity_id,
            entity_type=entity_data["type"],
            name=entity_data["name"],
            sources=entity_data["sources"],
        )
        nodes_created += 1

    logger.info("Added %d nodes to graph", nodes_created)

    # -------------------------------------------------------------------------
    # Step 3: Extract edges from all sources
    # -------------------------------------------------------------------------
    all_edge_candidates: list[EdgeCandidate] = []

    # STRING edges
    string_edges = _extract_string_edges(string_result)
    all_edge_candidates.extend(string_edges)

    # Co-occurrence edges
    cooc_edges = _extract_cooccurrence_edges(extraction_result.cooccurrence_edges)
    all_edge_candidates.extend(cooc_edges)

    # LLM edges
    llm_edges = _extract_llm_edges(extraction_result.llm_relationships)
    all_edge_candidates.extend(llm_edges)

    logger.info(
        "Collected %d total edge candidates: %d STRING, %d co-occurrence, %d LLM",
        len(all_edge_candidates),
        len(string_edges),
        len(cooc_edges),
        len(llm_edges),
    )

    # -------------------------------------------------------------------------
    # Step 4: Deduplicate edges
    # -------------------------------------------------------------------------
    edges_before = len(all_edge_candidates)
    deduplicated_edges = deduplicate_edges(all_edge_candidates)
    edges_deduplicated = edges_before - len(deduplicated_edges)

    # -------------------------------------------------------------------------
    # Step 5: Add edges to graph
    # -------------------------------------------------------------------------
    edges_created = 0
    evidence_aggregated = 0

    for edge_data in deduplicated_edges:
        source = edge_data["source"]
        target = edge_data["target"]
        rel_type = _map_relationship_type(edge_data["relation_type"])
        confidence = edge_data["confidence"]
        evidence_texts = edge_data.get("evidence", [])
        pmids = edge_data.get("pmids", [])
        data_sources = edge_data.get("data_sources", [])

        # Build evidence list for the KnowledgeGraph format
        evidence_list: list[dict[str, Any]] = []

        # Add evidence from PMIDs
        for pmid in pmids:
            evidence_list.append({
                "source_type": EvidenceSource.LITERATURE.value,
                "source_id": pmid,
                "confidence": confidence,
            })

        # Add STRING evidence
        if "STRING" in data_sources:
            evidence_list.append({
                "source_type": EvidenceSource.STRING.value,
                "source_id": "STRING_DB",
                "confidence": edge_data.get("string_score", 0.7),
            })

        # Add evidence texts as snippets
        for i, text in enumerate(evidence_texts):
            if text and i < len(evidence_list):
                evidence_list[i]["text_snippet"] = text

        evidence_aggregated += len(evidence_list)

        # Ensure source and target nodes exist (normalize IDs)
        norm_source = _normalize_entity_id(source)
        norm_target = _normalize_entity_id(target)

        # Add the relationship
        graph.add_relationship(
            source=norm_source,
            target=norm_target,
            rel_type=rel_type,
            evidence=evidence_list if evidence_list else None,
            ml_score=confidence,
            data_sources=data_sources,
            string_score=edge_data.get("string_score"),
            pmi_score=edge_data.get("pmi_score"),
            is_multi_source=edge_data.get("is_multi_source", False),
        )
        edges_created += 1

    logger.info(
        "Added %d edges to graph (%d deduplicated, %d evidence items)",
        edges_created,
        edges_deduplicated,
        evidence_aggregated,
    )

    # -------------------------------------------------------------------------
    # Step 6: Finalize metrics
    # -------------------------------------------------------------------------
    metrics.increment_items_processed(nodes_created + edges_created)
    metrics.complete()
    report.end_phase(phase_name)

    # Update report summary
    report.nodes_created = nodes_created
    report.edges_created = edges_created

    elapsed_time = time.time() - start_time
    logger.info(
        "Graph merge complete in %.2fs: %d nodes, %d edges",
        elapsed_time,
        nodes_created,
        edges_created,
    )

    return MergeResult(
        graph=graph,
        nodes_created=nodes_created,
        edges_created=edges_created,
        edges_deduplicated=edges_deduplicated,
        evidence_aggregated=evidence_aggregated,
        phase_metrics=metrics,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def get_merge_summary(result: MergeResult) -> dict[str, Any]:
    """
    Generate a summary of the merge operation.

    Args:
        result: MergeResult from merge_to_knowledge_graph.

    Returns:
        Dictionary with merge statistics and graph summary.
    """
    graph_summary = result.graph.to_summary()

    return {
        "merge_stats": {
            "nodes_created": result.nodes_created,
            "edges_created": result.edges_created,
            "edges_deduplicated": result.edges_deduplicated,
            "evidence_aggregated": result.evidence_aggregated,
        },
        "graph": graph_summary,
        "timing": {
            "duration_s": (
                result.phase_metrics.duration_s if result.phase_metrics else 0.0
            ),
        },
    }


def filter_high_confidence_edges(
    graph: KnowledgeGraph,
    min_confidence: float = 0.7,
) -> list[dict[str, Any]]:
    """
    Extract edges with confidence above a threshold.

    Args:
        graph: KnowledgeGraph to filter.
        min_confidence: Minimum confidence score (0-1).

    Returns:
        List of high-confidence edge dictionaries.
    """
    high_confidence = []

    for rel_key, rel_data in graph.relationships.items():
        ml_score = rel_data.get("ml_score", 0.0)
        if ml_score and ml_score >= min_confidence:
            high_confidence.append({
                "source": rel_key[0],
                "target": rel_key[1],
                "relation_type": rel_key[2],
                "confidence": ml_score,
                "evidence_count": len(rel_data.get("evidence", [])),
                "is_multi_source": rel_data.get("is_multi_source", False),
            })

    # Sort by confidence descending
    high_confidence.sort(key=lambda x: x["confidence"], reverse=True)

    return high_confidence
