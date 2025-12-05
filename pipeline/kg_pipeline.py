"""
Knowledge Graph Pipeline - Deterministic processing of fetched data.

This module provides a pure Python async pipeline that processes data
collected by the DataFetchAgent. It runs batched LLM mining and builds
the knowledge graph without any agent orchestration.

The separation between data fetching (LLM-driven) and data processing
(deterministic) allows for better efficiency and reliability.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from extraction import create_batched_miner, BatchedLiteLLMMiner
from models.knowledge_graph import (
    KnowledgeGraph,
    RelationshipType,
    EvidenceSource,
    EntityType,
)
from pipeline.models import PipelineInput, PipelineResult, GraphStatistics

logger = logging.getLogger(__name__)


def _normalize_for_lookup(text: str) -> str:
    """
    Normalize entity text for fuzzy type lookup.

    Removes case differences, hyphens, and spaces to enable matching
    between LLM-extracted entity names and PubTator annotations.

    Examples:
        "ACE-2" -> "ace2"
        "COVID 19" -> "covid19"
        "HCRTR-1" -> "hcrtr1"
    """
    return text.lower().strip().replace("-", "").replace(" ", "").replace("_", "")


# Entity synonym mapping for canonicalization
# Maps variant names to their canonical (STRING/official) form
# This merges entities that are the same but appear under different names
_ENTITY_SYNONYMS: dict[str, str] = {
    # Orexin receptors
    "ox1r": "HCRTR1",
    "ox-1r": "HCRTR1",
    "ox 1r": "HCRTR1",
    "orexin receptor 1": "HCRTR1",
    "orexin receptor type 1": "HCRTR1",
    "orexin-1 receptor": "HCRTR1",
    "hcrtr-1": "HCRTR1",
    "ox2r": "HCRTR2",
    "ox-2r": "HCRTR2",
    "ox 2r": "HCRTR2",
    "orexin receptor 2": "HCRTR2",
    "orexin receptor type 2": "HCRTR2",
    "orexin-2 receptor": "HCRTR2",
    "hcrtr-2": "HCRTR2",
    # Orexin/Hypocretin
    "orexin": "HCRT",
    "orexins": "HCRT",
    "orexin a": "HCRT",
    "orexin b": "HCRT",
    "orexin-a": "HCRT",
    "orexin-b": "HCRT",
    "hypocretin": "HCRT",
    "hypocretins": "HCRT",
    "hcrt/orexin": "HCRT",
    "orexin/hcrt": "HCRT",
    "prepro-orexin": "HCRT",
    "preproorexin": "HCRT",
    "hcrt": "HCRT",

    # Amyloid Precursor Protein / Amyloid Beta variants (NCBI: 351)
    "amyloid beta": "APP",
    "amyloidbeta": "APP",
    "amyloid-beta": "APP",
    "amyloid-β": "APP",
    "aβ": "APP",
    "abeta": "APP",
    "a-beta": "APP",
    "beta-amyloid": "APP",
    "β-amyloid": "APP",
    "beta amyloid": "APP",
    "amyloid beta peptide": "APP",
    "amyloid beta protein": "APP",
    "amyloid precursor protein": "APP",
    "amyloid-beta peptide": "APP",
    "aβ peptide": "APP",
    "aβ42": "APP",
    "aβ40": "APP",
    "abeta42": "APP",
    "abeta40": "APP",

    # Tau / MAPT variants (NCBI: 4137)
    "tau": "MAPT",
    "tau protein": "MAPT",
    "microtubule-associated protein tau": "MAPT",
    "microtubule associated protein tau": "MAPT",
    "p-tau": "MAPT",
    "ptau": "MAPT",
    "phospho-tau": "MAPT",
    "phosphorylated tau": "MAPT",
    "hyperphosphorylated tau": "MAPT",
    "tau tangles": "MAPT",
    "neurofibrillary tangles": "MAPT",

    # APOE variants (NCBI: 348)
    "apoe": "APOE",
    "apoe4": "APOE",
    "apoe ε4": "APOE",
    "apoe epsilon4": "APOE",
    "apoe epsilon 4": "APOE",
    "apoe-ε4": "APOE",
    "apoe-4": "APOE",
    "apoe3": "APOE",
    "apoe2": "APOE",
    "apolipoprotein e": "APOE",
    "apolipoprotein e4": "APOE",
    "apo e": "APOE",
    "apo-e": "APOE",

    # Common disease name variants
    "alzheimer's disease": "Alzheimer disease",
    "alzheimers disease": "Alzheimer disease",
    "alzheimer": "Alzheimer disease",
    "alzheimers": "Alzheimer disease",
    "ad": "Alzheimer disease",

    "parkinson's disease": "Parkinson disease",
    "parkinsons disease": "Parkinson disease",
    "parkinson": "Parkinson disease",
    "pd": "Parkinson disease",

    "huntington's disease": "Huntington disease",
    "huntingtons disease": "Huntington disease",
    "huntington": "Huntington disease",
    "hd": "Huntington disease",

    # Common chemical synonyms
    "vitamin e": "alpha-tocopherol",
    "vit e": "alpha-tocopherol",
    "tocopherol": "alpha-tocopherol",

    # BRCA genes
    "brca-1": "BRCA1",
    "brca 1": "BRCA1",
    "brca-2": "BRCA2",
    "brca 2": "BRCA2",

    # Common receptor variants
    "insulin receptor": "INSR",
    "ir": "INSR",
    "igf-1r": "IGF1R",
    "igf1 receptor": "IGF1R",
    "igf-1 receptor": "IGF1R",

    # TREM2 variants
    "trem-2": "TREM2",
    "triggering receptor expressed on myeloid cells 2": "TREM2",

    # GSK3B variants
    "gsk-3β": "GSK3B",
    "gsk-3b": "GSK3B",
    "gsk3β": "GSK3B",
    "gsk3 beta": "GSK3B",
    "glycogen synthase kinase 3 beta": "GSK3B",
}


def _canonicalize_entity(name: str) -> str:
    """
    Canonicalize entity name to merge synonyms.

    Maps variant names (OX1R, Orexin, etc.) to their canonical form
    (HCRTR1, HCRT) to ensure the same entity isn't duplicated in the graph.

    Args:
        name: Entity name from any source (STRING, PubTator, LLM extraction)

    Returns:
        Canonical entity name, or original name if no synonym mapping exists.
    """
    # Normalize for lookup
    normalized = name.lower().strip().replace("-", " ").replace("_", " ")
    normalized_no_space = normalized.replace(" ", "")

    # Check both forms against synonym map
    if normalized_no_space in _ENTITY_SYNONYMS:
        return _ENTITY_SYNONYMS[normalized_no_space]
    if normalized in _ENTITY_SYNONYMS:
        return _ENTITY_SYNONYMS[normalized]

    # If no synonym found, preserve original case
    return name


class KGPipeline:
    """
    Pure Python async pipeline for knowledge graph construction.

    Takes PipelineInput (articles, annotations, STRING interactions)
    and produces a KnowledgeGraph through deterministic processing.

    No LLM orchestration - uses BatchedLiteLLMMiner directly for extraction.

    Example:
        >>> pipeline = KGPipeline(extractor_name="cerebras")
        >>> result = await pipeline.run(pipeline_input)
        >>> graph = result.graph
        >>> print(f"Built graph with {graph.to_summary()['node_count']} nodes")
    """

    def __init__(
        self,
        extractor_name: str = "cerebras",
        evidence_threshold: float = 0.7,
    ):
        """
        Initialize the KG pipeline.

        Args:
            extractor_name: Name of the LLM extractor to use for relationship mining.
                Options: "cerebras" (fast), "gemini" (high quality)
            evidence_threshold: Minimum similarity for evidence validation (0.0-1.0).
        """
        self._extractor_name = extractor_name
        self._evidence_threshold = evidence_threshold
        self._miner: BatchedLiteLLMMiner | None = None

    def _get_miner(self) -> BatchedLiteLLMMiner:
        """Get or create the batched miner instance."""
        if self._miner is None:
            self._miner = create_batched_miner(
                extractor_name=self._extractor_name,
                evidence_threshold=self._evidence_threshold,
            )
        return self._miner

    async def run(self, input_data: PipelineInput) -> PipelineResult:
        """
        Run the full knowledge graph pipeline.

        Steps:
        1. Run batched LLM mining on articles to extract relationships
        2. Build knowledge graph from STRING + extracted relationships
        3. Return the graph with statistics

        Args:
            input_data: PipelineInput with articles, annotations, and STRING data.

        Returns:
            PipelineResult with the built KnowledgeGraph and statistics.
        """
        start_time = time.time()
        errors: list[str] = []

        if input_data.is_empty:
            logger.warning("Pipeline received empty input data")
            return PipelineResult(
                graph=KnowledgeGraph(),
                errors=["No input data provided"],
            )

        logger.info(
            f"KGPipeline starting: {input_data.article_count} articles, "
            f"{input_data.interaction_count} STRING interactions, "
            f"{input_data.annotation_count} annotations"
        )

        # Step 1: Run batched LLM mining for relationship extraction
        mining_result = await self._extract_relationships(input_data)
        relationships = mining_result.get("valid_relationships", [])
        mining_stats = mining_result.get("statistics", {})

        if mining_result.get("errors"):
            errors.extend(mining_result["errors"])

        logger.info(
            f"Extraction complete: {len(relationships)} valid relationships "
            f"from {mining_stats.get('total_relationships', 0)} total"
        )

        # Step 2: Build knowledge graph (full, with co-occurrence edges)
        full_graph = self._build_graph(
            string_interactions=input_data.string_interactions,
            literature_relationships=relationships,
            annotations_by_pmid=input_data.annotations,
        )

        # Step 3: Compute stats before pruning
        stats_before = self._compute_graph_stats(full_graph)
        logger.info(
            f"Graph before pruning: {stats_before.node_count} nodes, "
            f"{stats_before.edge_count} edges, {stats_before.component_count} components"
        )

        # Step 4: Prune co-occurrence-only nodes
        pruned_graph = self._prune_cooccurrence_only_nodes(full_graph)

        # Step 5: Entity deduplication (merge duplicates by various criteria)
        dedup_stats = self._deduplicate_entities(pruned_graph)
        if dedup_stats.get("total_merged", 0) > 0:
            logger.info(
                f"Entity deduplication: merged {dedup_stats['total_merged']} entities "
                f"({dedup_stats.get('ncbi_merged', 0)} by NCBI ID, "
                f"{dedup_stats.get('case_merged', 0)} by case)"
            )

        # Step 6: Compute stats after pruning and deduplication
        stats_after = self._compute_graph_stats(pruned_graph)

        nodes_pruned = stats_before.node_count - stats_after.node_count
        edges_pruned = stats_before.edge_count - stats_after.edge_count

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"KGPipeline complete: {stats_after.node_count} nodes, "
            f"{stats_after.edge_count} edges (pruned {nodes_pruned} nodes, "
            f"{edges_pruned} edges) in {processing_time_ms:.0f}ms"
        )

        return PipelineResult(
            graph=pruned_graph,
            relationships_extracted=mining_stats.get("total_relationships", 0),
            relationships_valid=len(relationships),
            entities_found=stats_after.node_count,
            processing_time_ms=processing_time_ms,
            mining_statistics=mining_stats,
            errors=errors,
            stats_before_pruning=stats_before,
            stats_after_pruning=stats_after,
            nodes_pruned=nodes_pruned,
            edges_pruned=edges_pruned,
        )

    async def _extract_relationships(
        self,
        input_data: PipelineInput,
    ) -> dict[str, Any]:
        """
        Extract relationships from articles using batched LLM mining.

        Args:
            input_data: PipelineInput with articles and annotations.

        Returns:
            Mining result dict with relationships and statistics.
        """
        if not input_data.articles:
            logger.info("No articles to extract relationships from")
            return {
                "relationships": [],
                "valid_relationships": [],
                "statistics": {"total_relationships": 0, "valid_relationships": 0},
                "errors": [],
            }

        miner = self._get_miner()

        try:
            result = await miner.run(
                articles=input_data.articles,
                annotations=input_data.annotations,
            )
            return result
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return {
                "relationships": [],
                "valid_relationships": [],
                "statistics": {},
                "errors": [str(e)],
            }

    def _build_graph(
        self,
        string_interactions: list[dict[str, Any]],
        literature_relationships: list[dict[str, Any]],
        annotations_by_pmid: dict[str, list[dict[str, Any]]],
    ) -> KnowledgeGraph:
        """
        Build knowledge graph from all data sources.

        This method is adapted from AgentTools.build_knowledge_graph
        but operates as a pure function without state dependency.

        Args:
            string_interactions: Protein interactions from STRING.
            literature_relationships: Extracted relationships from LLM mining.
            annotations_by_pmid: NER annotations keyed by PMID.

        Returns:
            Built KnowledgeGraph.
        """
        graph = KnowledgeGraph()

        # Track STRING entity IDs for post-processing (these are always proteins)
        string_entity_ids: set[str] = set()

        # Build entity type lookup from PubTator annotations for later use
        # Store multiple variants to handle LLM output variations:
        # - Original text (and its canonical form)
        # - Lowercase
        # - Uppercase
        # - Normalized (no hyphens, spaces, underscores)
        pubtator_types: dict[str, str] = {}
        for pmid, annotations in annotations_by_pmid.items():
            for ann in annotations:
                entity_text = ann.get("entity_text", "")
                entity_type = ann.get("entity_type", "unknown").lower()
                if entity_text and entity_type != "unknown":
                    # Store all case variants for robust lookup
                    pubtator_types[entity_text] = entity_type
                    pubtator_types[entity_text.lower()] = entity_type
                    pubtator_types[entity_text.upper()] = entity_type
                    # Also store normalized version for fuzzy matching
                    normalized = _normalize_for_lookup(entity_text)
                    if normalized:
                        pubtator_types[normalized] = entity_type
                    # Store canonical form for synonym lookup
                    canonical = _canonicalize_entity(entity_text)
                    if canonical != entity_text:
                        pubtator_types[canonical] = entity_type
                        pubtator_types[canonical.lower()] = entity_type

        # 1. Add ALL entities from PubTator NER annotations (with canonicalization)
        entity_to_pmids: dict[str, set[str]] = {}
        canonicalized_count = 0
        for pmid, annotations in annotations_by_pmid.items():
            for ann in annotations:
                entity_text = ann.get("entity_text", "")
                entity_type = ann.get("entity_type", "unknown").lower()
                entity_ncbi_id = ann.get("entity_id", "")

                if entity_text:
                    # Canonicalize entity name to merge synonyms
                    canonical_name = _canonicalize_entity(entity_text)
                    if canonical_name != entity_text:
                        canonicalized_count += 1

                    graph.add_entity(
                        entity_id=canonical_name,
                        entity_type=entity_type,
                        name=canonical_name,
                        entity_ncbi_id=entity_ncbi_id,
                    )
                    # Track PMIDs for co-occurrence (use canonical name)
                    if canonical_name not in entity_to_pmids:
                        entity_to_pmids[canonical_name] = set()
                    entity_to_pmids[canonical_name].add(pmid)

        # 2. Create co-occurrence edges (entities in same PMID) - use canonical names
        for pmid, annotations in annotations_by_pmid.items():
            # Canonicalize all entity names in this PMID
            entities_in_pmid = list(set(
                _canonicalize_entity(ann.get("entity_text", ""))
                for ann in annotations if ann.get("entity_text")
            ))
            # Create edges between all pairs of entities in same PMID
            for i, e1 in enumerate(entities_in_pmid):
                for e2 in entities_in_pmid[i + 1:]:
                    if e1 and e2 and e1 != e2:
                        graph.add_relationship(
                            source=e1,
                            target=e2,
                            rel_type=RelationshipType.COOCCURS_WITH,
                            evidence=[{
                                "source_type": EvidenceSource.LITERATURE.value,
                                "source_id": f"PMID:{pmid}",
                                "confidence": 0.3,  # Low confidence for co-occurrence
                            }],
                        )

        # 3. Add STRING interactions (high confidence)
        for interaction in string_interactions:
            protein_a = interaction.get("preferredName_A", "")
            protein_b = interaction.get("preferredName_B", "")
            score = interaction.get("score", 0.0)

            if protein_a and protein_b:
                graph.add_entity(protein_a, EntityType.PROTEIN.value, protein_a)
                graph.add_entity(protein_b, EntityType.PROTEIN.value, protein_b)
                # Track STRING entities for post-processing type enrichment
                string_entity_ids.add(protein_a)
                string_entity_ids.add(protein_b)

                graph.add_relationship(
                    source=protein_a,
                    target=protein_b,
                    rel_type=RelationshipType.INTERACTS_WITH,
                    evidence=[{
                        "source_type": EvidenceSource.STRING.value,
                        "source_id": f"STRING:{protein_a}-{protein_b}",
                        "confidence": float(score),
                    }],
                )

        # 4. Add typed literature relationships (enrich/override co-occurrence) - canonicalize names
        for rel in literature_relationships:
            raw_entity1 = rel.get("entity1", "")
            raw_entity2 = rel.get("entity2", "")
            rel_type_str = rel.get("relationship", "associated_with")
            confidence = rel.get("confidence", 0.5)
            pmid = rel.get("pmid", "")
            evidence_text = rel.get("evidence_text", "")

            if raw_entity1 and raw_entity2:
                # Canonicalize entity names to merge synonyms
                entity1 = _canonicalize_entity(raw_entity1)
                entity2 = _canonicalize_entity(raw_entity2)

                # Look up entity types from PubTator with fallback chain:
                # 1. Canonical form
                # 2. Exact match (original)
                # 3. Lowercase match
                # 4. Normalized match (removes hyphens, spaces, underscores)
                e1_type = (
                    pubtator_types.get(entity1)
                    or pubtator_types.get(raw_entity1)
                    or pubtator_types.get(raw_entity1.lower())
                    or pubtator_types.get(_normalize_for_lookup(raw_entity1))
                    or "unknown"
                )
                e2_type = (
                    pubtator_types.get(entity2)
                    or pubtator_types.get(raw_entity2)
                    or pubtator_types.get(raw_entity2.lower())
                    or pubtator_types.get(_normalize_for_lookup(raw_entity2))
                    or "unknown"
                )

                # Pre-create entities with correct types BEFORE adding relationship
                # This prevents add_relationship from creating them as "unknown"
                if entity1 not in graph.entities:
                    graph.add_entity(entity1, e1_type, entity1)
                if entity2 not in graph.entities:
                    graph.add_entity(entity2, e2_type, entity2)

                rel_type_map = {
                    "activates": RelationshipType.ACTIVATES,
                    "inhibits": RelationshipType.INHIBITS,
                    "associated_with": RelationshipType.ASSOCIATED_WITH,
                    "regulates": RelationshipType.REGULATES,
                    "binds_to": RelationshipType.BINDS_TO,
                    "cooccurs_with": RelationshipType.COOCCURS_WITH,
                }
                rel_type = rel_type_map.get(
                    rel_type_str.lower(),
                    RelationshipType.ASSOCIATED_WITH
                )

                graph.add_relationship(
                    source=entity1,
                    target=entity2,
                    rel_type=rel_type,
                    evidence=[{
                        "source_type": EvidenceSource.LITERATURE.value,
                        "source_id": f"PMID:{pmid}",
                        "confidence": float(confidence),
                        "text_snippet": evidence_text,
                    }],
                )

        # 5. POST-PROCESSING: Enrich "unknown" entity types
        # This catches entities that were added by LLM extraction with names
        # that didn't match PubTator annotations exactly
        unknown_count_before = sum(
            1 for e in graph.entities.values() if e.get("type") == "unknown"
        )

        for entity_id, entity_data in graph.entities.items():
            current_type = entity_data.get("type", "unknown")

            # Skip if already typed
            if current_type != "unknown":
                continue

            # Priority 1: If entity was source/target of STRING interaction -> protein
            if entity_id in string_entity_ids:
                graph.entities[entity_id]["type"] = EntityType.PROTEIN.value
                graph.graph.nodes[entity_id]["type"] = EntityType.PROTEIN.value
                graph.graph.nodes[entity_id]["entity_type"] = EntityType.PROTEIN.value
                continue

            # Priority 2: Try fuzzy matching against PubTator types
            # Check normalized form
            normalized_id = _normalize_for_lookup(entity_id)
            found_type = pubtator_types.get(normalized_id)

            if not found_type:
                # Try exact matches with case variants
                found_type = (
                    pubtator_types.get(entity_id)
                    or pubtator_types.get(entity_id.lower())
                    or pubtator_types.get(entity_id.upper())
                )

            if found_type and found_type != "unknown":
                graph.entities[entity_id]["type"] = found_type
                graph.graph.nodes[entity_id]["type"] = found_type
                graph.graph.nodes[entity_id]["entity_type"] = found_type

        unknown_count_after = sum(
            1 for e in graph.entities.values() if e.get("type") == "unknown"
        )

        if unknown_count_before > unknown_count_after:
            logger.info(
                f"Entity type enrichment: resolved {unknown_count_before - unknown_count_after} "
                f"unknown types ({unknown_count_before} -> {unknown_count_after})"
            )

        # Log canonicalization stats
        if canonicalized_count > 0:
            logger.info(
                f"Entity canonicalization: merged {canonicalized_count} synonym occurrences "
                f"into canonical forms (e.g., OX1R->HCRTR1, Orexin->HCRT)"
            )

        return graph

    def _deduplicate_entities(self, graph: KnowledgeGraph) -> dict[str, Any]:
        """
        Deduplicate entities in the graph by merging duplicates.

        Applies multiple deduplication strategies in order:
        1. By NCBI ID - merge entities with same NCBI gene/protein ID
        2. By case-insensitive matching - merge "Tau" with "tau", etc.

        Args:
            graph: KnowledgeGraph to deduplicate (modified in place).

        Returns:
            Statistics about the deduplication:
                - total_merged: Total entities merged
                - ncbi_merged: Entities merged by NCBI ID
                - case_merged: Entities merged by case-insensitive match
        """
        stats = {
            "total_merged": 0,
            "ncbi_merged": 0,
            "case_merged": 0,
        }

        # Step 1: Deduplicate by NCBI ID (most reliable)
        try:
            ncbi_result = graph.deduplicate_by_ncbi_id()
            ncbi_merged = ncbi_result.get("original_count", 0) - ncbi_result.get("final_count", 0)
            stats["ncbi_merged"] = ncbi_merged
            stats["total_merged"] += ncbi_merged

            if ncbi_merged > 0:
                logger.debug(
                    f"NCBI deduplication: merged {ncbi_merged} entities into "
                    f"{len(ncbi_result.get('merged_groups', []))} groups"
                )
                for group in ncbi_result.get("merged_groups", [])[:5]:
                    logger.debug(
                        f"  NCBI {group['ncbi_id']}: {group['canonical']} <- {group['merged']}"
                    )
        except Exception as e:
            logger.warning(f"NCBI deduplication failed: {e}")

        # Step 2: Deduplicate by case-insensitive matching
        try:
            case_result = graph.deduplicate_by_case_insensitive()
            case_merged = case_result.get("original_count", 0) - case_result.get("final_count", 0)
            stats["case_merged"] = case_merged
            stats["total_merged"] += case_merged

            if case_merged > 0:
                logger.debug(
                    f"Case-insensitive deduplication: merged {case_merged} entities"
                )
                for group in case_result.get("merged_groups", [])[:5]:
                    logger.debug(
                        f"  '{group['normalized']}': {group['canonical']} <- {group['merged']}"
                    )
        except Exception as e:
            logger.warning(f"Case-insensitive deduplication failed: {e}")

        return stats

    def _compute_graph_stats(self, graph: KnowledgeGraph) -> GraphStatistics:
        """Compute statistics for a graph."""
        import networkx as nx

        summary = graph.to_summary()
        rel_types = summary.get("relationship_types", {})

        # Count connected components
        try:
            component_count = nx.number_connected_components(graph.graph.to_undirected())
        except Exception:
            component_count = 0

        return GraphStatistics(
            node_count=summary.get("node_count", 0),
            edge_count=summary.get("edge_count", 0),
            component_count=component_count,
            relationship_types=rel_types,
        )

    def _prune_cooccurrence_only_nodes(
        self,
        graph: KnowledgeGraph,
    ) -> KnowledgeGraph:
        """
        Prune nodes that are only connected via co-occurrence edges.

        Keeps nodes that have at least one typed relationship (activates,
        inhibits, associated_with, regulates, binds_to, interacts_with).

        Args:
            graph: Original KnowledgeGraph.

        Returns:
            Pruned KnowledgeGraph with only nodes having typed relationships.
        """
        import networkx as nx

        # Typed relationship types (not co-occurrence)
        typed_rel_types = {
            RelationshipType.ACTIVATES.value,
            RelationshipType.INHIBITS.value,
            RelationshipType.ASSOCIATED_WITH.value,
            RelationshipType.REGULATES.value,
            RelationshipType.BINDS_TO.value,
            RelationshipType.INTERACTS_WITH.value,
        }

        # Find nodes with at least one typed edge
        nodes_with_typed_edges = set()
        for u, v, data in graph.graph.edges(data=True):
            # Check both "relation_type" (how it's stored) and "rel_type" (fallback)
            rel_type = data.get("relation_type", data.get("rel_type", ""))
            if rel_type in typed_rel_types:
                nodes_with_typed_edges.add(u)
                nodes_with_typed_edges.add(v)

        # Create pruned graph
        pruned = KnowledgeGraph()

        # Add only nodes with typed edges
        for node_id in nodes_with_typed_edges:
            if node_id in graph.graph.nodes:
                node_data = graph.graph.nodes[node_id]
                # Entity type is stored as "type" in the graph (not "entity_type")
                pruned.add_entity(
                    entity_id=node_id,
                    entity_type=node_data.get("type", "unknown"),
                    name=node_data.get("name", node_id),
                    entity_ncbi_id=node_data.get("entity_ncbi_id", ""),
                )

        # Add only typed edges between kept nodes
        for u, v, data in graph.graph.edges(data=True):
            rel_type = data.get("relation_type", data.get("rel_type", ""))
            if (
                u in nodes_with_typed_edges
                and v in nodes_with_typed_edges
                and rel_type in typed_rel_types
            ):
                # Convert rel_type string back to enum
                try:
                    rel_enum = RelationshipType(rel_type)
                except ValueError:
                    rel_enum = RelationshipType.ASSOCIATED_WITH

                pruned.add_relationship(
                    source=u,
                    target=v,
                    rel_type=rel_enum,
                    evidence=data.get("evidence", []),
                )

        logger.info(
            f"Pruning: {graph.to_summary()['node_count']} → {pruned.to_summary()['node_count']} nodes, "
            f"{graph.to_summary()['edge_count']} → {pruned.to_summary()['edge_count']} edges"
        )

        return pruned


async def run_kg_pipeline(
    input_data: PipelineInput,
    extractor_name: str = "cerebras",
    evidence_threshold: float = 0.7,
) -> PipelineResult:
    """
    Convenience function to run the KG pipeline.

    Args:
        input_data: PipelineInput with articles, annotations, and STRING data.
        extractor_name: Name of the LLM extractor to use.
        evidence_threshold: Minimum similarity for evidence validation.

    Returns:
        PipelineResult with the built KnowledgeGraph.
    """
    pipeline = KGPipeline(
        extractor_name=extractor_name,
        evidence_threshold=evidence_threshold,
    )
    return await pipeline.run(input_data)
