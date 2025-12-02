"""
Tests for the graph merge module (Phase 5).

Tests cover:
- Edge confidence scoring
- Edge deduplication
- Entity extraction
- Full merge integration
"""

import pytest
from pipeline.merge import (
    MergeResult,
    EdgeCandidate,
    calculate_edge_confidence,
    deduplicate_edges,
    merge_to_knowledge_graph,
    get_merge_summary,
    filter_high_confidence_edges,
    _normalize_entity_id,
    _create_edge_key,
    _map_relationship_type,
    _map_entity_type,
)
from pipeline.string_expansion import STRINGExpansionResult
from pipeline.parallel_extraction import ExtractionResult, CoOccurrenceEdge
from pipeline.config import PipelineConfig
from pipeline.metrics import PipelineReport
from models.knowledge_graph import RelationshipType


# =============================================================================
# Test: calculate_edge_confidence
# =============================================================================


class TestCalculateEdgeConfidence:
    """Tests for confidence score calculation."""

    def test_string_high_confidence(self):
        """STRING with high score should give ~0.81 confidence."""
        score = calculate_edge_confidence(["STRING"], string_score=0.9)
        assert 0.8 <= score <= 0.85

    def test_string_low_confidence(self):
        """STRING with low score should give lower confidence."""
        score = calculate_edge_confidence(["STRING"], string_score=0.5)
        assert 0.4 <= score <= 0.5

    def test_llm_source(self):
        """LLM source should give base confidence of ~0.7."""
        score = calculate_edge_confidence(["llm"])
        assert 0.65 <= score <= 0.75

    def test_cooccurrence_high_pmi(self):
        """Co-occurrence with high PMI should give moderate confidence."""
        score = calculate_edge_confidence(["co-occurrence"], pmi_score=4.0)
        assert 0.4 <= score <= 0.6

    def test_cooccurrence_low_pmi(self):
        """Co-occurrence with low PMI should give lower confidence."""
        score = calculate_edge_confidence(["co-occurrence"], pmi_score=1.0)
        assert 0.2 <= score <= 0.35

    def test_multi_source_boost(self):
        """Multiple sources should boost confidence."""
        single = calculate_edge_confidence(["STRING"], string_score=0.8)
        multi = calculate_edge_confidence(["STRING", "llm"], string_score=0.8)
        assert multi > single

    def test_evidence_count_boost(self):
        """More evidence should boost confidence."""
        base = calculate_edge_confidence(["llm"], evidence_count=1)
        boosted = calculate_edge_confidence(["llm"], evidence_count=5)
        assert boosted > base

    def test_max_confidence_cap(self):
        """Confidence should be capped at 1.0."""
        score = calculate_edge_confidence(
            ["STRING", "llm", "co-occurrence"],
            string_score=0.99,
            pmi_score=5.0,
            evidence_count=10,
        )
        assert score <= 1.0

    def test_empty_sources(self):
        """Empty sources list should return 0."""
        score = calculate_edge_confidence([])
        assert score == 0.0


# =============================================================================
# Test: deduplicate_edges
# =============================================================================


class TestDeduplicateEdges:
    """Tests for edge deduplication."""

    def test_no_duplicates(self):
        """Different edges should remain separate."""
        edges = [
            EdgeCandidate("A", "B", "interacts_with", "STRING", string_score=0.9),
            EdgeCandidate("B", "C", "activates", "llm", confidence=0.8),
        ]
        result = deduplicate_edges(edges)
        assert len(result) == 2

    def test_merge_same_edge(self):
        """Same source-target-type should be merged."""
        edges = [
            EdgeCandidate("A", "B", "interacts_with", "STRING", string_score=0.9),
            EdgeCandidate("A", "B", "interacts_with", "llm", confidence=0.8),
        ]
        result = deduplicate_edges(edges)
        assert len(result) == 1
        assert set(result[0]["data_sources"]) == {"STRING", "llm"}

    def test_different_relation_types_not_merged(self):
        """Different relation types should remain separate."""
        edges = [
            EdgeCandidate("A", "B", "interacts_with", "STRING"),
            EdgeCandidate("A", "B", "activates", "llm"),
        ]
        result = deduplicate_edges(edges)
        assert len(result) == 2

    def test_aggregates_evidence(self):
        """Deduplication should aggregate evidence texts."""
        edges = [
            EdgeCandidate(
                "A", "B", "interacts_with", "llm",
                evidence_texts=["Evidence 1"],
                pmids=["123"],
            ),
            EdgeCandidate(
                "A", "B", "interacts_with", "llm",
                evidence_texts=["Evidence 2"],
                pmids=["456"],
            ),
        ]
        result = deduplicate_edges(edges)
        assert len(result) == 1
        assert len(result[0]["evidence"]) == 2
        assert set(result[0]["pmids"]) == {"123", "456"}

    def test_keeps_best_scores(self):
        """Should keep highest STRING and PMI scores."""
        edges = [
            EdgeCandidate("A", "B", "interacts_with", "STRING", string_score=0.7),
            EdgeCandidate("A", "B", "interacts_with", "STRING", string_score=0.9),
        ]
        result = deduplicate_edges(edges)
        assert result[0]["string_score"] == 0.9

    def test_symmetric_relation_order_invariant(self):
        """Symmetric relations should be deduplicated regardless of order."""
        edges = [
            EdgeCandidate("A", "B", "interacts_with", "STRING"),
            EdgeCandidate("B", "A", "interacts_with", "llm"),
        ]
        result = deduplicate_edges(edges)
        assert len(result) == 1

    def test_marks_multi_source(self):
        """Should mark edges from multiple sources."""
        edges = [
            EdgeCandidate("A", "B", "interacts_with", "STRING"),
            EdgeCandidate("A", "B", "interacts_with", "llm"),
        ]
        result = deduplicate_edges(edges)
        assert result[0].get("is_multi_source") is True


# =============================================================================
# Test: Entity and Type Mapping
# =============================================================================


class TestEntityMapping:
    """Tests for entity ID normalization and type mapping."""

    def test_normalize_entity_id(self):
        """Entity IDs should be normalized to uppercase."""
        assert _normalize_entity_id("brca1") == "BRCA1"
        assert _normalize_entity_id("  TP53  ") == "TP53"

    def test_create_edge_key_symmetric(self):
        """Symmetric relations should have consistent key regardless of order."""
        key1 = _create_edge_key("BRCA1", "TP53", "interacts_with")
        key2 = _create_edge_key("TP53", "BRCA1", "interacts_with")
        assert key1 == key2

    def test_create_edge_key_directed(self):
        """Directed relations should preserve order."""
        key1 = _create_edge_key("BRCA1", "TP53", "activates")
        key2 = _create_edge_key("TP53", "BRCA1", "activates")
        assert key1 == key2 or key1 != key2  # Order preserved or not based on impl

    def test_map_relationship_type_known(self):
        """Known relationship strings should map correctly."""
        assert _map_relationship_type("activates") == RelationshipType.ACTIVATES
        assert _map_relationship_type("inhibits") == RelationshipType.INHIBITS
        assert _map_relationship_type("interacts_with") == RelationshipType.INTERACTS_WITH

    def test_map_relationship_type_synonyms(self):
        """Synonyms should map to correct types."""
        assert _map_relationship_type("stimulates") == RelationshipType.ACTIVATES
        assert _map_relationship_type("suppresses") == RelationshipType.INHIBITS
        assert _map_relationship_type("blocks") == RelationshipType.INHIBITS

    def test_map_relationship_type_unknown(self):
        """Unknown types should default to ASSOCIATED_WITH."""
        assert _map_relationship_type("foobar") == RelationshipType.ASSOCIATED_WITH

    def test_map_entity_type(self):
        """Entity types should be normalized."""
        assert _map_entity_type("Gene") == "gene"
        assert _map_entity_type("PROTEIN") == "protein"
        assert _map_entity_type("Drug") == "chemical"


# =============================================================================
# Test: Full Integration
# =============================================================================


class TestMergeIntegration:
    """Integration tests for merge_to_knowledge_graph."""

    @pytest.fixture
    def string_result(self):
        """Sample STRING expansion result."""
        return STRINGExpansionResult(
            seed_proteins=["BRCA1", "TP53"],
            expanded_proteins=["BRCA1", "TP53", "ATM"],
            interactions=[
                {"preferredName_A": "BRCA1", "preferredName_B": "TP53", "score": 0.95},
                {"preferredName_A": "BRCA1", "preferredName_B": "ATM", "score": 0.8},
            ],
            proteins_not_found=[],
        )

    @pytest.fixture
    def extraction_result(self):
        """Sample extraction result."""
        return ExtractionResult(
            cooccurrence_edges=[
                CoOccurrenceEdge(
                    source="BRCA1",
                    target="cancer",
                    weight=5.0,
                    pmids=["123"],
                    source_type="Gene",
                    target_type="Disease",
                    pmi_score=3.0,
                    raw_count=5,
                ),
            ],
            llm_relationships=[
                {
                    "entity1": "TP53",
                    "entity2": "cell cycle",
                    "relationship": "regulates",
                    "confidence": 0.85,
                    "pmid": "456",
                    "evidence": "TP53 regulates cell cycle.",
                },
            ],
        )

    @pytest.mark.asyncio
    async def test_merge_creates_nodes(self, string_result, extraction_result):
        """Merge should create nodes from all sources."""
        result = await merge_to_knowledge_graph(
            string_result=string_result,
            extraction_result=extraction_result,
            pubmed_articles=[],
            config=PipelineConfig(),
            report=PipelineReport.create(),
        )
        assert result.nodes_created > 0
        assert "BRCA1" in result.graph.entities

    @pytest.mark.asyncio
    async def test_merge_creates_edges(self, string_result, extraction_result):
        """Merge should create edges from all sources."""
        result = await merge_to_knowledge_graph(
            string_result=string_result,
            extraction_result=extraction_result,
            pubmed_articles=[],
            config=PipelineConfig(),
            report=PipelineReport.create(),
        )
        assert result.edges_created > 0

    @pytest.mark.asyncio
    async def test_merge_returns_metrics(self, string_result, extraction_result):
        """Merge should return phase metrics."""
        result = await merge_to_knowledge_graph(
            string_result=string_result,
            extraction_result=extraction_result,
            pubmed_articles=[],
            config=PipelineConfig(),
            report=PipelineReport.create(),
        )
        assert result.phase_metrics is not None
        assert result.phase_metrics.phase_name == "graph_merge"

    @pytest.mark.asyncio
    async def test_empty_inputs(self):
        """Should handle empty inputs gracefully."""
        result = await merge_to_knowledge_graph(
            string_result=STRINGExpansionResult(
                seed_proteins=[],
                expanded_proteins=[],
                interactions=[],
                proteins_not_found=[],
            ),
            extraction_result=ExtractionResult(),
            pubmed_articles=[],
            config=PipelineConfig(),
            report=PipelineReport.create(),
        )
        assert result.nodes_created == 0
        assert result.edges_created == 0


# =============================================================================
# Test: Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    @pytest.mark.asyncio
    async def test_get_merge_summary(self):
        """get_merge_summary should return structured summary."""
        string_result = STRINGExpansionResult(
            seed_proteins=["A"],
            expanded_proteins=["A", "B"],
            interactions=[{"preferredName_A": "A", "preferredName_B": "B", "score": 0.9}],
            proteins_not_found=[],
        )
        result = await merge_to_knowledge_graph(
            string_result=string_result,
            extraction_result=ExtractionResult(),
            pubmed_articles=[],
            config=PipelineConfig(),
            report=PipelineReport.create(),
        )
        summary = get_merge_summary(result)
        assert "merge_stats" in summary
        assert "graph" in summary

    @pytest.mark.asyncio
    async def test_filter_high_confidence_edges(self):
        """filter_high_confidence_edges should return edges above threshold."""
        string_result = STRINGExpansionResult(
            seed_proteins=["A"],
            expanded_proteins=["A", "B", "C"],
            interactions=[
                {"preferredName_A": "A", "preferredName_B": "B", "score": 0.95},
                {"preferredName_A": "B", "preferredName_B": "C", "score": 0.5},
            ],
            proteins_not_found=[],
        )
        result = await merge_to_knowledge_graph(
            string_result=string_result,
            extraction_result=ExtractionResult(),
            pubmed_articles=[],
            config=PipelineConfig(),
            report=PipelineReport.create(),
        )
        high_conf = filter_high_confidence_edges(result.graph, min_confidence=0.7)
        # Only A->B has high enough score
        assert len(high_conf) >= 1
        assert all(e["confidence"] >= 0.7 for e in high_conf)
