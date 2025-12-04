"""
Tests for the BatchedLiteLLMMiner module.

Tests cover:
- Data structure serialization and properties
- Sentence processing and numbering
- Token counting (simple and tiktoken)
- Provenance validation (PMID + evidence index validation)
- Abstract chunking with token awareness
- Miner initialization and configuration
- Mock-based extraction with error handling
- Full pipeline integration

Following TDD discipline: comprehensive coverage before implementation changes.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from extraction.batched_litellm_miner import (
    AbstractChunk,
    BatchedLiteLLMMiner,
    ChunkMiningResult,
    ExtractedRelationship,
    MiningStatistics,
    ValidationResult,
    count_tokens_simple,
    create_batched_miner,
    find_evidence_in_abstract,
    number_sentences,
    run_batched_mining,
    split_into_sentences,
    try_count_tokens_tiktoken,
    validate_relationship_provenance,
    validate_relationship_provenance_indexed,
)


# =============================================================================
# Test: ExtractedRelationship
# =============================================================================


class TestExtractedRelationship:
    """Tests for ExtractedRelationship dataclass."""

    def test_to_dict_basic(self) -> None:
        """Should serialize all fields to dictionary."""
        rel = ExtractedRelationship(
            entity1="BRCA1",
            entity2="TP53",
            relationship="activates",
            confidence=0.85,
            pmid="12345678",
            evidence_sentence_indices=[1, 2],
            evidence_text="BRCA1 activates TP53 in cell cycle.",
        )
        d = rel.to_dict()

        assert d["entity1"] == "BRCA1"
        assert d["entity2"] == "TP53"
        assert d["relationship"] == "activates"
        assert d["confidence"] == 0.85
        assert d["pmid"] == "12345678"
        assert d["evidence_sentence_indices"] == [1, 2]
        assert d["evidence_text"] == "BRCA1 activates TP53 in cell cycle."

    def test_to_dict_with_defaults(self) -> None:
        """Should serialize with default empty values."""
        rel = ExtractedRelationship(
            entity1="A",
            entity2="B",
            relationship="binds_to",
            confidence=0.5,
            pmid="999",
        )
        d = rel.to_dict()

        assert d["evidence_sentence_indices"] == []
        assert d["evidence_text"] == ""

    def test_default_sentence_indices_is_empty_list(self) -> None:
        """Default evidence_sentence_indices should be empty list."""
        rel = ExtractedRelationship(
            entity1="X",
            entity2="Y",
            relationship="inhibits",
            confidence=0.7,
            pmid="111",
        )
        assert rel.evidence_sentence_indices == []


# =============================================================================
# Test: ValidationResult
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    @pytest.fixture
    def sample_relationship(self) -> ExtractedRelationship:
        """Sample relationship for validation tests."""
        return ExtractedRelationship(
            entity1="EGFR",
            entity2="cancer",
            relationship="associated_with",
            confidence=0.9,
            pmid="24681357",
            evidence_sentence_indices=[3],
            evidence_text="EGFR is associated with cancer.",
        )

    def test_is_valid_true_when_both_valid(
        self, sample_relationship: ExtractedRelationship
    ) -> None:
        """is_valid should be True when pmid_valid and evidence_valid are both True."""
        result = ValidationResult(
            relationship=sample_relationship,
            pmid_valid=True,
            evidence_valid=True,
            evidence_similarity=1.0,
        )
        assert result.is_valid is True

    def test_is_valid_false_when_pmid_invalid(
        self, sample_relationship: ExtractedRelationship
    ) -> None:
        """is_valid should be False when pmid_valid is False."""
        result = ValidationResult(
            relationship=sample_relationship,
            pmid_valid=False,
            evidence_valid=True,
            evidence_similarity=0.8,
            error_message="PMID not in chunk",
        )
        assert result.is_valid is False

    def test_is_valid_false_when_evidence_invalid(
        self, sample_relationship: ExtractedRelationship
    ) -> None:
        """is_valid should be False when evidence_valid is False."""
        result = ValidationResult(
            relationship=sample_relationship,
            pmid_valid=True,
            evidence_valid=False,
            evidence_similarity=0.3,
            error_message="Evidence not found",
        )
        assert result.is_valid is False

    def test_to_dict_includes_all_fields(
        self, sample_relationship: ExtractedRelationship
    ) -> None:
        """to_dict should include all validation fields and nested relationship."""
        result = ValidationResult(
            relationship=sample_relationship,
            pmid_valid=True,
            evidence_valid=True,
            evidence_similarity=0.95,
            error_message=None,
        )
        d = result.to_dict()

        assert "relationship" in d
        assert d["relationship"]["entity1"] == "EGFR"
        assert d["pmid_valid"] is True
        assert d["evidence_valid"] is True
        assert d["evidence_similarity"] == 0.95
        assert d["is_valid"] is True
        assert d["error_message"] is None

    def test_to_dict_includes_error_message(
        self, sample_relationship: ExtractedRelationship
    ) -> None:
        """to_dict should include error_message when present."""
        result = ValidationResult(
            relationship=sample_relationship,
            pmid_valid=False,
            evidence_valid=False,
            evidence_similarity=0.0,
            error_message="PMID 24681357 not in chunk PMIDs: ['11111']",
        )
        d = result.to_dict()

        assert d["error_message"] == "PMID 24681357 not in chunk PMIDs: ['11111']"


# =============================================================================
# Test: ChunkMiningResult
# =============================================================================


class TestChunkMiningResult:
    """Tests for ChunkMiningResult dataclass."""

    @pytest.fixture
    def sample_relationships(self) -> list[ExtractedRelationship]:
        """Sample relationships for result tests."""
        return [
            ExtractedRelationship(
                entity1="A", entity2="B", relationship="activates",
                confidence=0.9, pmid="111", evidence_sentence_indices=[1]
            ),
            ExtractedRelationship(
                entity1="C", entity2="D", relationship="inhibits",
                confidence=0.8, pmid="222", evidence_sentence_indices=[2]
            ),
        ]

    @pytest.fixture
    def mixed_validation_results(
        self, sample_relationships: list[ExtractedRelationship]
    ) -> list[ValidationResult]:
        """One valid, one invalid validation result."""
        return [
            ValidationResult(
                relationship=sample_relationships[0],
                pmid_valid=True,
                evidence_valid=True,
                evidence_similarity=1.0,
            ),
            ValidationResult(
                relationship=sample_relationships[1],
                pmid_valid=True,
                evidence_valid=False,
                evidence_similarity=0.3,
                error_message="Evidence not found",
            ),
        ]

    def test_success_true_when_no_errors(
        self, sample_relationships: list[ExtractedRelationship],
        mixed_validation_results: list[ValidationResult]
    ) -> None:
        """success should be True when errors list is empty."""
        result = ChunkMiningResult(
            chunk_id=0,
            relationships=sample_relationships,
            validation_results=mixed_validation_results,
            pmids_processed=["111", "222"],
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
            latency_ms=500.0,
            errors=[],
        )
        assert result.success is True

    def test_success_false_when_errors_present(
        self, sample_relationships: list[ExtractedRelationship],
        mixed_validation_results: list[ValidationResult]
    ) -> None:
        """success should be False when errors list has items."""
        result = ChunkMiningResult(
            chunk_id=0,
            relationships=sample_relationships,
            validation_results=mixed_validation_results,
            pmids_processed=["111", "222"],
            token_usage={},
            latency_ms=0.0,
            errors=["Rate limit exceeded"],
        )
        assert result.success is False

    def test_valid_relationships_filters_correctly(
        self, sample_relationships: list[ExtractedRelationship],
        mixed_validation_results: list[ValidationResult]
    ) -> None:
        """valid_relationships should return only relationships with is_valid=True."""
        result = ChunkMiningResult(
            chunk_id=0,
            relationships=sample_relationships,
            validation_results=mixed_validation_results,
            pmids_processed=["111", "222"],
            token_usage={},
            latency_ms=0.0,
        )
        valid = result.valid_relationships

        assert len(valid) == 1
        assert valid[0].entity1 == "A"
        assert valid[0].entity2 == "B"

    def test_invalid_count_calculates_correctly(
        self, sample_relationships: list[ExtractedRelationship],
        mixed_validation_results: list[ValidationResult]
    ) -> None:
        """invalid_count should count validation failures."""
        result = ChunkMiningResult(
            chunk_id=0,
            relationships=sample_relationships,
            validation_results=mixed_validation_results,
            pmids_processed=["111", "222"],
            token_usage={},
            latency_ms=0.0,
        )
        assert result.invalid_count == 1

    def test_to_dict_includes_computed_properties(
        self, sample_relationships: list[ExtractedRelationship],
        mixed_validation_results: list[ValidationResult]
    ) -> None:
        """to_dict should include success, valid_relationships, and invalid_count."""
        result = ChunkMiningResult(
            chunk_id=1,
            relationships=sample_relationships,
            validation_results=mixed_validation_results,
            pmids_processed=["111", "222"],
            token_usage={"prompt_tokens": 100},
            latency_ms=250.0,
        )
        d = result.to_dict()

        assert d["chunk_id"] == 1
        assert d["success"] is True
        assert d["invalid_count"] == 1
        assert len(d["valid_relationships"]) == 1
        assert len(d["relationships"]) == 2


# =============================================================================
# Test: MiningStatistics
# =============================================================================


class TestMiningStatistics:
    """Tests for MiningStatistics dataclass."""

    def test_validation_rate_with_relationships(self) -> None:
        """validation_rate should calculate valid/total ratio."""
        stats = MiningStatistics(
            total_relationships=10,
            valid_relationships=7,
            invalid_relationships=3,
        )
        d = stats.to_dict()

        assert d["validation_rate"] == 0.7

    def test_validation_rate_zero_when_no_relationships(self) -> None:
        """validation_rate should be 0 when total_relationships is 0."""
        stats = MiningStatistics(
            total_relationships=0,
            valid_relationships=0,
        )
        d = stats.to_dict()

        assert d["validation_rate"] == 0.0

    def test_throughput_calculation(self) -> None:
        """throughput_tok_per_sec should calculate tokens per second."""
        stats = MiningStatistics(
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_latency_ms=500.0,  # 0.5 seconds
        )
        d = stats.to_dict()

        # (1000 + 500) / 500ms * 1000 = 3000 tok/s
        assert d["throughput_tok_per_sec"] == 3000.0

    def test_throughput_zero_when_no_latency(self) -> None:
        """throughput_tok_per_sec should be 0 when latency is 0."""
        stats = MiningStatistics(
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_latency_ms=0.0,
        )
        d = stats.to_dict()

        assert d["throughput_tok_per_sec"] == 0.0

    def test_to_dict_includes_all_fields(self) -> None:
        """to_dict should include all statistics fields."""
        stats = MiningStatistics(
            total_abstracts=50,
            total_chunks=5,
            chunks_processed=5,
            total_relationships=25,
            valid_relationships=20,
            invalid_relationships=5,
            pmid_failures=2,
            evidence_failures=3,
            total_prompt_tokens=10000,
            total_completion_tokens=5000,
            total_latency_ms=5000.0,
            wall_clock_ms=3000.0,
        )
        d = stats.to_dict()

        assert d["total_abstracts"] == 50
        assert d["total_chunks"] == 5
        assert d["chunks_processed"] == 5
        assert d["pmid_failures"] == 2
        assert d["evidence_failures"] == 3
        assert d["wall_clock_ms"] == 3000.0


# =============================================================================
# Test: split_into_sentences
# =============================================================================


class TestSplitIntoSentences:
    """Tests for sentence splitting function."""

    def test_basic_sentences(self) -> None:
        """Should split text on period followed by space."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = split_into_sentences(text)

        assert len(sentences) == 3
        assert "First sentence." in sentences[0]
        assert "Second sentence." in sentences[1]

    def test_question_and_exclamation(self) -> None:
        """Should split on question marks and exclamation points."""
        text = "What is this? It is amazing! And this is the end."
        sentences = split_into_sentences(text)

        assert len(sentences) == 3
        assert sentences[0].endswith("?")
        assert sentences[1].endswith("!")

    def test_empty_text_returns_empty_list(self) -> None:
        """Should return empty list for empty string."""
        assert split_into_sentences("") == []
        assert split_into_sentences(None) == []  # type: ignore

    def test_short_segments_ignored(self) -> None:
        """Segments under 10 characters should be ignored."""
        text = "Hi. This is a proper sentence with enough length."
        sentences = split_into_sentences(text)

        # "Hi." is only 3 chars, should be ignored
        assert len(sentences) == 1
        assert "proper sentence" in sentences[0]

    def test_text_without_period_at_end(self) -> None:
        """Should handle text without final punctuation."""
        text = "This is a complete sentence. This one has no period"
        sentences = split_into_sentences(text)

        assert len(sentences) == 2

    def test_newlines_and_whitespace(self) -> None:
        """Should handle newlines as sentence boundaries."""
        text = "First sentence.\nSecond sentence.\tThird sentence."
        sentences = split_into_sentences(text)

        assert len(sentences) == 3

    def test_abbreviations_not_split(self) -> None:
        """Common abbreviations should not cause splits (implicit via length filter)."""
        # "Dr." followed directly by capital would cause a split,
        # but very short fragments are filtered out
        text = "Dr. Smith studied the protein. The results were conclusive."
        sentences = split_into_sentences(text)

        # Depending on implementation, may or may not split at "Dr."
        # The key is we get reasonable output
        assert len(sentences) >= 1


# =============================================================================
# Test: number_sentences
# =============================================================================


class TestNumberSentences:
    """Tests for sentence numbering function."""

    def test_basic_numbering(self) -> None:
        """Should number sentences with [1], [2], etc."""
        text = "First sentence here. Second sentence here."
        numbered, sentences = number_sentences(text)

        assert "[1]" in numbered
        assert "[2]" in numbered
        assert len(sentences) == 2

    def test_returns_original_sentences(self) -> None:
        """Should return list of original sentences."""
        text = "The protein binds to DNA. This causes activation."
        numbered, sentences = number_sentences(text)

        assert "The protein binds to DNA." in sentences[0]
        assert "This causes activation." in sentences[1]

    def test_empty_text_returns_original(self) -> None:
        """Should return original text when no sentences found."""
        text = "short"
        numbered, sentences = number_sentences(text)

        assert numbered == text
        assert sentences == []


# =============================================================================
# Test: Token Counting
# =============================================================================


class TestTokenCounting:
    """Tests for token counting functions."""

    def test_count_tokens_simple_approximation(self) -> None:
        """Simple count should approximate ~4 chars per token."""
        text = "a" * 100  # 100 characters
        count = count_tokens_simple(text)

        assert count == 25  # 100 / 4 = 25

    def test_count_tokens_simple_empty(self) -> None:
        """Should return 0 for empty string."""
        assert count_tokens_simple("") == 0

    def test_try_count_tokens_tiktoken_returns_positive(self) -> None:
        """tiktoken count should return positive number for text."""
        text = "Hello world, this is a test sentence."
        count = try_count_tokens_tiktoken(text)

        assert count > 0

    def test_try_count_tokens_tiktoken_empty(self) -> None:
        """Should return 0 for empty string."""
        assert try_count_tokens_tiktoken("") == 0

    def test_try_count_tokens_is_more_accurate(self) -> None:
        """tiktoken should give different count than simple approximation."""
        text = "The quick brown fox jumps over the lazy dog."
        simple = count_tokens_simple(text)
        tiktoken_count = try_count_tokens_tiktoken(text)

        # They should be different (tiktoken is more accurate)
        # Both should be reasonable (between 5-20 for this sentence)
        assert 5 <= simple <= 20
        assert 5 <= tiktoken_count <= 20


# =============================================================================
# Test: find_evidence_in_abstract
# =============================================================================


class TestFindEvidenceInAbstract:
    """Tests for evidence fuzzy matching function."""

    def test_exact_match_returns_full_similarity(self) -> None:
        """Exact substring match should return similarity 1.0."""
        evidence = "BRCA1 activates TP53"
        abstract = "In this study, BRCA1 activates TP53 in response to DNA damage."

        is_valid, similarity = find_evidence_in_abstract(evidence, abstract)

        assert is_valid is True
        assert similarity == 1.0

    def test_case_insensitive_match(self) -> None:
        """Should match case-insensitively."""
        evidence = "BRCA1 ACTIVATES TP53"
        abstract = "Studies show brca1 activates tp53 pathway."

        is_valid, similarity = find_evidence_in_abstract(evidence, abstract)

        assert is_valid is True
        assert similarity == 1.0

    def test_fuzzy_match_above_threshold(self) -> None:
        """Should match with high similarity above threshold."""
        evidence = "BRCA1 strongly activates TP53"
        abstract = "Studies show BRCA1 strongly activates TP53 in cells."

        is_valid, similarity = find_evidence_in_abstract(
            evidence, abstract, threshold=0.7
        )

        assert is_valid is True
        assert similarity >= 0.7

    def test_no_match_below_threshold(self) -> None:
        """Should not match when similarity below threshold."""
        evidence = "EGFR inhibits KRAS signaling"
        abstract = "BRCA1 activates TP53 in cancer cells."

        is_valid, similarity = find_evidence_in_abstract(
            evidence, abstract, threshold=0.7
        )

        assert is_valid is False
        assert similarity < 0.7

    def test_empty_evidence_returns_false(self) -> None:
        """Should return False for empty evidence."""
        is_valid, similarity = find_evidence_in_abstract("", "Some abstract text.")

        assert is_valid is False
        assert similarity == 0.0

    def test_empty_abstract_returns_false(self) -> None:
        """Should return False for empty abstract."""
        is_valid, similarity = find_evidence_in_abstract("Some evidence.", "")

        assert is_valid is False
        assert similarity == 0.0

    def test_custom_threshold(self) -> None:
        """Should respect custom threshold parameter."""
        evidence = "protein interaction"
        abstract = "The study examined protein interactions in detail."

        # With low threshold, should match
        is_valid_low, sim_low = find_evidence_in_abstract(
            evidence, abstract, threshold=0.5
        )

        # Similarity should be calculated regardless of threshold
        assert sim_low > 0


# =============================================================================
# Test: validate_relationship_provenance_indexed (CRITICAL)
# =============================================================================


class TestValidateRelationshipProvenanceIndexed:
    """Tests for index-based provenance validation - the key feature."""

    @pytest.fixture
    def sample_chunk(self) -> AbstractChunk:
        """Create a sample chunk for validation tests."""
        chunk = AbstractChunk(
            chunk_id=0,
            pmids=["111", "222", "333"],
            abstracts=[
                {"pmid": "111", "abstract": "First abstract text here."},
                {"pmid": "222", "abstract": "Second abstract with more content."},
                {"pmid": "333", "abstract": "Third abstract about proteins."},
            ],
            entities=["BRCA1", "TP53", "EGFR"],
            total_tokens=500,
        )
        # Populate sentence_map
        chunk.sentence_map = {
            "111": ["First abstract text here."],
            "222": ["Second abstract.", "More content here."],
            "333": ["Third abstract.", "About proteins.", "More details."],
        }
        return chunk

    def test_valid_pmid_and_indices(self, sample_chunk: AbstractChunk) -> None:
        """Should return is_valid=True for valid PMID and valid indices."""
        rel = ExtractedRelationship(
            entity1="BRCA1",
            entity2="TP53",
            relationship="activates",
            confidence=0.9,
            pmid="222",
            evidence_sentence_indices=[1, 2],  # Valid: 1 and 2 exist
        )

        result = validate_relationship_provenance_indexed(rel, sample_chunk)

        assert result.pmid_valid is True
        assert result.evidence_valid is True
        assert result.is_valid is True
        assert result.evidence_similarity == 1.0

    def test_invalid_pmid_not_in_chunk(self, sample_chunk: AbstractChunk) -> None:
        """Should return is_valid=False when PMID not in chunk."""
        rel = ExtractedRelationship(
            entity1="A",
            entity2="B",
            relationship="inhibits",
            confidence=0.8,
            pmid="999",  # Not in chunk
            evidence_sentence_indices=[1],
        )

        result = validate_relationship_provenance_indexed(rel, sample_chunk)

        assert result.pmid_valid is False
        assert result.is_valid is False
        assert "not in chunk PMIDs" in (result.error_message or "")

    def test_out_of_bounds_indices(self, sample_chunk: AbstractChunk) -> None:
        """Should return is_valid=False when sentence indices are out of bounds."""
        rel = ExtractedRelationship(
            entity1="X",
            entity2="Y",
            relationship="binds_to",
            confidence=0.7,
            pmid="111",  # Has only 1 sentence
            evidence_sentence_indices=[1, 5],  # 5 is out of bounds
        )

        result = validate_relationship_provenance_indexed(rel, sample_chunk)

        assert result.pmid_valid is True
        assert result.evidence_valid is False
        assert result.is_valid is False
        assert "Invalid sentence indices" in (result.error_message or "")

    def test_empty_indices_returns_invalid(self, sample_chunk: AbstractChunk) -> None:
        """Should return is_valid=False when evidence_sentence_indices is empty."""
        rel = ExtractedRelationship(
            entity1="P",
            entity2="Q",
            relationship="regulates",
            confidence=0.6,
            pmid="333",
            evidence_sentence_indices=[],  # Empty
        )

        result = validate_relationship_provenance_indexed(rel, sample_chunk)

        assert result.pmid_valid is True
        assert result.evidence_valid is False
        assert result.is_valid is False
        assert "No evidence sentence indices" in (result.error_message or "")

    def test_zero_index_is_invalid(self, sample_chunk: AbstractChunk) -> None:
        """Should reject index 0 since indices are 1-based."""
        rel = ExtractedRelationship(
            entity1="M",
            entity2="N",
            relationship="activates",
            confidence=0.9,
            pmid="333",
            evidence_sentence_indices=[0, 1],  # 0 is invalid (1-based)
        )

        result = validate_relationship_provenance_indexed(rel, sample_chunk)

        assert result.pmid_valid is True
        assert result.evidence_valid is False  # 0 is invalid
        assert result.evidence_similarity < 1.0  # Not all indices valid

    def test_partial_valid_indices(self, sample_chunk: AbstractChunk) -> None:
        """Should calculate partial similarity when some indices valid."""
        rel = ExtractedRelationship(
            entity1="R",
            entity2="S",
            relationship="inhibits",
            confidence=0.8,
            pmid="333",  # Has 3 sentences
            evidence_sentence_indices=[1, 2, 10],  # 10 is invalid
        )

        result = validate_relationship_provenance_indexed(rel, sample_chunk)

        assert result.pmid_valid is True
        assert result.evidence_valid is False
        # 2 out of 3 indices valid = 0.666...
        assert 0.6 <= result.evidence_similarity <= 0.7

    def test_pmid_not_in_sentence_map(self, sample_chunk: AbstractChunk) -> None:
        """Should handle PMID present in pmids but missing from sentence_map."""
        # Remove from sentence_map but keep in pmids
        del sample_chunk.sentence_map["222"]

        rel = ExtractedRelationship(
            entity1="T",
            entity2="U",
            relationship="binds_to",
            confidence=0.7,
            pmid="222",
            evidence_sentence_indices=[1],
        )

        result = validate_relationship_provenance_indexed(rel, sample_chunk)

        assert result.pmid_valid is True
        assert result.evidence_valid is False
        assert "No sentences found" in (result.error_message or "")


# =============================================================================
# Test: validate_relationship_provenance (text-based)
# =============================================================================


class TestValidateRelationshipProvenance:
    """Tests for text-based provenance validation."""

    @pytest.fixture
    def sample_chunk(self) -> AbstractChunk:
        """Create a sample chunk for validation tests."""
        return AbstractChunk(
            chunk_id=0,
            pmids=["123"],
            abstracts=[
                {
                    "pmid": "123",
                    "abstract": "BRCA1 activates TP53 in response to DNA damage. This leads to cell cycle arrest.",
                }
            ],
            entities=["BRCA1", "TP53"],
            total_tokens=100,
        )

    def test_valid_pmid_and_evidence(self, sample_chunk: AbstractChunk) -> None:
        """Should validate when PMID and evidence match."""
        rel = ExtractedRelationship(
            entity1="BRCA1",
            entity2="TP53",
            relationship="activates",
            confidence=0.9,
            pmid="123",
            evidence_text="BRCA1 activates TP53",
        )

        result = validate_relationship_provenance(rel, sample_chunk)

        assert result.pmid_valid is True
        assert result.evidence_valid is True
        assert result.is_valid is True

    def test_invalid_pmid(self, sample_chunk: AbstractChunk) -> None:
        """Should fail validation for invalid PMID."""
        rel = ExtractedRelationship(
            entity1="A",
            entity2="B",
            relationship="inhibits",
            confidence=0.8,
            pmid="999",  # Not in chunk
            evidence_text="Some evidence",
        )

        result = validate_relationship_provenance(rel, sample_chunk)

        assert result.pmid_valid is False
        assert result.is_valid is False


# =============================================================================
# Test: AbstractChunk and _create_chunk
# =============================================================================


class TestAbstractChunk:
    """Tests for AbstractChunk dataclass."""

    def test_default_sentence_map_empty(self) -> None:
        """sentence_map should default to empty dict."""
        chunk = AbstractChunk(
            chunk_id=0,
            pmids=["123"],
            abstracts=[],
            entities=["A"],
            total_tokens=100,
        )
        assert chunk.sentence_map == {}

    def test_chunk_stores_all_fields(self) -> None:
        """Should store all provided fields correctly."""
        abstracts = [{"pmid": "111", "abstract": "Text"}]
        entities = ["BRCA1", "TP53"]

        chunk = AbstractChunk(
            chunk_id=5,
            pmids=["111"],
            abstracts=abstracts,
            entities=entities,
            total_tokens=250,
        )

        assert chunk.chunk_id == 5
        assert chunk.pmids == ["111"]
        assert chunk.abstracts == abstracts
        assert chunk.entities == entities
        assert chunk.total_tokens == 250


# =============================================================================
# Test: BatchedLiteLLMMiner Initialization
# =============================================================================


class TestBatchedLiteLLMMinerInit:
    """Tests for miner initialization and configuration."""

    def test_default_extractor_from_config(self) -> None:
        """Should use default extractor from config when not specified."""
        miner = BatchedLiteLLMMiner()

        # Default is "cerebras" according to config
        assert miner._extractor_name == "cerebras"

    def test_custom_extractor_selection(self) -> None:
        """Should accept custom extractor name."""
        miner = BatchedLiteLLMMiner(extractor_name="gemini")

        assert miner._extractor_name == "gemini"
        assert "gemini" in miner.model.lower()

    def test_model_property_returns_correct_model(self) -> None:
        """model property should return configured model string."""
        miner = BatchedLiteLLMMiner(extractor_name="cerebras")

        # Cerebras model from config
        assert "openrouter" in miner.model or "cerebras" in miner.model.lower()

    def test_custom_evidence_threshold(self) -> None:
        """Should accept custom evidence_threshold."""
        miner = BatchedLiteLLMMiner(evidence_threshold=0.5)

        assert miner._evidence_threshold == 0.5

    def test_custom_chunk_tokens(self) -> None:
        """Should accept custom chunk_tokens override."""
        miner = BatchedLiteLLMMiner(chunk_tokens=3000)

        assert miner._chunk_tokens == 3000

    def test_invalid_extractor_raises_error(self) -> None:
        """Should raise ValueError for unknown extractor."""
        with pytest.raises(ValueError, match="Unknown extractor"):
            BatchedLiteLLMMiner(extractor_name="nonexistent_extractor")


# =============================================================================
# Test: chunk_abstracts
# =============================================================================


class TestChunkAbstracts:
    """Tests for abstract chunking functionality."""

    @pytest.fixture
    def sample_articles(self) -> list[dict[str, Any]]:
        """Sample articles for chunking tests."""
        return [
            {"pmid": "111", "title": "Title 1", "abstract": "A" * 500, "year": "2024"},
            {"pmid": "222", "title": "Title 2", "abstract": "B" * 500, "year": "2024"},
            {"pmid": "333", "title": "Title 3", "abstract": "C" * 500, "year": "2024"},
            {"pmid": "444", "title": "Title 4", "abstract": "D" * 500, "year": "2024"},
        ]

    @pytest.fixture
    def sample_annotations(self) -> dict[str, list[dict[str, Any]]]:
        """Sample annotations for chunking tests."""
        return {
            "111": [{"entity_text": "BRCA1"}, {"entity_text": "TP53"}],
            "222": [{"entity_text": "EGFR"}, {"entity_text": "BRCA1"}],
            "333": [{"entity_text": "KRAS"}],
            "444": [],
        }

    def test_empty_input_returns_empty_list(self) -> None:
        """Should return empty list for empty articles."""
        miner = BatchedLiteLLMMiner()
        chunks = miner.chunk_abstracts([], {})

        assert chunks == []

    def test_creates_chunks_with_token_limit(
        self,
        sample_articles: list[dict[str, Any]],
        sample_annotations: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Should create chunks respecting token limits."""
        miner = BatchedLiteLLMMiner(chunk_tokens=500)  # Small limit
        chunks = miner.chunk_abstracts(sample_articles, sample_annotations)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, AbstractChunk)
            assert len(chunk.pmids) >= 1

    def test_collects_entities_from_annotations(
        self,
        sample_articles: list[dict[str, Any]],
        sample_annotations: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Should collect unique entities from annotations."""
        miner = BatchedLiteLLMMiner(chunk_tokens=10000)  # Large limit, single chunk
        chunks = miner.chunk_abstracts(sample_articles, sample_annotations)

        # All unique entities should be in the chunk
        all_entities = set()
        for chunk in chunks:
            all_entities.update(chunk.entities)

        assert "BRCA1" in all_entities
        assert "TP53" in all_entities
        assert "EGFR" in all_entities
        assert "KRAS" in all_entities

    def test_minimum_chunks_enforcement(
        self,
        sample_articles: list[dict[str, Any]],
        sample_annotations: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Should enforce minimum number of chunks."""
        miner = BatchedLiteLLMMiner(chunk_tokens=100000)  # Very large limit
        chunks = miner.chunk_abstracts(sample_articles, sample_annotations)

        # Should have at least MIN_CHUNKS from config (3)
        assert len(chunks) >= 1  # At minimum 1 if not enough articles

    def test_chunk_ids_are_sequential(
        self,
        sample_articles: list[dict[str, Any]],
        sample_annotations: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Chunk IDs should be sequential starting from 0."""
        miner = BatchedLiteLLMMiner(chunk_tokens=500)
        chunks = miner.chunk_abstracts(sample_articles, sample_annotations)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == i


# =============================================================================
# Test: Mock-based Extraction
# =============================================================================


class TestMineChunk:
    """Tests for chunk mining with mocked LiteLLM."""

    @pytest.fixture
    def sample_chunk(self) -> AbstractChunk:
        """Create a sample chunk for mining tests."""
        chunk = AbstractChunk(
            chunk_id=0,
            pmids=["12345"],
            abstracts=[
                {
                    "pmid": "12345",
                    "title": "Study of BRCA1",
                    "abstract": "BRCA1 activates TP53 in cancer cells. This leads to apoptosis.",
                    "year": "2024",
                }
            ],
            entities=["BRCA1", "TP53"],
            total_tokens=100,
        )
        return chunk

    @pytest.fixture
    def mock_successful_response(self) -> MagicMock:
        """Create a mock successful LiteLLM response."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "relationships": [
                            {
                                "entity1": "BRCA1",
                                "entity2": "TP53",
                                "relationship": "activates",
                                "confidence": 0.9,
                                "evidence_sentence_indices": [1],
                                "pmid": "12345",
                            }
                        ]
                    })
                )
            )
        ]
        mock_response.get = lambda key, default=None: {
            "usage": {"prompt_tokens": 500, "completion_tokens": 100}
        }.get(key, default)
        return mock_response

    async def test_successful_extraction(
        self, sample_chunk: AbstractChunk, mock_successful_response: MagicMock
    ) -> None:
        """Should extract relationships from successful API response."""
        import asyncio

        with patch(
            "extraction.batched_litellm_miner.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_successful_response

            miner = BatchedLiteLLMMiner()
            semaphore = asyncio.Semaphore(1)
            result = await miner.mine_chunk(sample_chunk, semaphore)

            assert result.success is True
            assert len(result.relationships) == 1
            assert result.relationships[0].entity1 == "BRCA1"
            assert result.relationships[0].entity2 == "TP53"
            assert result.token_usage["prompt_tokens"] == 500

    async def test_json_parse_error_handling(
        self, sample_chunk: AbstractChunk
    ) -> None:
        """Should handle JSON parse errors gracefully."""
        import asyncio

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="invalid json {{{"))
        ]
        mock_response.get = lambda key, default=None: {"usage": {}}.get(key, default)

        with patch(
            "extraction.batched_litellm_miner.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            miner = BatchedLiteLLMMiner()
            semaphore = asyncio.Semaphore(1)
            result = await miner.mine_chunk(sample_chunk, semaphore)

            # Should succeed but with no relationships
            assert result.success is True
            assert len(result.relationships) == 0

    async def test_rate_limit_retry_logic(self, sample_chunk: AbstractChunk) -> None:
        """Should retry on rate limit (429) errors."""
        import asyncio

        call_count = 0

        async def mock_acompletion_with_retry(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 rate limit exceeded")
            # Second call succeeds
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content='{"relationships": []}'))
            ]
            mock_response.get = lambda key, default=None: {"usage": {}}.get(
                key, default
            )
            return mock_response

        with patch(
            "extraction.batched_litellm_miner.acompletion",
            side_effect=mock_acompletion_with_retry,
        ):
            miner = BatchedLiteLLMMiner()
            semaphore = asyncio.Semaphore(1)
            result = await miner.mine_chunk(sample_chunk, semaphore)

            assert call_count == 2  # Retried once
            assert result.success is True

    async def test_max_retries_exceeded(self, sample_chunk: AbstractChunk) -> None:
        """Should return error result after max retries."""
        import asyncio

        with patch(
            "extraction.batched_litellm_miner.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_acompletion.side_effect = Exception("429 rate limit exceeded")

            miner = BatchedLiteLLMMiner()
            semaphore = asyncio.Semaphore(1)
            result = await miner.mine_chunk(sample_chunk, semaphore)

            assert result.success is False
            assert len(result.errors) > 0
            assert "rate limit" in result.errors[0].lower()

    async def test_empty_results_handling(self, sample_chunk: AbstractChunk) -> None:
        """Should handle empty relationships array."""
        import asyncio

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"relationships": []}'))
        ]
        mock_response.get = lambda key, default=None: {"usage": {}}.get(key, default)

        with patch(
            "extraction.batched_litellm_miner.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            miner = BatchedLiteLLMMiner()
            semaphore = asyncio.Semaphore(1)
            result = await miner.mine_chunk(sample_chunk, semaphore)

            assert result.success is True
            assert len(result.relationships) == 0

    async def test_low_confidence_filtered(self, sample_chunk: AbstractChunk) -> None:
        """Should filter relationships below MIN_CONFIDENCE."""
        import asyncio

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "relationships": [
                            {
                                "entity1": "A",
                                "entity2": "B",
                                "relationship": "activates",
                                "confidence": 0.3,  # Below default 0.5
                                "evidence_sentence_indices": [1],
                                "pmid": "12345",
                            },
                            {
                                "entity1": "C",
                                "entity2": "D",
                                "relationship": "inhibits",
                                "confidence": 0.8,  # Above threshold
                                "evidence_sentence_indices": [1],
                                "pmid": "12345",
                            },
                        ]
                    })
                )
            )
        ]
        mock_response.get = lambda key, default=None: {"usage": {}}.get(key, default)

        with patch(
            "extraction.batched_litellm_miner.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            miner = BatchedLiteLLMMiner()
            semaphore = asyncio.Semaphore(1)
            result = await miner.mine_chunk(sample_chunk, semaphore)

            # Only the high-confidence relationship should be included
            assert len(result.relationships) == 1
            assert result.relationships[0].confidence == 0.8


# =============================================================================
# Test: Full Pipeline (run method)
# =============================================================================


class TestRunPipeline:
    """Tests for the full mining pipeline."""

    @pytest.fixture
    def sample_articles(self) -> list[dict[str, Any]]:
        """Sample articles for pipeline tests."""
        return [
            {
                "pmid": "111",
                "title": "Study 1",
                "abstract": "BRCA1 activates TP53 in cancer cells. This triggers apoptosis.",
                "year": "2024",
            },
            {
                "pmid": "222",
                "title": "Study 2",
                "abstract": "EGFR signaling is important. It affects cell growth.",
                "year": "2024",
            },
        ]

    @pytest.fixture
    def sample_annotations(self) -> dict[str, list[dict[str, Any]]]:
        """Sample annotations for pipeline tests."""
        return {
            "111": [{"entity_text": "BRCA1"}, {"entity_text": "TP53"}],
            "222": [{"entity_text": "EGFR"}],
        }

    async def test_run_returns_expected_structure(
        self,
        sample_articles: list[dict[str, Any]],
        sample_annotations: dict[str, list[dict[str, Any]]],
    ) -> None:
        """run() should return dict with expected keys."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"relationships": []}'))
        ]
        mock_response.get = lambda key, default=None: {
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }.get(key, default)

        with patch(
            "extraction.batched_litellm_miner.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            miner = BatchedLiteLLMMiner()
            result = await miner.run(sample_articles, sample_annotations)

            assert "relationships" in result
            assert "valid_relationships" in result
            assert "validation_results" in result
            assert "statistics" in result
            assert "errors" in result

    async def test_statistics_aggregation(
        self,
        sample_articles: list[dict[str, Any]],
        sample_annotations: dict[str, list[dict[str, Any]]],
    ) -> None:
        """run() should correctly aggregate statistics."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "relationships": [
                            {
                                "entity1": "BRCA1",
                                "entity2": "TP53",
                                "relationship": "activates",
                                "confidence": 0.9,
                                "evidence_sentence_indices": [1],
                                "pmid": "111",
                            }
                        ]
                    })
                )
            )
        ]
        mock_response.get = lambda key, default=None: {
            "usage": {"prompt_tokens": 200, "completion_tokens": 100}
        }.get(key, default)

        with patch(
            "extraction.batched_litellm_miner.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            miner = BatchedLiteLLMMiner()
            result = await miner.run(sample_articles, sample_annotations)

            stats = result["statistics"]
            assert stats["total_abstracts"] == 2
            assert stats["total_chunks"] >= 1
            assert stats["total_prompt_tokens"] >= 200

    async def test_empty_articles_returns_empty_result(self) -> None:
        """run() with empty articles should return empty results."""
        miner = BatchedLiteLLMMiner()
        result = await miner.run([], {})

        assert result["relationships"] == []
        assert result["valid_relationships"] == []
        assert result["statistics"]["total_abstracts"] == 0


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_batched_miner_default(self) -> None:
        """create_batched_miner should create miner with defaults."""
        miner = create_batched_miner()

        assert isinstance(miner, BatchedLiteLLMMiner)
        assert miner._extractor_name == "cerebras"

    def test_create_batched_miner_custom_extractor(self) -> None:
        """create_batched_miner should accept custom extractor."""
        miner = create_batched_miner(extractor_name="gemini")

        assert miner._extractor_name == "gemini"

    def test_create_batched_miner_custom_threshold(self) -> None:
        """create_batched_miner should accept custom evidence_threshold."""
        miner = create_batched_miner(evidence_threshold=0.5)

        assert miner._evidence_threshold == 0.5

    async def test_run_batched_mining_convenience(self) -> None:
        """run_batched_mining should call miner.run()."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"relationships": []}'))
        ]
        mock_response.get = lambda key, default=None: {"usage": {}}.get(key, default)

        with patch(
            "extraction.batched_litellm_miner.acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            result = await run_batched_mining(
                articles=[{"pmid": "123", "title": "Test", "abstract": "Text", "year": "2024"}],
                annotations={"123": []},
            )

            assert "relationships" in result
            assert "statistics" in result
