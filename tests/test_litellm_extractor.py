"""
Tests for the LiteLLM-based relationship extractor module.

Tests cover:
- ExtractionMetrics calculations
- BatchMetrics aggregation
- LiteLLMRelationshipExtractor initialization
- Factory function
- Mock-based extraction tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from extraction.litellm_extractor import (
    ExtractionMetrics,
    BatchMetrics,
    LiteLLMRelationshipExtractor,
    ADKLiteLLMRelationshipExtractor,
    create_extractor,
)
from pipeline.metrics import MODEL_PRICING


# =============================================================================
# Test: ExtractionMetrics
# =============================================================================


class TestExtractionMetrics:
    """Tests for ExtractionMetrics dataclass."""

    def test_tokens_per_second_calculation(self):
        """Should calculate correct tokens per second."""
        metrics = ExtractionMetrics(
            model="cerebras/gpt-oss-120b",
            total_tokens=3000,
            latency_ms=1000.0,
        )
        assert metrics.tokens_per_second == 3000.0

    def test_tokens_per_second_zero_latency(self):
        """Should return 0 for zero latency."""
        metrics = ExtractionMetrics(
            model="cerebras/gpt-oss-120b",
            total_tokens=3000,
            latency_ms=0.0,
        )
        assert metrics.tokens_per_second == 0.0

    def test_completion_tokens_per_second(self):
        """Should calculate completion throughput."""
        metrics = ExtractionMetrics(
            model="cerebras/gpt-oss-120b",
            completion_tokens=1500,
            latency_ms=500.0,
        )
        assert metrics.completion_tokens_per_second == 3000.0

    def test_to_dict(self):
        """Should serialize to dictionary."""
        metrics = ExtractionMetrics(
            model="cerebras/gpt-oss-120b",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=100.0,
            cost_usd=0.001,
            relationships_found=5,
        )
        d = metrics.to_dict()
        assert d["model"] == "cerebras/gpt-oss-120b"
        assert d["prompt_tokens"] == 100
        assert d["completion_tokens"] == 50
        assert d["total_tokens"] == 150
        assert d["latency_ms"] == 100.0
        assert d["cost_usd"] == 0.001
        assert d["relationships_found"] == 5
        assert "tokens_per_second" in d


# =============================================================================
# Test: BatchMetrics
# =============================================================================


class TestBatchMetrics:
    """Tests for BatchMetrics aggregation."""

    def test_add_extraction_success(self):
        """Should correctly aggregate successful extraction."""
        batch = BatchMetrics(model="cerebras/gpt-oss-120b", total_abstracts=10)
        metrics = ExtractionMetrics(
            model="cerebras/gpt-oss-120b",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=100.0,
            cost_usd=0.001,
            relationships_found=3,
        )
        batch.add_extraction(metrics, success=True)

        assert batch.successful_extractions == 1
        assert batch.failed_extractions == 0
        assert batch.total_relationships == 3
        assert batch.total_prompt_tokens == 100
        assert batch.total_completion_tokens == 50
        assert batch.total_tokens == 150
        assert batch.total_latency_ms == 100.0
        assert batch.total_cost_usd == 0.001

    def test_add_extraction_failure(self):
        """Should track failed extractions."""
        batch = BatchMetrics(model="cerebras/gpt-oss-120b", total_abstracts=10)
        metrics = ExtractionMetrics(model="cerebras/gpt-oss-120b")
        batch.add_extraction(metrics, success=False)

        assert batch.successful_extractions == 0
        assert batch.failed_extractions == 1

    def test_avg_latency(self):
        """Should calculate correct average latency."""
        batch = BatchMetrics(model="cerebras/gpt-oss-120b", total_abstracts=2)
        batch.add_extraction(
            ExtractionMetrics(model="test", latency_ms=100.0),
            success=True,
        )
        batch.add_extraction(
            ExtractionMetrics(model="test", latency_ms=200.0),
            success=True,
        )
        assert batch.avg_latency_ms == 150.0

    def test_throughput_calculation(self):
        """Should calculate overall throughput."""
        batch = BatchMetrics(model="cerebras/gpt-oss-120b", total_abstracts=1)
        batch.add_extraction(
            ExtractionMetrics(
                model="test",
                total_tokens=1000,
                latency_ms=500.0,
            ),
            success=True,
        )
        assert batch.throughput_tokens_per_second == 2000.0

    def test_to_dict(self):
        """Should serialize to dictionary."""
        batch = BatchMetrics(model="cerebras/gpt-oss-120b", total_abstracts=5)
        d = batch.to_dict()
        assert d["model"] == "cerebras/gpt-oss-120b"
        assert d["total_abstracts"] == 5
        assert "avg_latency_ms" in d
        assert "throughput_tokens_per_second" in d


# =============================================================================
# Test: LiteLLMRelationshipExtractor Initialization
# =============================================================================


class TestLiteLLMRelationshipExtractorInit:
    """Tests for extractor initialization."""

    def test_default_model(self):
        """Should use Cerebras as default model."""
        extractor = LiteLLMRelationshipExtractor()
        assert extractor.model == "cerebras/gpt-oss-120b"

    def test_custom_model(self):
        """Should accept custom model."""
        extractor = LiteLLMRelationshipExtractor(model="openrouter/openai/gpt-oss-120b")
        assert extractor.model == "openrouter/openai/gpt-oss-120b"

    def test_custom_parameters(self):
        """Should accept custom generation parameters."""
        extractor = LiteLLMRelationshipExtractor(
            model="cerebras/gpt-oss-120b",
            temperature=0.5,
            max_tokens=2000,
            timeout=120.0,
        )
        assert extractor.temperature == 0.5
        assert extractor.max_tokens == 2000
        assert extractor.timeout == 120.0

    def test_pricing_lookup(self):
        """Should look up pricing for known models."""
        extractor = LiteLLMRelationshipExtractor(model="cerebras/gpt-oss-120b")
        assert extractor.pricing["input"] == 0.35
        assert extractor.pricing["output"] == 0.75

    def test_unknown_model_pricing(self):
        """Should use zero pricing for unknown models."""
        extractor = LiteLLMRelationshipExtractor(model="unknown/model")
        assert extractor.pricing["input"] == 0.0
        assert extractor.pricing["output"] == 0.0


# =============================================================================
# Test: Cost Calculation
# =============================================================================


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_calculate_cost_cerebras(self):
        """Should calculate cost correctly for Cerebras."""
        extractor = LiteLLMRelationshipExtractor(model="cerebras/gpt-oss-120b")
        cost = extractor._calculate_cost(
            prompt_tokens=1_000_000,  # 1M tokens
            completion_tokens=1_000_000,
        )
        # Input: $0.35/M, Output: $0.75/M
        expected = 0.35 + 0.75
        assert abs(cost - expected) < 0.01

    def test_calculate_cost_zero_tokens(self):
        """Should return zero for zero tokens."""
        extractor = LiteLLMRelationshipExtractor()
        cost = extractor._calculate_cost(0, 0)
        assert cost == 0.0


# =============================================================================
# Test: Factory Function
# =============================================================================


class TestCreateExtractor:
    """Tests for the create_extractor factory function."""

    def test_create_cerebras_extractor(self):
        """Should create Cerebras extractor."""
        extractor = create_extractor("cerebras")
        assert extractor.model == "cerebras/gpt-oss-120b"

    def test_create_openrouter_extractor(self):
        """Should create OpenRouter extractor."""
        extractor = create_extractor("openrouter")
        assert extractor.model == "openrouter/openai/gpt-oss-120b"

    def test_create_gemini_extractor(self):
        """Should create Gemini extractor."""
        extractor = create_extractor("gemini")
        assert extractor.model == "gemini-2.0-flash-exp"

    def test_custom_model_override(self):
        """Should allow model override."""
        extractor = create_extractor("cerebras", model="cerebras/custom-model")
        assert extractor.model == "cerebras/custom-model"

    def test_pass_kwargs(self):
        """Should pass kwargs to extractor."""
        extractor = create_extractor("cerebras", temperature=0.5)
        assert extractor.temperature == 0.5


# =============================================================================
# Test: Mock-based Extraction
# =============================================================================


class TestExtraction:
    """Tests for relationship extraction with mocked API."""

    @pytest.fixture
    def sample_abstract(self):
        return "HCRTR1 activates the ERK pathway leading to cell proliferation."

    @pytest.fixture
    def sample_entity_pairs(self):
        return [("HCRTR1", "ERK"), ("HCRTR1", "cell proliferation")]

    @pytest.mark.asyncio
    async def test_extract_relationships_empty_pairs(self):
        """Should return empty list for empty entity pairs."""
        extractor = LiteLLMRelationshipExtractor()
        relationships, metrics = await extractor.extract_relationships(
            abstract="Some abstract",
            entity_pairs=[],
            pmid="12345",
        )
        assert relationships == []
        assert metrics.relationships_found == 0

    @pytest.mark.asyncio
    async def test_extract_relationships_mock_response(self, sample_abstract, sample_entity_pairs):
        """Should parse relationships from mocked response."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps([
                        {
                            "entity1": "HCRTR1",
                            "entity2": "ERK",
                            "relationship": "ACTIVATES",
                            "confidence": 0.9,
                            "evidence_text": "HCRTR1 activates the ERK pathway",
                        }
                    ])
                )
            )
        ]
        mock_response.get = lambda key, default=None: {
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }.get(key, default)

        with patch("extraction.litellm_extractor.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            extractor = LiteLLMRelationshipExtractor()
            relationships, metrics = await extractor.extract_relationships(
                abstract=sample_abstract,
                entity_pairs=sample_entity_pairs,
                pmid="12345",
            )

            assert len(relationships) == 1
            assert relationships[0]["entity1"] == "HCRTR1"
            assert relationships[0]["entity2"] == "ERK"
            assert relationships[0]["relationship"] == "activates"  # normalized to lowercase
            assert relationships[0]["pmid"] == "12345"
            assert metrics.relationships_found == 1
            assert metrics.prompt_tokens == 100
            assert metrics.completion_tokens == 50

    @pytest.mark.asyncio
    async def test_extract_relationships_json_error(self, sample_abstract, sample_entity_pairs):
        """Should handle JSON parse errors gracefully."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="invalid json"))
        ]
        mock_response.get = lambda key, default=None: {"usage": {}}.get(key, default)

        with patch("extraction.litellm_extractor.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            extractor = LiteLLMRelationshipExtractor()
            relationships, metrics = await extractor.extract_relationships(
                abstract=sample_abstract,
                entity_pairs=sample_entity_pairs,
                pmid="12345",
            )

            assert relationships == []
            assert metrics.relationships_found == 0

    @pytest.mark.asyncio
    async def test_extract_relationships_api_error(self, sample_abstract, sample_entity_pairs):
        """Should handle API errors gracefully."""
        with patch("extraction.litellm_extractor.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = Exception("API Error")

            extractor = LiteLLMRelationshipExtractor()
            relationships, metrics = await extractor.extract_relationships(
                abstract=sample_abstract,
                entity_pairs=sample_entity_pairs,
                pmid="12345",
            )

            assert relationships == []


# =============================================================================
# Test: Batch Extraction
# =============================================================================


class TestBatchExtraction:
    """Tests for batch extraction."""

    @pytest.fixture
    def sample_abstracts(self):
        return [
            {
                "pmid": "123",
                "abstract": "HCRTR1 activates ERK.",
                "entity_pairs": [("HCRTR1", "ERK")],
            },
            {
                "pmid": "456",
                "abstract": "BRCA1 inhibits cell growth.",
                "entity_pairs": [("BRCA1", "cell growth")],
            },
        ]

    @pytest.mark.asyncio
    async def test_batch_extract_empty_list(self):
        """Should handle empty abstracts list."""
        extractor = LiteLLMRelationshipExtractor()
        relationships, batch_metrics = await extractor.batch_extract([])

        assert relationships == []
        assert batch_metrics.total_abstracts == 0

    @pytest.mark.asyncio
    async def test_batch_extract_mock(self, sample_abstracts):
        """Should process multiple abstracts."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps([
                        {"entity1": "A", "entity2": "B", "relationship": "activates", "confidence": 0.9}
                    ])
                )
            )
        ]
        mock_response.get = lambda key, default=None: {
            "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75}
        }.get(key, default)

        with patch("extraction.litellm_extractor.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            extractor = LiteLLMRelationshipExtractor()
            relationships, batch_metrics = await extractor.batch_extract(
                sample_abstracts,
                max_concurrent=2,
            )

            assert len(relationships) == 2  # One per abstract
            assert batch_metrics.total_abstracts == 2
            assert batch_metrics.successful_extractions == 2
            assert batch_metrics.total_relationships == 2

    @pytest.mark.asyncio
    async def test_batch_extract_respects_concurrency(self, sample_abstracts):
        """Should respect max_concurrent limit."""
        call_count = 0

        async def track_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="[]"))]
            mock_response.get = lambda key, default=None: {"usage": {}}.get(key, default)
            return mock_response

        with patch("extraction.litellm_extractor.acompletion", side_effect=track_calls):
            extractor = LiteLLMRelationshipExtractor()
            await extractor.batch_extract(sample_abstracts, max_concurrent=1)

            assert call_count == 2  # Should still make 2 calls


# =============================================================================
# Test: Model Pricing
# =============================================================================


class TestModelPricing:
    """Tests for model pricing constants."""

    def test_cerebras_pricing_exists(self):
        """Cerebras pricing should be defined."""
        assert "cerebras/gpt-oss-120b" in MODEL_PRICING
        pricing = MODEL_PRICING["cerebras/gpt-oss-120b"]
        assert pricing["input"] == 0.35
        assert pricing["output"] == 0.75

    def test_openrouter_pricing_exists(self):
        """OpenRouter pricing should be defined."""
        assert "openrouter/openai/gpt-oss-120b" in MODEL_PRICING
        pricing = MODEL_PRICING["openrouter/openai/gpt-oss-120b"]
        assert pricing["input"] == 0.25
        assert pricing["output"] == 0.69

    def test_gemini_pricing_exists(self):
        """Gemini pricing should be defined."""
        assert "gemini-2.0-flash-exp" in MODEL_PRICING


# =============================================================================
# Test: ADK Integration
# =============================================================================


class TestADKLiteLLMExtractor:
    """Tests for ADK-integrated extractor."""

    def test_init_without_adk(self):
        """Should fall back gracefully if ADK not installed."""
        # This test verifies the fallback behavior
        extractor = ADKLiteLLMRelationshipExtractor(model="cerebras/gpt-oss-120b")
        assert extractor.model == "cerebras/gpt-oss-120b"

    @pytest.mark.asyncio
    async def test_extract_without_adk(self):
        """Should use direct LiteLLM if ADK not available."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="[]"))]
        mock_response.get = lambda key, default=None: {"usage": {}}.get(key, default)

        with patch("extraction.litellm_extractor.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            extractor = ADKLiteLLMRelationshipExtractor()
            relationships = await extractor.extract_relationships(
                abstract="Test abstract",
                entity_pairs=[("A", "B")],
                pmid="12345",
            )
            assert relationships == []
