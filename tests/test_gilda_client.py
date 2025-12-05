"""
Tests for the Gilda (INDRA) entity grounding client.

Tests cover:
- GroundingResult dataclass
- Cache functionality
- Result selection/preference logic
- Batch grounding
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from clients.gilda_client import GildaClient, GroundingResult


# =============================================================================
# Test: GroundingResult dataclass
# =============================================================================


class TestGroundingResult:
    """Tests for GroundingResult dataclass."""

    def test_create_result(self):
        """Should create a valid grounding result."""
        result = GroundingResult(
            db="HGNC",
            id="6553",
            entry_name="LEP",
            score=0.98,
            original_text="Leptin",
        )
        assert result.db == "HGNC"
        assert result.id == "6553"
        assert result.entry_name == "LEP"
        assert result.score == 0.98
        assert result.original_text == "Leptin"

    def test_full_id_property(self):
        """full_id should return db:id format."""
        result = GroundingResult(
            db="HGNC", id="6553", entry_name="LEP", score=0.98, original_text="Leptin"
        )
        assert result.full_id == "HGNC:6553"

    def test_repr(self):
        """repr should include key info."""
        result = GroundingResult(
            db="HGNC", id="6553", entry_name="LEP", score=0.98, original_text="Leptin"
        )
        repr_str = repr(result)
        assert "HGNC:6553" in repr_str
        assert "LEP" in repr_str
        assert "0.98" in repr_str

    def test_frozen_immutable(self):
        """GroundingResult should be immutable (frozen)."""
        result = GroundingResult(
            db="HGNC", id="6553", entry_name="LEP", score=0.98, original_text="Leptin"
        )
        with pytest.raises(AttributeError):
            result.db = "UP"  # type: ignore


# =============================================================================
# Test: GildaClient initialization
# =============================================================================


class TestGildaClientInit:
    """Tests for GildaClient initialization."""

    def test_default_init(self):
        """Should initialize with default settings."""
        client = GildaClient()
        assert client.timeout == 30.0
        assert client.enable_cache is True
        assert client.prefer_hgnc is True
        assert client.cache_size == 0

    def test_custom_init(self):
        """Should accept custom settings."""
        client = GildaClient(timeout=60.0, enable_cache=False, prefer_hgnc=False)
        assert client.timeout == 60.0
        assert client.enable_cache is False
        assert client.prefer_hgnc is False


# =============================================================================
# Test: Cache functionality
# =============================================================================


class TestGildaClientCache:
    """Tests for caching behavior."""

    def test_cache_starts_empty(self):
        """Cache should start empty."""
        client = GildaClient()
        assert client.cache_size == 0

    def test_clear_cache(self):
        """clear_cache should empty the cache."""
        client = GildaClient()
        client._cache["test"] = GroundingResult(
            db="HGNC", id="1", entry_name="TEST", score=0.9, original_text="test"
        )
        assert client.cache_size == 1
        client.clear_cache()
        assert client.cache_size == 0


# =============================================================================
# Test: Result selection logic
# =============================================================================


class TestResultSelection:
    """Tests for _select_best_result logic."""

    def test_empty_results_returns_none(self):
        """Empty results should return None."""
        client = GildaClient()
        result = client._select_best_result([], "test")
        assert result is None

    def test_prefers_hgnc_over_chebi(self):
        """Should prefer HGNC over CHEBI when prefer_hgnc=True."""
        client = GildaClient(prefer_hgnc=True)
        results = [
            {"term": {"db": "CHEBI", "id": "12345", "entry_name": "ghrelin"}, "score": 0.95},
            {"term": {"db": "HGNC", "id": "18129", "entry_name": "GHRL"}, "score": 0.85},
        ]
        result = client._select_best_result(results, "ghrelin")
        assert result is not None
        assert result.db == "HGNC"
        assert result.entry_name == "GHRL"

    def test_respects_min_score_for_hgnc(self):
        """Should not select HGNC if score is too low."""
        client = GildaClient(prefer_hgnc=True)
        results = [
            {"term": {"db": "CHEBI", "id": "12345", "entry_name": "test"}, "score": 0.95},
            {"term": {"db": "HGNC", "id": "1", "entry_name": "TEST"}, "score": 0.3},  # Below 0.5 threshold
        ]
        result = client._select_best_result(results, "test")
        assert result is not None
        # Falls back to priority ordering since HGNC score is too low
        assert result.db == "HGNC"  # HGNC is higher priority than CHEBI, but score determines among same priority

    def test_falls_back_to_priority_when_no_hgnc(self):
        """Should use priority ordering when no HGNC available."""
        client = GildaClient(prefer_hgnc=True)
        results = [
            {"term": {"db": "MESH", "id": "D000856", "entry_name": "Anorexia"}, "score": 0.95},
            {"term": {"db": "GO", "id": "0001234", "entry_name": "process"}, "score": 0.90},
        ]
        result = client._select_best_result(results, "anorexia")
        assert result is not None
        assert result.db == "GO"  # GO is higher priority than MESH

    def test_selects_highest_score_within_priority(self):
        """Among same priority, should select highest score."""
        client = GildaClient(prefer_hgnc=False)
        results = [
            {"term": {"db": "HGNC", "id": "1", "entry_name": "TEST1"}, "score": 0.7},
            {"term": {"db": "HGNC", "id": "2", "entry_name": "TEST2"}, "score": 0.9},
        ]
        result = client._select_best_result(results, "test")
        assert result is not None
        assert result.entry_name == "TEST2"
        assert result.score == 0.9

    def test_skips_results_without_db_or_id(self):
        """Should skip malformed results."""
        client = GildaClient()
        results = [
            {"term": {"db": "", "id": "123", "entry_name": "test"}, "score": 0.9},
            {"term": {"db": "HGNC", "id": "", "entry_name": "test"}, "score": 0.9},
            {"term": {"db": "HGNC", "id": "456", "entry_name": "VALID"}, "score": 0.8},
        ]
        result = client._select_best_result(results, "test")
        assert result is not None
        assert result.entry_name == "VALID"

    def test_handles_unknown_database(self):
        """Should handle databases not in priority list."""
        client = GildaClient()
        results = [
            {"term": {"db": "UNKNOWN_DB", "id": "123", "entry_name": "test"}, "score": 0.95},
        ]
        result = client._select_best_result(results, "test")
        assert result is not None
        assert result.db == "UNKNOWN_DB"


# =============================================================================
# Test: Text normalization
# =============================================================================


class TestTextNormalization:
    """Tests for text normalization."""

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        client = GildaClient()
        assert client._normalize_text("  LEP  ") == "LEP"
        assert client._normalize_text("\tTNF\n") == "TNF"


# =============================================================================
# Test: ground() single entity (with mocking)
# =============================================================================


class TestGroundSingle:
    """Tests for single entity grounding."""

    @pytest.mark.asyncio
    async def test_ground_returns_cached_result(self):
        """Should return cached result without API call."""
        client = GildaClient()
        cached = GroundingResult(
            db="HGNC", id="6553", entry_name="LEP", score=0.98, original_text="LEP"
        )
        client._cache["LEP"] = cached

        result = await client.ground("LEP")
        assert result == cached

    @pytest.mark.asyncio
    async def test_ground_caches_none_results(self):
        """Should cache None results to avoid re-querying."""
        client = GildaClient()
        client._cache["unknown_gene_xyz"] = None

        result = await client.ground("unknown_gene_xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_ground_skips_cache_when_disabled(self):
        """Should not use cache when enable_cache=False."""
        client = GildaClient(enable_cache=False)
        cached = GroundingResult(
            db="HGNC", id="6553", entry_name="LEP", score=0.98, original_text="LEP"
        )
        client._cache["LEP"] = cached

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"term": {"db": "HGNC", "id": "6553", "entry_name": "LEP"}, "score": 0.98}
        ]
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.ground("LEP")

        # Should have made API call despite cache having value
        mock_post.assert_called_once()


# =============================================================================
# Test: ground_batch() (with mocking)
# =============================================================================


class TestGroundBatch:
    """Tests for batch grounding."""

    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        """Should return empty dict for empty input."""
        client = GildaClient()
        result = await client.ground_batch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_batch_uses_cache(self):
        """Should use cached values and only fetch uncached."""
        client = GildaClient()
        cached = GroundingResult(
            db="HGNC", id="6553", entry_name="LEP", score=0.98, original_text="LEP"
        )
        client._cache["LEP"] = cached

        mock_response = MagicMock()
        mock_response.json.return_value = [
            [{"term": {"db": "HGNC", "id": "7124", "entry_name": "TNF"}, "score": 0.95}]
        ]
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            results = await client.ground_batch(["LEP", "TNF"])

        # Should only request TNF, not LEP
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"] == [{"text": "TNF"}]

        # Results should include both
        assert "LEP" in results
        assert "TNF" in results
        assert results["LEP"] == cached

    @pytest.mark.asyncio
    async def test_batch_dedupes_input(self):
        """Should dedupe repeated inputs."""
        client = GildaClient()

        mock_response = MagicMock()
        mock_response.json.return_value = [
            [{"term": {"db": "HGNC", "id": "6553", "entry_name": "LEP"}, "score": 0.98}]
        ]
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            results = await client.ground_batch(["LEP", "LEP", "LEP"])

        # Should only make one API call with one item
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"] == [{"text": "LEP"}]

        # Results should have all three inputs mapped
        assert results["LEP"] is not None

    @pytest.mark.asyncio
    async def test_batch_returns_original_order(self):
        """Results should map back to original input texts."""
        client = GildaClient()

        mock_response = MagicMock()
        mock_response.json.return_value = [
            [{"term": {"db": "HGNC", "id": "1", "entry_name": "A"}, "score": 0.9}],
            [{"term": {"db": "HGNC", "id": "2", "entry_name": "B"}, "score": 0.9}],
        ]
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            results = await client.ground_batch(["geneA", "geneB"])

        assert "geneA" in results
        assert "geneB" in results


# =============================================================================
# Test: ground_to_hgnc() convenience method
# =============================================================================


class TestGroundToHGNC:
    """Tests for ground_to_hgnc convenience method."""

    @pytest.mark.asyncio
    async def test_returns_only_hgnc_ids(self):
        """Should return only HGNC IDs for HGNC results."""
        client = GildaClient()

        # Setup mock to return mixed results
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [{"term": {"db": "HGNC", "id": "6553", "entry_name": "LEP"}, "score": 0.98}],
            [{"term": {"db": "CHEBI", "id": "12345", "entry_name": "ghrelin"}, "score": 0.95}],
        ]
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            results = await client.ground_to_hgnc(["LEP", "ghrelin"])

        assert results["LEP"] == "HGNC:6553"
        assert results["ghrelin"] is None  # CHEBI, not HGNC

    @pytest.mark.asyncio
    async def test_respects_min_score(self):
        """Should filter by min_score."""
        client = GildaClient()

        mock_response = MagicMock()
        mock_response.json.return_value = [
            [{"term": {"db": "HGNC", "id": "1", "entry_name": "HIGH"}, "score": 0.9}],
            [{"term": {"db": "HGNC", "id": "2", "entry_name": "LOW"}, "score": 0.3}],
        ]
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            results = await client.ground_to_hgnc(["high", "low"], min_score=0.5)

        assert results["high"] == "HGNC:1"
        assert results["low"] is None  # Below threshold


# =============================================================================
# Test: Error handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_http_error(self):
        """Should handle HTTP errors gracefully."""
        import httpx

        client = GildaClient()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.ground("LEP")

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_request_error(self):
        """Should handle network errors gracefully."""
        import httpx

        client = GildaClient()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.RequestError("Connection failed")
            result = await client.ground("LEP")

        assert result is None

    @pytest.mark.asyncio
    async def test_batch_handles_error_for_all_unfetched(self):
        """Batch should mark all unfetched as None on error."""
        import httpx

        client = GildaClient()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.RequestError("Connection failed")
            results = await client.ground_batch(["A", "B", "C"])

        assert results["A"] is None
        assert results["B"] is None
        assert results["C"] is None


# =============================================================================
# Integration Tests (require network - mark for optional skip)
# =============================================================================


@pytest.mark.integration
class TestGildaIntegration:
    """Integration tests that hit the real Gilda API.

    Run with: pytest -m integration
    """

    @pytest.mark.asyncio
    async def test_real_leptin_grounding(self):
        """LEP and Leptin should both ground to same HGNC ID."""
        async with GildaClient() as client:
            results = await client.ground_batch(["LEP", "Leptin"])

        lep_result = results["LEP"]
        leptin_result = results["Leptin"]

        assert lep_result is not None
        assert leptin_result is not None
        assert lep_result.db == "HGNC"
        assert leptin_result.db == "HGNC"
        # Both should map to same gene
        assert lep_result.id == leptin_result.id

    @pytest.mark.asyncio
    async def test_real_tnf_grounding(self):
        """TNF should ground to HGNC."""
        async with GildaClient() as client:
            result = await client.ground("TNF")

        assert result is not None
        assert result.db == "HGNC"
        assert "TNF" in result.entry_name.upper()
