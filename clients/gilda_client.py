"""
Async client for INDRA/Gilda entity grounding API.

Gilda (Grounding Integrating Learned Disambiguations) provides entity
normalization to standard identifiers like HGNC, UniProt, CHEBI, etc.

API Documentation: https://github.com/gyorilab/gilda
Public endpoint: https://grounding.indra.bio
"""

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """Result from entity grounding.

    Attributes:
        db: Database source (e.g., "HGNC", "UP", "CHEBI", "MESH")
        id: Database identifier (e.g., "6553" for HGNC:6553)
        entry_name: Canonical entry name (e.g., "LEP" for leptin)
        score: Grounding confidence score (0-1)
        original_text: Original text that was grounded
    """
    db: str
    id: str
    entry_name: str
    score: float
    original_text: str

    @property
    def full_id(self) -> str:
        """Return full identifier in db:id format."""
        return f"{self.db}:{self.id}"

    def __repr__(self) -> str:
        return f"GroundingResult({self.db}:{self.id} '{self.entry_name}' score={self.score:.3f})"


class GildaClient:
    """Async client for INDRA/Gilda grounding API.

    Provides methods to normalize entity names to standard database identifiers,
    with preference for HGNC (gene-level) grounding for protein/gene entities.

    Example:
        async with GildaClient() as client:
            results = await client.ground_batch(["LEP", "Leptin", "TNF"])
            for text, result in results.items():
                if result:
                    print(f"{text} -> {result.full_id} ({result.entry_name})")
    """

    BASE_URL = "https://grounding.indra.bio"
    DEFAULT_TIMEOUT = 30.0

    # Preferred databases for gene/protein grounding (in priority order)
    PREFERRED_DBS = ("HGNC", "UP", "FPLX", "GO", "MESH", "CHEBI")

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        enable_cache: bool = True,
        prefer_hgnc: bool = True,
    ):
        """Initialize Gilda client.

        Args:
            timeout: Request timeout in seconds.
            enable_cache: Enable in-memory caching of grounding results.
            prefer_hgnc: Prefer HGNC results over other databases for gene/protein entities.
        """
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.prefer_hgnc = prefer_hgnc
        self._client = httpx.AsyncClient(timeout=timeout)
        self._cache: dict[str, GroundingResult | None] = {}

    async def __aenter__(self) -> "GildaClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def clear_cache(self) -> None:
        """Clear the grounding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def _normalize_text(self, text: str) -> str:
        """Normalize input text for consistent caching."""
        return text.strip()

    def _select_best_result(
        self,
        results: list[dict[str, Any]],
        original_text: str,
    ) -> GroundingResult | None:
        """Select the best grounding result based on preferences.

        Args:
            results: List of grounding results from API.
            original_text: Original text that was grounded.

        Returns:
            Best GroundingResult or None if no suitable result.
        """
        if not results:
            return None

        # Parse all results into structured format
        parsed_results: list[tuple[int, float, GroundingResult]] = []

        for result in results:
            term = result.get("term", {})
            db = term.get("db", "")
            id_ = term.get("id", "")
            entry_name = term.get("entry_name", "")
            score = result.get("score", 0.0)

            if not db or not id_:
                continue

            # Calculate priority based on preferred databases
            try:
                priority = self.PREFERRED_DBS.index(db)
            except ValueError:
                priority = len(self.PREFERRED_DBS)  # Lower priority for unknown DBs

            grounding = GroundingResult(
                db=db,
                id=id_,
                entry_name=entry_name,
                score=score,
                original_text=original_text,
            )

            parsed_results.append((priority, score, grounding))

        if not parsed_results:
            return None

        if self.prefer_hgnc:
            # First try to find HGNC result with reasonable score
            hgnc_results = [
                (p, s, g) for p, s, g in parsed_results
                if g.db == "HGNC" and s >= 0.5
            ]
            if hgnc_results:
                # Return highest scoring HGNC result
                hgnc_results.sort(key=lambda x: x[1], reverse=True)
                return hgnc_results[0][2]

        # Fall back to best result by priority then score
        parsed_results.sort(key=lambda x: (x[0], -x[1]))  # Lower priority number is better, higher score is better
        return parsed_results[0][2]

    async def ground(self, text: str) -> GroundingResult | None:
        """Ground a single entity text.

        Args:
            text: Entity text to ground (e.g., "Leptin", "LEP", "TNF-alpha").

        Returns:
            GroundingResult if grounding succeeded, None otherwise.
        """
        normalized = self._normalize_text(text)

        # Check cache
        if self.enable_cache and normalized in self._cache:
            return self._cache[normalized]

        try:
            response = await self._client.post(
                f"{self.BASE_URL}/ground",
                json={"text": normalized},
            )
            response.raise_for_status()
            results = response.json()

            result = self._select_best_result(results, normalized)

            if self.enable_cache:
                self._cache[normalized] = result

            return result

        except httpx.HTTPStatusError as e:
            print(f"Gilda API HTTP error: {e.response.status_code} - {e.response.text[:200]}")
            return None
        except httpx.RequestError as e:
            print(f"Gilda API request failed: {e}")
            return None
        except Exception as e:
            print(f"Gilda API unexpected error: {e}")
            return None

    async def ground_batch(
        self,
        texts: list[str],
    ) -> dict[str, GroundingResult | None]:
        """Ground multiple entity texts in a single batch request.

        Uses the /ground_multi endpoint for efficiency.

        Args:
            texts: List of entity texts to ground.

        Returns:
            Dictionary mapping input text to GroundingResult (or None if failed).
        """
        if not texts:
            return {}

        # Normalize and dedupe inputs while preserving order
        normalized_texts = [self._normalize_text(t) for t in texts]
        unique_texts = list(dict.fromkeys(normalized_texts))  # Preserve order, remove dupes

        # Check cache for already-grounded texts
        results: dict[str, GroundingResult | None] = {}
        texts_to_fetch: list[str] = []

        for text in unique_texts:
            if self.enable_cache and text in self._cache:
                results[text] = self._cache[text]
            else:
                texts_to_fetch.append(text)

        if texts_to_fetch:
            try:
                # Build request payload
                payload = [{"text": text} for text in texts_to_fetch]

                response = await self._client.post(
                    f"{self.BASE_URL}/ground_multi",
                    json=payload,
                )
                response.raise_for_status()
                batch_results = response.json()

                # Process results - API returns list of lists in same order as input
                for text, text_results in zip(texts_to_fetch, batch_results):
                    result = self._select_best_result(text_results, text)
                    results[text] = result

                    if self.enable_cache:
                        self._cache[text] = result

            except httpx.HTTPStatusError as e:
                print(f"Gilda API HTTP error: {e.response.status_code} - {e.response.text[:200]}")
                # Mark all unfetched as None
                for text in texts_to_fetch:
                    results[text] = None
            except httpx.RequestError as e:
                print(f"Gilda API request failed: {e}")
                for text in texts_to_fetch:
                    results[text] = None
            except Exception as e:
                print(f"Gilda API unexpected error: {e}")
                for text in texts_to_fetch:
                    results[text] = None

        # Return results in original input order
        return {text: results.get(self._normalize_text(text)) for text in texts}

    async def ground_to_hgnc(
        self,
        texts: list[str],
        min_score: float = 0.5,
    ) -> dict[str, str | None]:
        """Ground texts and return only HGNC IDs.

        Convenience method for getting normalized gene identifiers.

        Args:
            texts: List of entity texts to ground.
            min_score: Minimum score threshold for accepting result.

        Returns:
            Dictionary mapping input text to HGNC ID (e.g., "HGNC:6553") or None.
        """
        results = await self.ground_batch(texts)

        hgnc_map: dict[str, str | None] = {}
        for text, result in results.items():
            if result and result.db == "HGNC" and result.score >= min_score:
                hgnc_map[text] = result.full_id
            else:
                hgnc_map[text] = None

        return hgnc_map
