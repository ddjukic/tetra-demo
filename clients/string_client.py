"""
Async client for STRING protein-protein interaction database API.

STRING (Search Tool for the Retrieval of Interacting Genes/Proteins) provides
known and predicted protein-protein interactions.

API Documentation: https://string-db.org/help/api/
"""

from typing import Any

import httpx


class StringClient:
    """Async client for STRING database API.

    Provides methods to query protein interaction networks, interaction partners,
    and functional annotations from the STRING database.

    Example:
        async with StringClient() as client:
            interactions = await client.get_network(["BRCA1", "TP53"])
            print(f"Found {len(interactions)} interactions")
    """

    BASE_URL = "https://string-db.org/api"
    SPECIES_HUMAN = 9606
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        caller_identity: str = "tetra-kg-agent",
        species: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize STRING client.

        Args:
            caller_identity: Application identifier for API calls (required by STRING).
            species: NCBI taxonomy ID. Defaults to 9606 (human).
            timeout: Request timeout in seconds.
        """
        self.caller_identity = caller_identity
        self.species = species or self.SPECIES_HUMAN
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> "StringClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _format_identifiers(self, proteins: list[str]) -> str:
        """Format protein list for STRING API.

        STRING expects identifiers separated by %0d (URL-encoded carriage return).
        """
        return "%0d".join(proteins)

    def _base_params(self) -> dict[str, Any]:
        """Get base parameters for all API calls."""
        return {
            "species": self.species,
            "caller_identity": self.caller_identity,
        }

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any],
        method: str = "GET",
    ) -> list[dict[str, Any]]:
        """Make API request and return parsed JSON.

        Args:
            endpoint: API endpoint path (e.g., "/json/network").
            params: Query parameters.
            method: HTTP method.

        Returns:
            Parsed JSON response as list of dicts.

        Raises:
            httpx.HTTPStatusError: On HTTP error responses.
        """
        url = f"{self.BASE_URL}{endpoint}"

        try:
            if method == "GET":
                response = await self._client.get(url, params=params)
            else:
                response = await self._client.post(url, data=params)

            response.raise_for_status()

            result = response.json()
            # STRING API returns list for success, might return dict for errors
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                # Single result or error
                if "Error" in result or "error" in result:
                    return []
                return [result]
            return []

        except httpx.HTTPStatusError as e:
            # Log error but don't crash
            print(f"STRING API error: {e.response.status_code} - {e.response.text[:200]}")
            return []
        except httpx.RequestError as e:
            print(f"STRING API request failed: {e}")
            return []
        except Exception as e:
            print(f"STRING API unexpected error: {e}")
            return []

    async def get_network(
        self,
        proteins: list[str],
        min_score: int = 400,
        network_type: str = "functional",
        add_nodes: int = 0,
    ) -> list[dict[str, Any]]:
        """Get interaction network for proteins.

        Retrieves STRING interaction data between the specified proteins.

        Args:
            proteins: List of protein names/identifiers (e.g., ["BRCA1", "TP53"]).
            min_score: Minimum interaction score (0-1000). Default 400 (medium).
            network_type: "functional" (all) or "physical" (physical only).
            add_nodes: Number of additional interacting partners to add. Default 0.

        Returns:
            List of interaction dicts with keys:
                - stringId_A: STRING identifier for protein A
                - stringId_B: STRING identifier for protein B
                - preferredName_A: Common name for protein A
                - preferredName_B: Common name for protein B
                - score: Combined interaction score (0-1)
                - nscore: Neighborhood score
                - fscore: Fusion score
                - pscore: Phylogenetic profile score
                - ascore: Coexpression score
                - escore: Experimental score
                - dscore: Database score
                - tscore: Text mining score
        """
        if not proteins:
            return []

        params = {
            **self._base_params(),
            "identifiers": self._format_identifiers(proteins),
            "required_score": min_score,
            "network_type": network_type,
        }

        if add_nodes > 0:
            params["add_nodes"] = add_nodes

        return await self._request("/json/network", params)

    async def get_interaction_partners(
        self,
        proteins: list[str],
        limit: int = 50,
        min_score: int = 400,
    ) -> list[dict[str, Any]]:
        """Get interaction partners for proteins.

        Returns all STRING interaction partners for the specified proteins,
        including proteins not in the original query.

        Args:
            proteins: List of protein names/identifiers.
            limit: Maximum partners to return per protein. Default 50.
            min_score: Minimum interaction score (0-1000). Default 400.

        Returns:
            List of interaction dicts (same format as get_network).
        """
        if not proteins:
            return []

        params = {
            **self._base_params(),
            "identifiers": self._format_identifiers(proteins),
            "limit": limit,
            "required_score": min_score,
        }

        return await self._request("/json/interaction_partners", params)

    async def get_functional_annotation(
        self,
        proteins: list[str],
        allow_pubmed: bool = False,
    ) -> list[dict[str, Any]]:
        """Get functional annotations for proteins.

        Retrieves Gene Ontology, UniProt Keywords, Pfam domains, InterPro,
        and SMART annotations for the specified proteins.

        Args:
            proteins: List of protein names/identifiers.
            allow_pubmed: Include PubMed references. Default False.

        Returns:
            List of annotation dicts with keys:
                - category: Annotation category (GO Process, GO Function, etc.)
                - term: Annotation term ID
                - description: Human-readable description
                - number_of_genes: Count of proteins with this annotation
                - number_of_genes_in_background: Total proteins with annotation
                - preferredNames: Proteins that have this annotation
        """
        if not proteins:
            return []

        params = {
            **self._base_params(),
            "identifiers": self._format_identifiers(proteins),
            "allow_pubmed": 1 if allow_pubmed else 0,
        }

        return await self._request("/json/functional_annotation", params)

    async def get_enrichment(
        self,
        proteins: list[str],
    ) -> list[dict[str, Any]]:
        """Get functional enrichment analysis for proteins.

        Performs enrichment analysis for GO terms, KEGG pathways,
        and other functional categories.

        Args:
            proteins: List of protein names/identifiers.

        Returns:
            List of enrichment dicts with keys:
                - category: Enrichment category
                - term: Term ID
                - description: Term description
                - number_of_genes: Count in query
                - number_of_genes_in_background: Total with term
                - p_value: Enrichment p-value
                - fdr: False discovery rate
                - preferredNames: Proteins contributing to enrichment
        """
        if not proteins:
            return []

        params = {
            **self._base_params(),
            "identifiers": self._format_identifiers(proteins),
        }

        return await self._request("/json/enrichment", params)

    async def resolve_identifiers(
        self,
        proteins: list[str],
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """Resolve protein names to STRING identifiers.

        Useful for validating protein names and getting canonical IDs.

        Args:
            proteins: List of protein names/identifiers.
            limit: Maximum matches per identifier. Default 1.

        Returns:
            List of mapping dicts with keys:
                - queryItem: Original query term
                - queryIndex: Index in query list
                - stringId: STRING identifier
                - ncbiTaxonId: NCBI taxonomy ID
                - taxonName: Species name
                - preferredName: Canonical protein name
                - annotation: Protein description
        """
        if not proteins:
            return []

        params = {
            **self._base_params(),
            "identifiers": self._format_identifiers(proteins),
            "limit": limit,
        }

        return await self._request("/json/get_string_ids", params)
