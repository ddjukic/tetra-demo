"""
Async client for PubMed E-utilities and PubTator3 APIs.

Provides access to:
- PubMed search and article retrieval via NCBI E-utilities
- Named entity recognition annotations via PubTator3

API Documentation:
- E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25499/
- PubTator3: https://www.ncbi.nlm.nih.gov/research/pubtator3/
"""

import xml.etree.ElementTree as ET
from typing import Any

import httpx


class PubMedClient:
    """Async client for PubMed E-utilities and PubTator3 APIs.

    Provides methods to search PubMed, fetch article abstracts,
    and retrieve named entity annotations from PubTator3.

    Example:
        async with PubMedClient() as client:
            pmids = await client.search("BRCA1 breast cancer", max_results=10)
            articles = await client.fetch_abstracts(pmids)
            annotations = await client.get_pubtator_annotations(pmids)
    """

    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    PUBTATOR_BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
    DEFAULT_TIMEOUT = 30.0

    # Entity types supported by PubTator3
    ENTITY_TYPES = {"Gene", "Disease", "Chemical", "Species", "Mutation", "CellLine"}

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize PubMed client.

        Args:
            api_key: NCBI API key for higher rate limits (optional).
                     Without key: 3 requests/second
                     With key: 10 requests/second
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> "PubMedClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _eutils_params(self) -> dict[str, str]:
        """Get base parameters for E-utilities calls."""
        params: dict[str, str] = {}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    async def search(
        self,
        query: str,
        max_results: int = 50,
        sort: str = "relevance",
    ) -> list[str]:
        """Search PubMed and return PMIDs.

        Args:
            query: PubMed search query (supports Boolean operators).
            max_results: Maximum results to return. Default 50.
            sort: Sort order - "relevance" or "date". Default "relevance".

        Returns:
            List of PMID strings.
        """
        if not query:
            return []

        url = f"{self.EUTILS_BASE}/esearch.fcgi"
        # Use POST for long queries to avoid 414 URI Too Long errors
        data_payload = {
            **self._eutils_params(),
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": sort,
        }

        try:
            # POST allows longer queries than GET
            response = await self._client.post(url, data=data_payload)
            response.raise_for_status()
            data = response.json()

            # Extract PMIDs from response
            result = data.get("esearchresult", {})
            pmids = result.get("idlist", [])
            return pmids

        except httpx.HTTPStatusError as e:
            print(f"PubMed search error: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            print(f"PubMed search request failed: {e}")
            return []
        except Exception as e:
            print(f"PubMed search unexpected error: {e}")
            return []

    async def fetch_abstracts(
        self,
        pmids: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch article details for PMIDs.

        Args:
            pmids: List of PubMed IDs.

        Returns:
            List of article dicts with keys:
                - pmid: PubMed ID
                - title: Article title
                - abstract: Article abstract
                - year: Publication year
                - journal: Journal name
                - authors: List of author names
        """
        if not pmids:
            return []

        url = f"{self.EUTILS_BASE}/efetch.fcgi"
        # Use POST to avoid 414 URI Too Long errors with many PMIDs
        data_payload = {
            **self._eutils_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }

        try:
            # POST allows longer PMID lists than GET
            response = await self._client.post(url, data=data_payload)
            response.raise_for_status()

            # Parse XML response
            articles = self._parse_efetch_xml(response.text)
            return articles

        except httpx.HTTPStatusError as e:
            print(f"PubMed fetch error: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            print(f"PubMed fetch request failed: {e}")
            return []
        except Exception as e:
            print(f"PubMed fetch unexpected error: {e}")
            return []

    def _parse_efetch_xml(self, xml_text: str) -> list[dict[str, Any]]:
        """Parse efetch XML response into article dicts.

        Args:
            xml_text: XML response from efetch.

        Returns:
            List of parsed article dicts.
        """
        articles = []

        try:
            root = ET.fromstring(xml_text)

            for article_elem in root.findall(".//PubmedArticle"):
                article = self._parse_article_element(article_elem)
                if article:
                    articles.append(article)

        except ET.ParseError as e:
            print(f"XML parse error: {e}")

        return articles

    def _parse_article_element(self, elem: ET.Element) -> dict[str, Any] | None:
        """Parse a single PubmedArticle XML element.

        Args:
            elem: PubmedArticle XML element.

        Returns:
            Article dict or None if parsing fails.
        """
        try:
            # Extract PMID
            pmid_elem = elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            if not pmid:
                return None

            # Extract title
            title_elem = elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""

            # Extract abstract - may have multiple AbstractText elements
            abstract_parts = []
            for abstract_text in elem.findall(".//AbstractText"):
                label = abstract_text.get("Label", "")
                text = abstract_text.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            # Extract publication year
            year = None
            pub_date = elem.find(".//PubDate")
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                if year_elem is not None and year_elem.text:
                    year = year_elem.text
                else:
                    # Try MedlineDate format
                    medline_date = pub_date.find("MedlineDate")
                    if medline_date is not None and medline_date.text:
                        # Extract year from "YYYY Mon-Mon" format
                        year = medline_date.text[:4]

            # Extract journal name
            journal_elem = elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""

            # Extract authors
            authors = []
            for author in elem.findall(".//Author"):
                last_name = author.find("LastName")
                fore_name = author.find("ForeName")
                if last_name is not None and last_name.text:
                    name = last_name.text
                    if fore_name is not None and fore_name.text:
                        name = f"{fore_name.text} {name}"
                    authors.append(name)

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "authors": authors,
            }

        except Exception as e:
            print(f"Error parsing article: {e}")
            return None

    async def get_pubtator_annotations(
        self,
        pmids: list[str],
        entity_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get PubTator3 NER annotations for articles.

        Retrieves named entity recognition annotations including
        genes, diseases, chemicals, and other biomedical entities.

        Args:
            pmids: List of PubMed IDs (max 100 per request).
            entity_types: Entity types to filter. Default ["Gene", "Disease", "Chemical"].
                         Options: Gene, Disease, Chemical, Species, Mutation, CellLine.

        Returns:
            List of annotation dicts with keys:
                - pmid: PubMed ID
                - entity_id: Normalized entity identifier
                - entity_text: Entity text as mentioned in article
                - entity_type: Entity type (Gene, Disease, etc.)
        """
        if not pmids:
            return []

        # Default entity types if not specified
        if entity_types is None:
            entity_types = ["Gene", "Disease", "Chemical"]

        # Validate entity types
        valid_types = set(entity_types) & self.ENTITY_TYPES
        if not valid_types:
            print(f"No valid entity types specified. Valid types: {self.ENTITY_TYPES}")
            return []

        # PubTator limits to 100 PMIDs per request
        all_annotations = []
        batch_size = 100

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            annotations = await self._fetch_pubtator_batch(batch, valid_types)
            all_annotations.extend(annotations)

        return all_annotations

    async def _fetch_pubtator_batch(
        self,
        pmids: list[str],
        entity_types: set[str],
    ) -> list[dict[str, Any]]:
        """Fetch PubTator annotations for a batch of PMIDs.

        Args:
            pmids: Batch of PubMed IDs (max 100).
            entity_types: Set of entity types to include.

        Returns:
            List of annotation dicts.
        """
        url = f"{self.PUBTATOR_BASE}/publications/export/biocjson"
        params = {"pmids": ",".join(pmids)}

        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()

            # Parse BioCJSON response
            annotations = self._parse_biocjson(response.text, entity_types)
            return annotations

        except httpx.HTTPStatusError as e:
            print(f"PubTator error: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            print(f"PubTator request failed: {e}")
            return []
        except Exception as e:
            print(f"PubTator unexpected error: {e}")
            return []

    def _parse_biocjson(
        self,
        text: str,
        entity_types: set[str],
    ) -> list[dict[str, Any]]:
        """Parse PubTator BioCJSON response.

        PubTator3 API returns JSON in format: {"PubTator3": [array of documents]}
        Each document contains passages with annotations.

        Args:
            text: BioCJSON response text.
            entity_types: Entity types to include.

        Returns:
            List of annotation dicts.
        """
        import json

        annotations = []

        try:
            data = json.loads(text)

            # PubTator3 wraps documents in {"PubTator3": [...]}
            documents = data.get("PubTator3", [])

            # If not wrapped, treat as single document or list
            if not documents and isinstance(data, list):
                documents = data
            elif not documents and isinstance(data, dict) and "passages" in data:
                documents = [data]

            for doc in documents:
                pmid = doc.get("id") or doc.get("pmid", "")

                # Process passages (title, abstract)
                for passage in doc.get("passages", []):
                    for annot in passage.get("annotations", []):
                        infons = annot.get("infons", {})
                        entity_type = infons.get("type", "")

                        # Filter by entity type
                        if entity_type not in entity_types:
                            continue

                        # Extract entity information
                        # identifier can be in different fields
                        entity_id = infons.get("identifier") or infons.get("normalized_id", "")
                        if isinstance(entity_id, list):
                            entity_id = str(entity_id[0]) if entity_id else ""
                        else:
                            entity_id = str(entity_id) if entity_id else ""

                        entity_text = annot.get("text", "")

                        if entity_text:
                            annotations.append(
                                {
                                    "pmid": str(pmid),
                                    "entity_id": entity_id,
                                    "entity_text": entity_text,
                                    "entity_type": entity_type,
                                }
                            )

        except json.JSONDecodeError as e:
            print(f"JSON parse error in PubTator response: {e}")

        return annotations

    async def search_by_gene(
        self,
        gene_symbol: str,
        additional_terms: str = "",
        max_results: int = 50,
    ) -> list[str]:
        """Convenience method to search for articles about a gene.

        Args:
            gene_symbol: Gene symbol (e.g., "BRCA1").
            additional_terms: Additional search terms to include.
            max_results: Maximum results to return.

        Returns:
            List of PMIDs.
        """
        query = f"{gene_symbol}[Gene Name]"
        if additional_terms:
            query = f"({query}) AND ({additional_terms})"

        return await self.search(query, max_results=max_results)

    async def search_by_disease(
        self,
        disease: str,
        additional_terms: str = "",
        max_results: int = 50,
    ) -> list[str]:
        """Convenience method to search for articles about a disease.

        Args:
            disease: Disease name (e.g., "breast cancer").
            additional_terms: Additional search terms to include.
            max_results: Maximum results to return.

        Returns:
            List of PMIDs.
        """
        query = f"{disease}[MeSH Terms]"
        if additional_terms:
            query = f"({query}) AND ({additional_terms})"

        return await self.search(query, max_results=max_results)
