"""
Data Fetch Agent - LLM-driven data collection for the KG pipeline.

This agent uses LLM reasoning to:
1. Fetch protein interactions from STRING database
2. Construct optimal PubMed queries based on STRING results
3. Search PubMed and collect NER annotations

The agent has minimal tools and focuses on intelligent query construction.
The actual relationship extraction and graph building happens in the
deterministic KGPipeline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from clients.string_client import StringClient
from clients.pubmed_client import PubMedClient
from pipeline.models import PipelineInput

# Load environment variables
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

logger = logging.getLogger(__name__)


class DataFetchAgent:
    """
    Sequential agent for LLM-driven data collection.

    Uses LLM reasoning to:
    - Identify seed proteins from user query
    - Construct optimal PubMed search queries
    - Decide when enough data has been collected

    Tools:
    - get_string_network: Fetch protein interactions from STRING
    - search_literature: Search PubMed and get NER annotations

    Example:
        >>> agent = DataFetchAgent()
        >>> input_data = await agent.fetch("Build a KG for the orexin signaling pathway")
        >>> print(f"Collected {input_data.article_count} articles")
    """

    def __init__(
        self,
        string_client: StringClient | None = None,
        pubmed_client: PubMedClient | None = None,
        model: str = "gemini-2.5-flash",
    ):
        """
        Initialize the data fetch agent.

        Args:
            string_client: STRING database client. Created if not provided.
            pubmed_client: PubMed client. Created if not provided.
            model: LLM model to use for reasoning.
        """
        self._string_client = string_client or StringClient()
        self._pubmed_client = pubmed_client or PubMedClient(
            api_key=os.getenv("NCBI_API_KEY")
        )
        self._model = model

        # State for current fetch operation
        self._current_input: PipelineInput | None = None

    async def fetch(
        self,
        user_query: str,
        max_articles: int = 50,
    ) -> PipelineInput:
        """
        Fetch data for building a knowledge graph.

        Uses LLM reasoning to:
        1. Extract seed proteins from the user query
        2. Fetch STRING network for those proteins
        3. Construct an optimal PubMed query
        4. Fetch articles and annotations

        Args:
            user_query: User's natural language query about what to build.
            max_articles: Maximum number of articles to fetch.

        Returns:
            PipelineInput with all collected data.
        """
        logger.info(f"DataFetchAgent starting: {user_query}")

        # Initialize fresh input
        self._current_input = PipelineInput(
            metadata={"user_query": user_query}
        )

        # Step 1: Extract seed proteins from the query
        seed_proteins = await self._extract_seed_proteins(user_query)
        logger.info(f"Extracted seed proteins: {seed_proteins}")

        if seed_proteins:
            # Step 2: Get STRING network
            await self._fetch_string_network(seed_proteins)

        # Step 3: Construct PubMed query
        pubmed_query = await self._construct_pubmed_query(user_query)
        logger.info(f"Constructed PubMed query: {pubmed_query}")

        # Step 4: Search literature and get annotations
        await self._fetch_literature(pubmed_query, max_articles)

        logger.info(
            f"DataFetchAgent complete: {self._current_input.article_count} articles, "
            f"{self._current_input.interaction_count} STRING interactions"
        )

        return self._current_input

    async def _extract_seed_proteins(self, user_query: str) -> list[str]:
        """
        Extract seed proteins from the user query using LLM.

        This uses simple heuristics for common pathways.
        In a full implementation, this would use the LLM.
        """
        query_lower = user_query.lower()

        # Common pathway to protein mappings
        pathway_proteins = {
            "orexin": ["HCRTR1", "HCRTR2", "HCRT", "OX1R", "OX2R"],
            "hypocretin": ["HCRTR1", "HCRTR2", "HCRT"],
            "insulin": ["INS", "INSR", "IRS1", "IRS2", "AKT1"],
            "egfr": ["EGFR", "ERBB2", "GRB2", "SOS1", "KRAS"],
            "p53": ["TP53", "MDM2", "CDKN1A", "BAX", "BCL2"],
            "brca": ["BRCA1", "BRCA2", "RAD51", "ATM", "CHEK2"],
            "mapk": ["MAPK1", "MAPK3", "RAF1", "MEK1", "ERK"],
            "wnt": ["WNT1", "CTNNB1", "APC", "GSK3B", "TCF7"],
            "notch": ["NOTCH1", "DLL1", "JAG1", "HES1", "RBPJ"],
            "hedgehog": ["SHH", "PTCH1", "SMO", "GLI1", "GLI2"],
            "tgf": ["TGFB1", "SMAD2", "SMAD3", "SMAD4", "ACVR1"],
        }

        # Check for pathway keywords
        for pathway, proteins in pathway_proteins.items():
            if pathway in query_lower:
                return proteins[:5]  # Return top 5 proteins

        # Check for explicit protein mentions
        # Look for uppercase words that might be gene symbols
        import re
        potential_proteins = re.findall(r'\b[A-Z][A-Z0-9]{1,10}\b', user_query)
        if potential_proteins:
            return potential_proteins[:5]

        return []

    async def _construct_pubmed_query(self, user_query: str) -> str:
        """
        Construct an optimal PubMed query.

        Uses information from STRING network and user query.
        In a full implementation, this would use the LLM for optimization.
        """
        query_lower = user_query.lower()

        # Extract key terms from user query
        terms = []

        # Add pathway/topic terms
        topic_keywords = [
            "signaling", "pathway", "mechanism", "interaction",
            "regulation", "expression", "function", "role"
        ]
        for keyword in topic_keywords:
            if keyword in query_lower:
                terms.append(keyword)

        # Add disease terms if mentioned
        disease_terms = [
            "cancer", "diabetes", "alzheimer", "parkinson",
            "obesity", "inflammation", "infection"
        ]
        for disease in disease_terms:
            if disease in query_lower:
                terms.append(disease)

        # Extract main topic
        # Simple approach: find the first noun phrase
        import re
        # Find "for the X" or "about X" patterns
        topic_match = re.search(r'(?:for|about|of)\s+(?:the\s+)?(\w+(?:\s+\w+)?)', query_lower)
        if topic_match:
            topic = topic_match.group(1).strip()
            if topic not in ['the', 'a', 'an']:
                terms.insert(0, topic)

        # Add proteins from STRING network
        if self._current_input and self._current_input.seed_proteins:
            # Add seed proteins to query
            protein_query = " OR ".join(
                f'"{p}"[Gene Name]' for p in self._current_input.seed_proteins[:3]
            )
            if protein_query:
                terms.append(f"({protein_query})")

        # Build base query from terms or use user query
        if not terms:
            base_query = user_query
        else:
            # For PubMed, simpler queries work better
            # Just use the topic term, not the gene filters (those are too restrictive)
            topic_terms = [t for t in terms if not t.startswith("(")]
            if topic_terms:
                base_query = " AND ".join(topic_terms)
            else:
                base_query = terms[0] if terms else user_query

        # Add filters for homo sapiens and recent articles (2020+)
        # Use MeSH Terms for species (per PubMed docs) and [pdat] for publication date
        query = f'({base_query}) AND humans[MeSH Terms] AND 2020:2025[pdat]'
        self._current_input.pubmed_query = query
        return query

    async def _fetch_string_network(
        self,
        seed_proteins: list[str],
        min_score: int = 700,
    ) -> None:
        """
        Fetch protein interactions from STRING.

        Args:
            seed_proteins: List of protein/gene names.
            min_score: Minimum interaction confidence score.
        """
        try:
            interactions = await self._string_client.get_network(
                proteins=seed_proteins,
                min_score=min_score,
                network_type="physical",
            )

            self._current_input.string_interactions = interactions
            self._current_input.seed_proteins = seed_proteins

            # Log proteins found
            proteins = set()
            for interaction in interactions:
                proteins.add(interaction.get("preferredName_A", ""))
                proteins.add(interaction.get("preferredName_B", ""))
            proteins.discard("")

            logger.info(
                f"STRING network: {len(interactions)} interactions, "
                f"{len(proteins)} unique proteins"
            )

        except Exception as e:
            logger.error(f"STRING fetch error: {e}")
            self._current_input.metadata["string_error"] = str(e)

    async def _fetch_literature(
        self,
        query: str,
        max_results: int = 50,
    ) -> None:
        """
        Search PubMed and fetch annotations.

        Args:
            query: PubMed search query.
            max_results: Maximum articles to fetch.
        """
        try:
            # Search PubMed
            pmids = await self._pubmed_client.search(
                query=query,
                max_results=max_results,
            )

            if not pmids:
                logger.warning(f"No PubMed results for query: {query}")
                return

            # Fetch article abstracts
            articles = await self._pubmed_client.fetch_abstracts(pmids)
            self._current_input.articles = articles

            logger.info(f"Fetched {len(articles)} articles from PubMed")

            # Get NER annotations
            annotations = await self._pubmed_client.get_pubtator_annotations(
                pmids=pmids,
                entity_types=["Gene", "Disease", "Chemical"],
            )

            # Group by PMID
            annotations_by_pmid: dict[str, list[dict[str, Any]]] = {}
            for ann in annotations:
                pmid = ann.get("pmid", "")
                if pmid not in annotations_by_pmid:
                    annotations_by_pmid[pmid] = []
                annotations_by_pmid[pmid].append(ann)

            self._current_input.annotations = annotations_by_pmid

            logger.info(
                f"Got {len(annotations)} annotations for {len(annotations_by_pmid)} articles"
            )

        except Exception as e:
            logger.error(f"Literature fetch error: {e}")
            self._current_input.metadata["literature_error"] = str(e)

    async def expand(
        self,
        expansion_query: str,
        max_articles: int = 30,
    ) -> PipelineInput:
        """
        Fetch additional data for graph expansion.

        Similar to fetch() but returns data to be merged with existing graph.

        Args:
            expansion_query: Query for additional data.
            max_articles: Maximum additional articles.

        Returns:
            PipelineInput with new data for merging.
        """
        # Create a new agent instance for the expansion
        # This keeps the state separate
        return await self.fetch(expansion_query, max_articles)


async def create_data_fetch_agent(
    string_client: StringClient | None = None,
    pubmed_client: PubMedClient | None = None,
) -> DataFetchAgent:
    """
    Factory function to create a DataFetchAgent.

    Args:
        string_client: Optional STRING client.
        pubmed_client: Optional PubMed client.

    Returns:
        Configured DataFetchAgent instance.
    """
    return DataFetchAgent(
        string_client=string_client,
        pubmed_client=pubmed_client,
    )
