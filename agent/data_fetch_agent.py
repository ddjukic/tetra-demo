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
from pipeline.config import PipelineConfig
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
        config: PipelineConfig | None = None,
        model: str = "gemini-2.5-flash",
    ):
        """
        Initialize the data fetch agent.

        Args:
            string_client: STRING database client. Created if not provided.
            pubmed_client: PubMed client. Created if not provided.
            config: Pipeline configuration. Uses defaults if not provided.
            model: LLM model to use for reasoning.
        """
        self._string_client = string_client or StringClient()
        self._pubmed_client = pubmed_client or PubMedClient(
            api_key=os.getenv("NCBI_API_KEY")
        )
        self._config = config or PipelineConfig()
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

        Uses expanded proteins from STRING network plus topic terms.
        Strategy: (topic_terms) OR (gene1 OR gene2 OR ...) to find articles
        that mention ANY of our proteins of interest.
        """
        import re
        query_lower = user_query.lower()

        # Extract topic term (e.g., "orexin" from "orexin signaling pathway")
        topic_term = None
        topic_match = re.search(r'(?:for|about|of)\s+(?:the\s+)?(\w+)', query_lower)
        if topic_match:
            topic = topic_match.group(1).strip()
            if topic not in ['the', 'a', 'an', 'build', 'create', 'make']:
                topic_term = topic

        # Build gene query from expanded STRING proteins
        # Prioritize original seeds, then add discovered proteins
        # Keep query short to avoid 414 URI Too Long errors
        gene_query = None
        max_genes_in_query = 8  # PubMed has URL length limits

        if self._current_input and self._current_input.seed_proteins:
            # Get original seeds from metadata (if available)
            original_seeds = set()
            if "string_extension" in self._current_input.metadata:
                original_seeds = set(
                    self._current_input.metadata["string_extension"].get("original_seeds", [])
                )

            # Prioritize: original seeds first, then discovered proteins
            all_proteins = self._current_input.seed_proteins
            if original_seeds:
                # Put original seeds first
                priority_proteins = [p for p in all_proteins if p in original_seeds]
                other_proteins = [p for p in all_proteins if p not in original_seeds]
                proteins = priority_proteins + other_proteins
            else:
                proteins = all_proteins

            proteins = proteins[:max_genes_in_query]
            if proteins:
                # Use [tiab] (title/abstract) for gene names - more flexible than [Gene Name]
                gene_terms = [f'"{p}"[tiab]' for p in proteins]
                gene_query = " OR ".join(gene_terms)
                logger.info(
                    f"PubMed query using {len(proteins)} proteins: {proteins}"
                )

        # Construct query: (topic OR genes) to maximize article retrieval
        # Articles matching EITHER the topic OR mentioning any of our proteins
        query_parts = []
        if topic_term:
            query_parts.append(f'"{topic_term}"[tiab]')
        if gene_query:
            query_parts.append(f"({gene_query})")

        if query_parts:
            base_query = " OR ".join(query_parts)
        else:
            # Fallback to user query if we couldn't extract anything
            base_query = user_query

        # Add filters for homo sapiens and recent articles (2020+)
        query = f'({base_query}) AND humans[MeSH Terms] AND 2020:2025[pdat]'
        self._current_input.pubmed_query = query
        return query

    async def _fetch_string_network(
        self,
        seed_proteins: list[str],
        min_score: int | None = None,
        extend_network: int | None = None,
    ) -> None:
        """
        Fetch protein interactions from STRING with optional network extension.

        This method implements a "wider network" strategy:
        1. First, get interaction partners to discover related proteins
        2. Then, get the extended network with add_nodes to fill connections
        3. Combine all unique interactions

        This creates a richer network with more diversity, including G-proteins,
        arrestins, and other related signaling proteins.

        Args:
            seed_proteins: List of protein/gene names.
            min_score: Minimum interaction confidence score. Defaults to config.
            extend_network: Number of additional proteins to add. Defaults to config.
        """
        # Use config defaults if not specified
        min_score = min_score or self._config.string_min_score
        extend_network = (
            extend_network if extend_network is not None
            else self._config.string_extend_network
        )

        try:
            all_interactions: list[dict[str, Any]] = []
            all_proteins: set[str] = set(seed_proteins)

            # Step 1: Get interaction partners if extension is enabled
            if extend_network > 0:
                logger.info(
                    f"Fetching interaction partners for {len(seed_proteins)} seeds "
                    f"(extend_network={extend_network})"
                )
                partners = await self._string_client.get_interaction_partners(
                    proteins=seed_proteins,
                    limit=extend_network,  # Partners per seed protein
                    min_score=min_score,
                )
                all_interactions.extend(partners)

                # Extract partner proteins
                for interaction in partners:
                    all_proteins.add(interaction.get("preferredName_A", ""))
                    all_proteins.add(interaction.get("preferredName_B", ""))
                all_proteins.discard("")

                logger.info(
                    f"Partners: {len(partners)} interactions, "
                    f"{len(all_proteins) - len(seed_proteins)} new proteins discovered"
                )

            # Step 2: Get extended network with add_nodes
            logger.info(
                f"Fetching extended network (add_nodes={extend_network})"
            )
            extended = await self._string_client.get_network(
                proteins=seed_proteins,
                min_score=min_score,
                network_type="physical",
                add_nodes=extend_network,
            )

            # Add new interactions (avoid duplicates by protein pair)
            existing_pairs: set[tuple[str, str]] = set()
            for interaction in all_interactions:
                pair = (
                    interaction.get("preferredName_A", ""),
                    interaction.get("preferredName_B", "")
                )
                existing_pairs.add(pair)
                existing_pairs.add((pair[1], pair[0]))  # Add reverse pair

            for interaction in extended:
                pair = (
                    interaction.get("preferredName_A", ""),
                    interaction.get("preferredName_B", "")
                )
                if pair not in existing_pairs:
                    all_interactions.append(interaction)
                    existing_pairs.add(pair)
                    existing_pairs.add((pair[1], pair[0]))

                # Extract proteins from extended network
                all_proteins.add(interaction.get("preferredName_A", ""))
                all_proteins.add(interaction.get("preferredName_B", ""))
            all_proteins.discard("")

            # Store results
            self._current_input.string_interactions = all_interactions
            # Store expanded protein list so PubMed query can use them
            self._current_input.seed_proteins = sorted(all_proteins)

            # Store metadata about the expansion
            self._current_input.metadata["string_extension"] = {
                "original_seeds": seed_proteins,
                "extend_network": extend_network,
                "min_score": min_score,
                "expanded_proteins": len(all_proteins),
                "total_interactions": len(all_interactions),
            }

            logger.info(
                f"STRING network expanded: {len(seed_proteins)} seeds -> "
                f"{len(all_proteins)} proteins, {len(all_interactions)} interactions"
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
    config: PipelineConfig | None = None,
) -> DataFetchAgent:
    """
    Factory function to create a DataFetchAgent.

    Args:
        string_client: Optional STRING client.
        pubmed_client: Optional PubMed client.
        config: Optional pipeline configuration.

    Returns:
        Configured DataFetchAgent instance.
    """
    return DataFetchAgent(
        string_client=string_client,
        pubmed_client=pubmed_client,
        config=config,
    )
