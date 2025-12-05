"""
Knowledge Graph Orchestrator - Coordinates data fetching and pipeline processing.

The orchestrator is the main entry point for building knowledge graphs.
It coordinates:
1. DataFetchAgent - LLM-driven data collection (STRING, PubMed)
2. KGPipeline - Deterministic processing (extraction, graph building)

This hybrid architecture separates LLM reasoning (for query construction)
from deterministic processing (for relationship extraction and graph building).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from dotenv import load_dotenv

from clients.pubmed_client import PubMedClient
from clients.string_client import StringClient
from models.knowledge_graph import KnowledgeGraph
from pipeline.config import PipelineConfig
from pipeline.kg_pipeline import KGPipeline
from pipeline.models import PipelineInput, PipelineResult

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from agent.data_fetch_agent import DataFetchAgent

# Load environment variables
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

logger = logging.getLogger(__name__)


class KGOrchestrator:
    """
    Orchestrates the hybrid KG construction architecture.

    Coordinates:
    - DataFetchAgent: LLM-driven data collection
    - KGPipeline: Deterministic processing

    The orchestrator maintains the current graph state and supports
    iterative expansion through the expand() method.

    Example:
        >>> orchestrator = KGOrchestrator()
        >>> graph = await orchestrator.build("Build a KG for the orexin pathway")
        >>> print(f"Built graph with {graph.to_summary()['node_count']} nodes")
        >>>
        >>> # Later, expand the graph
        >>> await orchestrator.expand("Add narcolepsy disease associations")
    """

    def __init__(
        self,
        fetch_agent: "DataFetchAgent | None" = None,
        pipeline: KGPipeline | None = None,
        config: PipelineConfig | None = None,
        extractor_name: str = "cerebras",
    ):
        """
        Initialize the orchestrator.

        Args:
            fetch_agent: Data fetch agent. Created if not provided.
            pipeline: KG pipeline. Created if not provided.
            config: Pipeline configuration. Created from env if not provided.
            extractor_name: LLM extractor for relationship mining.
        """
        # Create or use provided config
        self._config = config or PipelineConfig.from_env()

        # Create clients
        self._string_client = StringClient()
        self._pubmed_client = PubMedClient(api_key=os.getenv("NCBI_API_KEY"))

        # Create or use provided components (lazy import to avoid circular dependency)
        if fetch_agent is not None:
            self._fetch_agent = fetch_agent
        else:
            from agent.data_fetch_agent import DataFetchAgent
            self._fetch_agent = DataFetchAgent(
                string_client=self._string_client,
                pubmed_client=self._pubmed_client,
                config=self._config,
            )
        self._pipeline = pipeline or KGPipeline(extractor_name=extractor_name)

        # Current state
        self._graph: KnowledgeGraph | None = None
        self._accumulated_input: PipelineInput | None = None

    async def close(self) -> None:
        """
        Close all async clients.

        Should be called when done with the orchestrator to prevent
        "Task was destroyed but it is pending" warnings.
        """
        await self._string_client.close()
        await self._pubmed_client.close()

    async def __aenter__(self) -> "KGOrchestrator":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - ensures clients are closed."""
        await self.close()

    @property
    def graph(self) -> KnowledgeGraph | None:
        """Get the current knowledge graph."""
        return self._graph

    @property
    def has_graph(self) -> bool:
        """Check if a graph has been built."""
        return self._graph is not None

    async def build(
        self,
        user_query: str,
        max_articles: int = 50,
        run_link_prediction: bool = True,
    ) -> tuple[KnowledgeGraph, PipelineInput]:
        """
        Build a new knowledge graph from a user query.

        This is the main entry point for graph construction.

        Steps:
        1. DataFetchAgent collects data (STRING, PubMed) using LLM reasoning
        2. KGPipeline processes data (extraction, graph building) deterministically
        3. (Optional) Run ML link prediction to add novel predictions

        Args:
            user_query: Natural language query describing what to build.
            max_articles: Maximum number of articles to fetch.
            run_link_prediction: Whether to run ML link prediction after building.
                Default True. Set to False to skip link prediction.

        Returns:
            Tuple of (KnowledgeGraph, PipelineInput) containing the built graph
            and accumulated pipeline input data for display purposes.
        """
        logger.info(f"KGOrchestrator.build: {user_query}")

        # Step 1: Fetch data using the agent
        pipeline_input = await self._fetch_agent.fetch(
            user_query=user_query,
            max_articles=max_articles,
        )

        logger.info(
            f"Data fetched: {pipeline_input.article_count} articles, "
            f"{pipeline_input.interaction_count} STRING interactions"
        )

        # Step 2: Run the pipeline
        result = await self._pipeline.run(pipeline_input)

        # Store state
        self._graph = result.graph
        self._accumulated_input = pipeline_input

        logger.info(
            f"Graph built: {result.graph.to_summary()['node_count']} nodes, "
            f"{result.graph.to_summary()['edge_count']} edges"
        )

        # Step 3: Run link prediction (optional but enabled by default)
        if run_link_prediction:
            link_pred_result = self._run_link_prediction()
            if link_pred_result.get("status") == "success":
                enrichment_count = link_pred_result.get("stats", {}).get("enrichment_count", 0)
                novel_count = link_pred_result.get("stats", {}).get("novel_prediction_count", 0)
                if enrichment_count > 0 or novel_count > 0:
                    logger.info(
                        f"Link prediction: {enrichment_count} enrichment edges, "
                        f"{novel_count} novel predictions added"
                    )

        # Save graph to disk if configured
        if self._config.save_graph:
            saved_path = self._save_graph(user_query)
            logger.info(f"Graph saved to: {saved_path}")

        return self._graph, self._accumulated_input

    def _run_link_prediction(
        self,
        min_ml_score: float = 0.5,
        max_enrichment: int = 50,
        max_novel: int = 50,
    ) -> dict:
        """
        Run link prediction on the current graph.

        This method applies the ML link predictor to find:
        1. Enrichment edges: Known STRING interactions not yet in the graph
        2. Novel predictions: ML-predicted interactions not in STRING

        Args:
            min_ml_score: Minimum ML score for novel predictions.
            max_enrichment: Maximum enrichment edges to add.
            max_novel: Maximum novel predictions to add.

        Returns:
            Result dictionary with status, predictions, and stats.
        """
        from models.knowledge_graph import RelationshipType, EvidenceSource

        if self._graph is None:
            return {"status": "error", "message": "No graph built"}

        # Try to load link predictor
        try:
            # Try PyG first
            pyg_path = Path(__file__).parent.parent / "models" / "pyg_link_predictor.pkl"
            gensim_path = Path(__file__).parent.parent / "models" / "gensim_link_predictor.pkl"
            link_predictor = None

            if pyg_path.exists():
                try:
                    from ml.pyg_link_predictor import PyGLinkPredictor
                    link_predictor = PyGLinkPredictor.load(str(pyg_path))
                except ImportError:
                    pass

            if link_predictor is None and gensim_path.exists():
                try:
                    from ml.link_predictor import LinkPredictor
                    link_predictor = LinkPredictor.load(str(gensim_path))
                except Exception:
                    pass

            if link_predictor is None:
                return {"status": "skipped", "message": "No link predictor model found"}

        except Exception as e:
            logger.warning(f"Failed to load link predictor: {e}")
            return {"status": "error", "message": f"Link predictor load failed: {e}"}

        # Get all protein/gene entities from the graph
        proteins = [
            entity_id
            for entity_id, data in self._graph.entities.items()
            if data.get("type") in ["protein", "gene"]
        ]

        if len(proteins) < 2:
            return {
                "status": "skipped",
                "message": "Not enough proteins for prediction",
                "stats": {"total_pairs_evaluated": 0, "enrichment_count": 0, "novel_prediction_count": 0},
            }

        # Generate candidate pairs not already in graph
        candidate_pairs = []
        for i, p1 in enumerate(proteins):
            for p2 in proteins[i + 1:]:
                existing = self._graph.get_relationship(p1, p2)
                reverse_existing = self._graph.get_relationship(p2, p1)
                if not existing and not reverse_existing:
                    candidate_pairs.append((p1, p2))

        if not candidate_pairs:
            return {
                "status": "success",
                "message": "All protein pairs already have relationships",
                "enrichment": [],
                "novel_predictions": [],
                "stats": {"total_pairs_evaluated": 0, "enrichment_count": 0, "novel_prediction_count": 0},
            }

        # Run predictions
        try:
            predictions = link_predictor.predict(candidate_pairs)
        except Exception as e:
            return {"status": "error", "message": f"Prediction failed: {e}"}

        # Separate enrichment vs novel predictions
        enrichment_results = []
        novel_results = []
        proteins_not_in_string = 0

        for pred in predictions:
            if pred.get("error"):
                if "Unknown protein" in str(pred.get("error", "")):
                    proteins_not_in_string += 1
                continue

            ml_score = pred.get("ml_score", 0.0)
            in_string = pred.get("in_string", False)

            if in_string:
                enrichment_results.append({
                    "protein1": pred["protein1"],
                    "protein2": pred["protein2"],
                    "ml_score": round(ml_score, 4),
                    "in_string": True,
                })
            elif ml_score >= min_ml_score:
                novel_results.append({
                    "protein1": pred["protein1"],
                    "protein2": pred["protein2"],
                    "ml_score": round(ml_score, 4),
                    "in_string": False,
                })

        # Sort by score
        enrichment_results.sort(key=lambda x: x["ml_score"], reverse=True)
        novel_results.sort(key=lambda x: x["ml_score"], reverse=True)

        # Limit results
        top_enrichment = enrichment_results[:max_enrichment]
        top_novel = novel_results[:max_novel]

        # Add enrichment edges to graph
        for pred in top_enrichment:
            self._graph.add_relationship(
                source=pred["protein1"],
                target=pred["protein2"],
                rel_type=RelationshipType.INTERACTS_WITH,
                ml_score=pred["ml_score"],
                link_category="enrichment",
                evidence=[{
                    "source_type": EvidenceSource.STRING.value,
                    "source_id": "string_enrichment",
                    "confidence": pred["ml_score"],
                }],
            )

        # Add novel predictions to graph
        for pred in top_novel:
            self._graph.add_relationship(
                source=pred["protein1"],
                target=pred["protein2"],
                rel_type=RelationshipType.HYPOTHESIZED,
                ml_score=pred["ml_score"],
                link_category="novel_prediction",
                evidence=[{
                    "source_type": EvidenceSource.ML_PREDICTED.value,
                    "source_id": "link_predictor",
                    "confidence": pred["ml_score"],
                }],
            )

        return {
            "status": "success",
            "message": f"Added {len(top_enrichment)} enrichment edges and {len(top_novel)} novel predictions",
            "enrichment": top_enrichment,
            "novel_predictions": top_novel,
            "stats": {
                "total_pairs_evaluated": len(candidate_pairs),
                "enrichment_count": len(top_enrichment),
                "novel_prediction_count": len(top_novel),
                "proteins_not_in_string": proteins_not_in_string,
                "total_enrichment_available": len(enrichment_results),
                "total_novel_available": len(novel_results),
            },
        }

    async def expand(
        self,
        expansion_query: str,
        max_articles: int = 30,
    ) -> KnowledgeGraph:
        """
        Expand the current graph with additional data.

        Fetches new data based on the expansion query and merges
        it with the existing graph.

        Args:
            expansion_query: Query for additional data.
            max_articles: Maximum additional articles.

        Returns:
            Updated KnowledgeGraph.

        Raises:
            ValueError: If no graph has been built yet.
        """
        if self._graph is None:
            raise ValueError("No graph to expand. Call build() first.")

        logger.info(f"KGOrchestrator.expand: {expansion_query}")

        # Fetch additional data
        new_input = await self._fetch_agent.expand(
            expansion_query=expansion_query,
            max_articles=max_articles,
        )

        # Merge with accumulated input
        if self._accumulated_input is not None:
            merged_input = self._accumulated_input.merge(new_input)
        else:
            merged_input = new_input

        logger.info(
            f"Expansion data: {new_input.article_count} new articles, "
            f"{new_input.interaction_count} new STRING interactions"
        )

        # Run pipeline on merged input
        result = await self._pipeline.run(merged_input)

        # Update state
        self._graph = result.graph
        self._accumulated_input = merged_input

        logger.info(
            f"Graph expanded: {result.graph.to_summary()['node_count']} nodes, "
            f"{result.graph.to_summary()['edge_count']} edges"
        )

        # Save expanded graph if configured
        if self._config.save_graph:
            saved_path = self._save_graph(expansion_query, suffix="_expanded")
            logger.info(f"Expanded graph saved to: {saved_path}")

        return self._graph

    def _save_graph(self, query: str, suffix: str = "") -> str:
        """
        Save the current graph to disk.

        Args:
            query: Query used to build the graph (for filename).
            suffix: Optional suffix for the filename.

        Returns:
            Path to the saved file.
        """
        if self._graph is None:
            raise ValueError("No graph to save")

        # Create output directory
        output_dir = Path(self._config.graph_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from query
        # Extract key terms (e.g., "orexin signaling pathway" -> "orexin_signaling")
        query_slug = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
        query_slug = '_'.join(query_slug.split()[:3])  # First 3 words
        if not query_slug:
            query_slug = "graph"

        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{query_slug}{suffix}_{timestamp}"

        # Save using configured format
        file_path = output_dir / filename
        saved_path = self._graph.save(str(file_path), format=self._config.graph_format)

        return saved_path

    def get_graph_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current knowledge graph.

        Returns:
            Graph summary dictionary.
        """
        if self._graph is None:
            return {"error": "No graph built yet", "node_count": 0, "edge_count": 0}
        return self._graph.to_summary()

    async def query(self, question: str) -> dict[str, Any]:
        """
        Query the current knowledge graph.

        This is a placeholder for Q&A functionality that would
        be implemented in the Q&A agent.

        Args:
            question: Natural language question.

        Returns:
            Query result dictionary.
        """
        if self._graph is None:
            return {"error": "No graph to query. Build a graph first."}

        # Basic graph-based answering
        # This would be expanded in the Q&A agent
        summary = self._graph.to_summary()
        return {
            "status": "success",
            "graph_summary": summary,
            "message": "Graph built. Use the Q&A agent for detailed queries.",
        }

    def reset(self) -> None:
        """Reset the orchestrator state."""
        self._graph = None
        self._accumulated_input = None
        logger.info("Orchestrator state reset")


async def create_orchestrator(
    config: PipelineConfig | None = None,
    extractor_name: str = "cerebras",
) -> KGOrchestrator:
    """
    Factory function to create a KGOrchestrator.

    Args:
        config: Optional pipeline configuration.
        extractor_name: LLM extractor for relationship mining.

    Returns:
        Configured KGOrchestrator instance.
    """
    return KGOrchestrator(config=config, extractor_name=extractor_name)


# Module-level orchestrator instance for use by the Q&A agent
_orchestrator: KGOrchestrator | None = None


def get_orchestrator() -> KGOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = KGOrchestrator()
    return _orchestrator


def set_orchestrator(orchestrator: KGOrchestrator) -> None:
    """Set the global orchestrator instance."""
    global _orchestrator
    _orchestrator = orchestrator
