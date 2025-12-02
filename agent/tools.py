"""
Agent tools for the Scientific Knowledge Graph Agent.

This module provides the tools available to the orchestrator agent
for building and querying biological knowledge graphs.
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from ml.link_predictor import LinkPredictor
from clients.string_client import StringClient
from clients.pubmed_client import PubMedClient
from extraction.relationship_extractor import RelationshipExtractor
from extraction.relationship_inferrer import RelationshipInferrer, HypothesisGenerator
from models.knowledge_graph import (
    KnowledgeGraph,
    RelationshipType,
    EvidenceSource,
    EntityType,
)

logger = logging.getLogger(__name__)


class AgentTools:
    """Tools available to the orchestrator agent for knowledge graph operations."""

    def __init__(
        self,
        link_predictor: LinkPredictor,
        string_client: StringClient,
        pubmed_client: PubMedClient,
        relationship_extractor: RelationshipExtractor,
        relationship_inferrer: RelationshipInferrer,
    ):
        """
        Initialize agent tools with all required dependencies.

        Args:
            link_predictor: Trained ML link predictor
            string_client: STRING database API client
            pubmed_client: PubMed/PubTator API client
            relationship_extractor: LLM relationship extraction
            relationship_inferrer: LLM relationship inference
        """
        self.link_predictor = link_predictor
        self.string_client = string_client
        self.pubmed_client = pubmed_client
        self.relationship_extractor = relationship_extractor
        self.relationship_inferrer = relationship_inferrer
        self.hypothesis_generator = HypothesisGenerator()
        self.current_graph: Optional[KnowledgeGraph] = None

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Get OpenAI function definitions for all tools.

        Returns:
            List of tool definitions in OpenAI function calling format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_string_network",
                    "description": "Fetch known protein interaction network from STRING database for seed proteins",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "seed_proteins": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of protein/gene names (e.g., ['HCRTR1', 'HCRTR2', 'PPARG'])"
                            },
                            "min_score": {
                                "type": "integer",
                                "description": "Minimum STRING confidence score (0-1000), default 700 for high confidence",
                                "default": 700
                            }
                        },
                        "required": ["seed_proteins"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_string_partners",
                    "description": "Get interaction partners from STRING for proteins (expands the network)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "proteins": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of protein names to find partners for"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum partners per protein",
                                "default": 30
                            }
                        },
                        "required": ["proteins"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_literature",
                    "description": "Search PubMed for relevant biomedical literature",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "PubMed search query (supports Boolean operators)"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 50
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_entity_annotations",
                    "description": "Get NER annotations (genes, diseases, chemicals) from PubTator for articles",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pmids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of PubMed IDs to annotate"
                            }
                        },
                        "required": ["pmids"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_relationships",
                    "description": "Extract typed relationships between entities from abstracts using LLM",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "articles": {
                                "type": "array",
                                "description": "List of article objects with pmid and abstract"
                            },
                            "annotations_by_pmid": {
                                "type": "object",
                                "description": "Dict mapping PMID to list of annotations"
                            }
                        },
                        "required": ["articles", "annotations_by_pmid"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "build_knowledge_graph",
                    "description": "Build an evidence-backed knowledge graph from STRING interactions and literature relationships",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "string_interactions": {
                                "type": "array",
                                "description": "Interactions from STRING"
                            },
                            "literature_relationships": {
                                "type": "array",
                                "description": "Extracted relationships from literature"
                            },
                            "entities": {
                                "type": "object",
                                "description": "Dict mapping entity type to list of entity dicts"
                            }
                        },
                        "required": ["string_interactions", "literature_relationships", "entities"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "predict_novel_links",
                    "description": "Apply ML link predictor to find novel protein-protein interactions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "min_ml_score": {
                                "type": "number",
                                "description": "Minimum ML prediction score (0-1)",
                                "default": 0.7
                            },
                            "max_predictions": {
                                "type": "integer",
                                "description": "Maximum number of predictions to return",
                                "default": 20
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "infer_novel_relationships",
                    "description": "Use LLM to infer relationship types for top ML predictions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "predictions": {
                                "type": "array",
                                "description": "List of predictions from predict_novel_links"
                            },
                            "max_inferences": {
                                "type": "integer",
                                "description": "Maximum number of inferences to run",
                                "default": 5
                            }
                        },
                        "required": ["predictions"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_evidence",
                    "description": "Get all evidence for a relationship between two proteins",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "protein1": {"type": "string", "description": "First protein identifier"},
                            "protein2": {"type": "string", "description": "Second protein identifier"}
                        },
                        "required": ["protein1", "protein2"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_graph_summary",
                    "description": "Get summary statistics of the current knowledge graph",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_protein_neighborhood",
                    "description": "Get the neighborhood (connected proteins) of a specific protein",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "protein": {"type": "string", "description": "Protein identifier"},
                            "max_neighbors": {
                                "type": "integer",
                                "description": "Maximum neighbors to return",
                                "default": 10
                            }
                        },
                        "required": ["protein"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_hypothesis",
                    "description": "Generate a detailed testable hypothesis for a predicted protein interaction",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "protein1": {"type": "string", "description": "First protein"},
                            "protein2": {"type": "string", "description": "Second protein"}
                        },
                        "required": ["protein1", "protein2"]
                    }
                }
            },
            # Graph Data Science Tools
            {
                "type": "function",
                "function": {
                    "name": "compute_centrality",
                    "description": "Compute centrality scores (PageRank, betweenness, degree, closeness) to identify important hub proteins",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "description": "Centrality method: pagerank, betweenness, degree, closeness",
                                "default": "pagerank"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top entities to return",
                                "default": 10
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "detect_communities",
                    "description": "Detect functional communities/modules in the graph using Louvain algorithm",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_path",
                    "description": "Find the shortest path between two entities to trace relationship chains",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "Source entity ID"},
                            "target": {"type": "string", "description": "Target entity ID"}
                        },
                        "required": ["source", "target"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_diamond_module",
                    "description": "Run DIAMOnD algorithm to detect disease modules from seed genes (Ghiassian et al. 2015)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "seed_genes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Known disease-associated genes"
                            },
                            "max_iterations": {
                                "type": "integer",
                                "description": "Maximum genes to add to module",
                                "default": 200
                            }
                        },
                        "required": ["seed_genes"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_drug_disease_proximity",
                    "description": "Calculate network proximity between drug targets and disease genes (Guney et al. 2016)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "drug_targets": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Drug target genes/proteins"
                            },
                            "disease_genes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Disease-associated genes"
                            }
                        },
                        "required": ["drug_targets", "disease_genes"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "predict_drug_synergy",
                    "description": "Predict synergy potential for two drug targets in combination therapy (Cheng et al. 2019)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target1": {"type": "string", "description": "First drug target"},
                            "target2": {"type": "string", "description": "Second drug target"},
                            "disease_genes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Disease-associated genes for context"
                            }
                        },
                        "required": ["target1", "target2", "disease_genes"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_target_robustness",
                    "description": "Analyze therapeutic index by measuring disease impact vs global network disruption",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string", "description": "Drug target to analyze"},
                            "disease_genes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Disease-associated genes"
                            }
                        },
                        "required": ["target", "disease_genes"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_capabilities",
                    "description": "Get a description of what this agent can do",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

    # -------------------------------------------------------------------------
    # STRING Tools
    # -------------------------------------------------------------------------

    async def get_string_network(
        self,
        seed_proteins: list[str],
        min_score: int = 700,
    ) -> dict[str, Any]:
        """
        Fetch known interaction network from STRING for seed proteins.

        Args:
            seed_proteins: List of protein/gene names (e.g., ["HCRTR1", "HCRTR2", "OX1R"])
            min_score: Minimum STRING combined score (0-1000), default 700 (high confidence)

        Returns:
            Dictionary with interactions, proteins_found, and count.
        """
        try:
            interactions = await self.string_client.get_network(
                proteins=seed_proteins,
                min_score=min_score,
                network_type="physical",
            )

            # Collect unique proteins found
            proteins_found = set()
            for interaction in interactions:
                proteins_found.add(interaction.get("preferredName_A", ""))
                proteins_found.add(interaction.get("preferredName_B", ""))
            proteins_found.discard("")

            return {
                "interactions": interactions,
                "proteins_found": list(proteins_found),
                "count": len(interactions),
                "seed_proteins": seed_proteins,
                "min_score": min_score,
            }
        except Exception as e:
            logger.error(f"Error fetching STRING network: {e}")
            return {
                "interactions": [],
                "proteins_found": [],
                "count": 0,
                "error": str(e),
            }

    async def get_string_partners(
        self,
        proteins: list[str],
        limit: int = 30,
    ) -> dict[str, Any]:
        """
        Get interaction partners from STRING for given proteins.

        Args:
            proteins: List of protein/gene names to find partners for
            limit: Maximum partners per protein (default 30)

        Returns:
            Dictionary with partners and count.
        """
        try:
            partners = await self.string_client.get_interaction_partners(
                proteins=proteins,
                limit=limit,
                min_score=400,
            )

            return {
                "partners": partners,
                "count": len(partners),
                "query_proteins": proteins,
            }
        except Exception as e:
            logger.error(f"Error fetching STRING partners: {e}")
            return {
                "partners": [],
                "count": 0,
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # PubMed Tools
    # -------------------------------------------------------------------------

    async def search_literature(
        self,
        query: str,
        max_results: int = 50,
    ) -> dict[str, Any]:
        """
        Search PubMed for relevant abstracts.

        Args:
            query: PubMed search query (supports Boolean operators)
            max_results: Maximum number of results (default 50)

        Returns:
            Dictionary with articles, pmids, and count.
        """
        try:
            pmids = await self.pubmed_client.search(
                query=query,
                max_results=max_results,
            )

            if not pmids:
                return {
                    "articles": [],
                    "pmids": [],
                    "count": 0,
                    "query": query,
                }

            articles = await self.pubmed_client.fetch_abstracts(pmids)

            return {
                "articles": articles,
                "pmids": pmids,
                "count": len(articles),
                "query": query,
            }
        except Exception as e:
            logger.error(f"Error searching literature: {e}")
            return {
                "articles": [],
                "pmids": [],
                "count": 0,
                "error": str(e),
            }

    async def get_entity_annotations(
        self,
        pmids: list[str],
    ) -> dict[str, Any]:
        """
        Get NER annotations from PubTator for articles.

        Args:
            pmids: List of PubMed IDs to get annotations for

        Returns:
            Dictionary with annotations and annotations_by_pmid.
        """
        try:
            annotations = await self.pubmed_client.get_pubtator_annotations(
                pmids=pmids,
                entity_types=["Gene", "Disease", "Chemical"],
            )

            # Group annotations by PMID
            annotations_by_pmid: dict[str, list[dict]] = {}
            entity_type_counts: dict[str, int] = {}

            for annot in annotations:
                pmid = annot.get("pmid", "")
                if pmid not in annotations_by_pmid:
                    annotations_by_pmid[pmid] = []
                annotations_by_pmid[pmid].append(annot)

                etype = annot.get("entity_type", "unknown")
                entity_type_counts[etype] = entity_type_counts.get(etype, 0) + 1

            return {
                "annotations": annotations,
                "annotations_by_pmid": annotations_by_pmid,
                "entity_types": entity_type_counts,
                "count": len(annotations),
            }
        except Exception as e:
            logger.error(f"Error getting annotations: {e}")
            return {
                "annotations": [],
                "annotations_by_pmid": {},
                "entity_types": {},
                "count": 0,
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # Extraction Tools
    # -------------------------------------------------------------------------

    async def extract_relationships(
        self,
        articles: list[dict[str, Any]],
        annotations_by_pmid: dict[str, list[dict]],
    ) -> dict[str, Any]:
        """
        Extract typed relationships from abstracts using LLM.

        Args:
            articles: List of article dictionaries with pmid, abstract
            annotations_by_pmid: Dict mapping PMID to NER annotations

        Returns:
            Dictionary with relationships and count.
        """
        try:
            # Prepare batch extraction input
            extraction_inputs = []
            for article in articles:
                pmid = article.get("pmid", "")
                abstract = article.get("abstract", "")

                if not abstract:
                    continue

                # Get entity pairs from annotations
                annots = annotations_by_pmid.get(pmid, [])
                gene_entities = [
                    a.get("entity_text", "")
                    for a in annots
                    if a.get("entity_type") == "Gene"
                ]
                gene_entities = list(set(gene_entities))

                # Generate pairs
                entity_pairs = []
                for i, e1 in enumerate(gene_entities):
                    for e2 in gene_entities[i + 1:]:
                        entity_pairs.append((e1, e2))

                if entity_pairs:
                    extraction_inputs.append({
                        "pmid": pmid,
                        "abstract": abstract,
                        "entity_pairs": entity_pairs,
                    })

            if not extraction_inputs:
                return {
                    "relationships": [],
                    "count": 0,
                    "message": "No entity pairs to extract from",
                }

            # Batch extraction
            relationships = await self.relationship_extractor.batch_extract(
                abstracts=extraction_inputs,
                max_concurrent=5,
            )

            return {
                "relationships": relationships,
                "count": len(relationships),
            }
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return {
                "relationships": [],
                "count": 0,
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # Graph Tools
    # -------------------------------------------------------------------------

    def build_knowledge_graph(
        self,
        string_interactions: list[dict[str, Any]],
        literature_relationships: list[dict[str, Any]],
        entities: dict[str, list[dict]],
    ) -> dict[str, Any]:
        """
        Build evidence-backed knowledge graph from all sources.

        Args:
            string_interactions: Interactions from STRING
            literature_relationships: Extracted relationships from literature
            entities: Dict mapping entity type to list of entity dicts

        Returns:
            Dictionary with summary and statistics.
        """
        try:
            self.current_graph = KnowledgeGraph()

            # Add entities from annotations
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_id = entity.get("entity_text") or entity.get("name", "")
                    if entity_id:
                        self.current_graph.add_entity(
                            entity_id=entity_id,
                            entity_type=entity_type.lower(),
                            name=entity_id,
                            entity_ncbi_id=entity.get("entity_id", ""),
                        )

            # Add STRING interactions
            for interaction in string_interactions:
                protein_a = interaction.get("preferredName_A", "")
                protein_b = interaction.get("preferredName_B", "")
                score = interaction.get("score", 0.0)

                if protein_a and protein_b:
                    self.current_graph.add_entity(protein_a, EntityType.PROTEIN.value, protein_a)
                    self.current_graph.add_entity(protein_b, EntityType.PROTEIN.value, protein_b)

                    self.current_graph.add_relationship(
                        source=protein_a,
                        target=protein_b,
                        rel_type=RelationshipType.INTERACTS_WITH,
                        evidence=[{
                            "source_type": EvidenceSource.STRING.value,
                            "source_id": f"STRING:{protein_a}-{protein_b}",
                            "confidence": float(score),
                        }],
                    )

            # Add literature relationships
            for rel in literature_relationships:
                entity1 = rel.get("entity1", "")
                entity2 = rel.get("entity2", "")
                rel_type_str = rel.get("relationship", "associated_with")
                confidence = rel.get("confidence", 0.5)
                pmid = rel.get("pmid", "")
                evidence_text = rel.get("evidence_text", "")

                if entity1 and entity2:
                    rel_type_map = {
                        "activates": RelationshipType.ACTIVATES,
                        "inhibits": RelationshipType.INHIBITS,
                        "associated_with": RelationshipType.ASSOCIATED_WITH,
                        "regulates": RelationshipType.REGULATES,
                        "binds_to": RelationshipType.BINDS_TO,
                        "cooccurs_with": RelationshipType.COOCCURS_WITH,
                    }
                    rel_type = rel_type_map.get(
                        rel_type_str.lower(),
                        RelationshipType.ASSOCIATED_WITH
                    )

                    self.current_graph.add_relationship(
                        source=entity1,
                        target=entity2,
                        rel_type=rel_type,
                        evidence=[{
                            "source_type": EvidenceSource.LITERATURE.value,
                            "source_id": f"PMID:{pmid}",
                            "confidence": float(confidence),
                            "text_snippet": evidence_text,
                        }],
                    )

            summary = self.current_graph.to_summary()

            return {
                "summary": summary,
                "node_count": summary.get("node_count", 0),
                "edge_count": summary.get("edge_count", 0),
            }
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return {
                "summary": {},
                "node_count": 0,
                "edge_count": 0,
                "error": str(e),
            }

    def predict_novel_links(
        self,
        min_ml_score: float = 0.7,
        max_predictions: int = 20,
    ) -> dict[str, Any]:
        """
        Apply link predictor to find novel interactions in the graph.

        Args:
            min_ml_score: Minimum ML prediction score (default 0.7)
            max_predictions: Maximum number of predictions to return

        Returns:
            Dictionary with predictions and count.
        """
        try:
            if self.current_graph is None:
                return {
                    "predictions": [],
                    "count": 0,
                    "error": "No knowledge graph built. Call build_knowledge_graph first.",
                }

            # Get all protein entities from the graph
            proteins = [
                entity_id
                for entity_id, data in self.current_graph.entities.items()
                if data.get("type") in ["protein", "gene"]
            ]

            if len(proteins) < 2:
                return {
                    "predictions": [],
                    "count": 0,
                    "message": "Not enough proteins in graph for prediction",
                }

            # Generate pairs not already in graph
            candidate_pairs = []
            for i, p1 in enumerate(proteins):
                for p2 in proteins[i + 1:]:
                    existing = self.current_graph.get_relationship(p1, p2)
                    if not existing:
                        candidate_pairs.append((p1, p2))

            if not candidate_pairs:
                return {
                    "predictions": [],
                    "count": 0,
                    "message": "All protein pairs already have relationships",
                }

            # Run predictions
            pairs_to_predict = candidate_pairs[:max_predictions * 3]
            predictions = self.link_predictor.predict(pairs_to_predict)

            # Filter by score and sort
            filtered = [
                p for p in predictions
                if p.get("ml_score", 0) >= min_ml_score and not p.get("error")
            ]
            filtered.sort(key=lambda x: x.get("ml_score", 0), reverse=True)
            top_predictions = filtered[:max_predictions]

            # Add predictions to graph as hypothesized relationships
            for pred in top_predictions:
                self.current_graph.add_relationship(
                    source=pred["protein1"],
                    target=pred["protein2"],
                    rel_type=RelationshipType.HYPOTHESIZED,
                    ml_score=pred["ml_score"],
                    evidence=[{
                        "source_type": EvidenceSource.ML_PREDICTED.value,
                        "source_id": "link_predictor",
                        "confidence": pred["ml_score"],
                    }],
                )

            return {
                "predictions": top_predictions,
                "count": len(top_predictions),
                "total_candidates": len(candidate_pairs),
            }
        except Exception as e:
            logger.error(f"Error predicting links: {e}")
            return {
                "predictions": [],
                "count": 0,
                "error": str(e),
            }

    async def infer_novel_relationships(
        self,
        predictions: list[dict[str, Any]],
        max_inferences: int = 5,
    ) -> dict[str, Any]:
        """
        Use LLM to infer relationship types for top ML predictions.

        Args:
            predictions: List of prediction dicts from predict_novel_links
            max_inferences: Maximum number of inferences to run (default 5)

        Returns:
            Dictionary with inferences and count.
        """
        try:
            if self.current_graph is None:
                return {
                    "inferences": [],
                    "count": 0,
                    "error": "No knowledge graph built.",
                }

            top_preds = predictions[:max_inferences]

            inference_inputs = [
                {
                    "source": pred.get("protein1", ""),
                    "target": pred.get("protein2", ""),
                    "ml_score": pred.get("ml_score", 0.0),
                }
                for pred in top_preds
            ]

            inferences = await self.relationship_inferrer.batch_infer(
                predictions=inference_inputs,
                graph=self.current_graph,
                max_concurrent=3,
            )

            # Update graph with inferred relationship types
            for inference in inferences:
                if inference.get("hypothesized_relationship"):
                    protein_a = inference.get("protein_a", "")
                    protein_b = inference.get("protein_b", "")

                    rel_type_str = inference.get("hypothesized_relationship", "").lower()
                    rel_type_map = {
                        "activates": RelationshipType.ACTIVATES,
                        "inhibits": RelationshipType.INHIBITS,
                        "binds_to": RelationshipType.BINDS_TO,
                        "regulates": RelationshipType.REGULATES,
                        "complex_member": RelationshipType.BINDS_TO,
                        "pathway_neighbor": RelationshipType.ASSOCIATED_WITH,
                    }
                    rel_type = rel_type_map.get(rel_type_str, RelationshipType.HYPOTHESIZED)

                    self.current_graph.add_relationship(
                        source=protein_a,
                        target=protein_b,
                        rel_type=rel_type,
                        ml_score=inference.get("ml_score"),
                        reasoning=inference.get("reasoning"),
                        confidence_level=inference.get("confidence"),
                        validation_experiments=inference.get("validation_experiments", []),
                    )

            return {
                "inferences": inferences,
                "count": len(inferences),
            }
        except Exception as e:
            logger.error(f"Error inferring relationships: {e}")
            return {
                "inferences": [],
                "count": 0,
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # Query Tools
    # -------------------------------------------------------------------------

    def query_evidence(
        self,
        protein1: str,
        protein2: str,
    ) -> dict[str, Any]:
        """
        Get all evidence for a relationship between two proteins.

        Args:
            protein1: First protein identifier
            protein2: Second protein identifier

        Returns:
            Dictionary with relationships and evidence counts.
        """
        try:
            if self.current_graph is None:
                return {
                    "relationships": {},
                    "evidence_count": 0,
                    "error": "No knowledge graph built.",
                }

            relationships = self.current_graph.get_relationship(protein1, protein2)
            reverse_rels = self.current_graph.get_relationship(protein2, protein1)

            all_rels = {}
            if relationships:
                all_rels.update(relationships)
            if reverse_rels:
                for k, v in reverse_rels.items():
                    if k not in all_rels:
                        all_rels[f"{k}_reverse"] = v

            evidence_count = 0
            has_string = False
            has_literature = False
            has_ml = False

            for rel_data in all_rels.values():
                evidence = rel_data.get("evidence", [])
                evidence_count += len(evidence)
                for ev in evidence:
                    source_type = ev.get("source_type", "")
                    if source_type == EvidenceSource.STRING.value:
                        has_string = True
                    elif source_type == EvidenceSource.LITERATURE.value:
                        has_literature = True
                    elif source_type == EvidenceSource.ML_PREDICTED.value:
                        has_ml = True

            return {
                "protein1": protein1,
                "protein2": protein2,
                "relationships": all_rels,
                "evidence_count": evidence_count,
                "has_string": has_string,
                "has_literature": has_literature,
                "has_ml": has_ml,
            }
        except Exception as e:
            logger.error(f"Error querying evidence: {e}")
            return {
                "relationships": {},
                "evidence_count": 0,
                "error": str(e),
            }

    def get_graph_summary(self) -> dict[str, Any]:
        """Get summary of current knowledge graph."""
        try:
            if self.current_graph is None:
                return {
                    "error": "No knowledge graph built. Use build_knowledge_graph first.",
                    "node_count": 0,
                    "edge_count": 0,
                }

            return self.current_graph.to_summary()
        except Exception as e:
            logger.error(f"Error getting graph summary: {e}")
            return {
                "error": str(e),
                "node_count": 0,
                "edge_count": 0,
            }

    def get_protein_neighborhood(
        self,
        protein: str,
        max_neighbors: int = 10,
    ) -> dict[str, Any]:
        """
        Get neighborhood of a protein in the knowledge graph.

        Args:
            protein: Protein identifier
            max_neighbors: Maximum neighbors to return (default 10)

        Returns:
            Dictionary with neighbors and count.
        """
        try:
            if self.current_graph is None:
                return {
                    "protein": protein,
                    "neighbors": [],
                    "count": 0,
                    "error": "No knowledge graph built.",
                }

            neighbors = self.current_graph.get_neighbors(protein, max_neighbors)

            formatted_neighbors = []
            for neighbor_id, rel_data in neighbors:
                formatted_neighbors.append({
                    "neighbor": neighbor_id,
                    "direction": rel_data.get("direction", "unknown"),
                    "relation_type": rel_data.get("relation_type", "unknown"),
                    "ml_score": rel_data.get("ml_score"),
                    "evidence_count": len(rel_data.get("evidence", [])),
                })

            return {
                "protein": protein,
                "neighbors": formatted_neighbors,
                "count": len(formatted_neighbors),
            }
        except Exception as e:
            logger.error(f"Error getting neighborhood: {e}")
            return {
                "protein": protein,
                "neighbors": [],
                "count": 0,
                "error": str(e),
            }

    async def generate_hypothesis(
        self,
        protein1: str,
        protein2: str,
    ) -> dict[str, Any]:
        """
        Generate a detailed testable hypothesis for a protein pair.

        Args:
            protein1: First protein
            protein2: Second protein

        Returns:
            Structured hypothesis dictionary.
        """
        try:
            if self.current_graph is None:
                return {"error": "No knowledge graph built."}

            # Get ML prediction
            ml_score = 0.5
            pred = self.link_predictor.predict([(protein1, protein2)])
            if pred and not pred[0].get("error"):
                ml_score = pred[0].get("ml_score", 0.5)

            # Infer relationship type
            inference = await self.relationship_inferrer.infer_relationship(
                protein1, protein2, ml_score, self.current_graph
            )

            # Generate full hypothesis
            hypothesis = await self.hypothesis_generator.generate_hypothesis(
                inference, self.current_graph
            )

            return hypothesis
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            return {"error": str(e)}

    # -------------------------------------------------------------------------
    # Graph Data Science Tools
    # -------------------------------------------------------------------------

    def compute_centrality(
        self,
        method: str = "pagerank",
        top_k: int = 10
    ) -> dict[str, Any]:
        """
        Compute centrality scores for entities in the knowledge graph.

        Centrality metrics help identify important hub proteins and bridge entities.

        Args:
            method: Centrality method - "pagerank", "betweenness", "degree", "closeness"
            top_k: Number of top entities to return

        Returns:
            Dictionary with centrality scores and interpretation.
        """
        if self.current_graph is None:
            return {"error": "No knowledge graph built. Call build_knowledge_graph first."}

        result = self.current_graph.compute_centrality(method, top_k)
        return {
            "method": result.method,
            "top_entities": [
                {"entity": entity, "score": round(score, 4)}
                for entity, score in result.top_entities
            ],
            "total_nodes": len(result.all_scores),
            "interpretation": f"Entities ranked by {method}. Higher scores indicate greater importance in the network."
        }

    def detect_communities(self) -> dict[str, Any]:
        """
        Detect functional communities/modules in the knowledge graph.

        Uses the Louvain algorithm to identify tightly connected clusters
        that may represent protein complexes or functional modules.

        Returns:
            Dictionary with communities and modularity score.
        """
        if self.current_graph is None:
            return {"error": "No knowledge graph built. Call build_knowledge_graph first."}

        result = self.current_graph.detect_communities()
        return {
            "num_communities": result.num_communities,
            "modularity": round(result.modularity, 4),
            "communities": result.communities,
            "interpretation": f"Found {result.num_communities} communities with modularity {result.modularity:.3f}."
        }

    def find_path(
        self,
        source: str,
        target: str
    ) -> dict[str, Any]:
        """
        Find the shortest path between two entities.

        Traces the relationship chain connecting two proteins, genes, or diseases.

        Args:
            source: Source entity ID
            target: Target entity ID

        Returns:
            Dictionary with path details and alternative paths.
        """
        if self.current_graph is None:
            return {"error": "No knowledge graph built. Call build_knowledge_graph first."}

        result = self.current_graph.find_shortest_path(source, target)

        if result.distance < 0:
            return {
                "source": source,
                "target": target,
                "path_exists": False,
                "message": f"No path found between {source} and {target}."
            }

        return {
            "source": source,
            "target": target,
            "path_exists": True,
            "distance": result.distance,
            "path": result.shortest_path,
            "alternative_paths_count": result.path_count,
            "interpretation": f"Found path of length {result.distance} with {result.path_count} total paths."
        }

    def run_diamond_module(
        self,
        seed_genes: list[str],
        max_iterations: int = 200
    ) -> dict[str, Any]:
        """
        Run DIAMOnD algorithm to detect disease modules from seed genes.

        Based on Ghiassian et al. (2015) for identifying disease-associated
        gene modules through network connectivity analysis.

        Args:
            seed_genes: Known disease-associated genes (e.g., from GWAS)
            max_iterations: Maximum genes to add to module

        Returns:
            Dictionary with module genes and ranked expansion candidates.
        """
        if self.current_graph is None:
            return {"error": "No knowledge graph built. Call build_knowledge_graph first."}

        result = self.current_graph.run_diamond(seed_genes, max_iterations)

        return {
            "seed_genes": result.seed_genes,
            "module_genes": result.module_genes[:20],
            "module_size": result.module_size,
            "iterations_run": result.iterations_run,
            "top_candidates": [
                {"gene": gene, "score": round(score, 4)}
                for gene, score in result.ranked_candidates[:10]
            ],
            "interpretation": f"Expanded from {len(result.seed_genes)} seeds to {result.module_size} genes."
        }

    def calculate_drug_disease_proximity(
        self,
        drug_targets: list[str],
        disease_genes: list[str]
    ) -> dict[str, Any]:
        """
        Calculate network proximity between drug targets and disease genes.

        Based on Guney et al. (2016) for predicting drug efficacy through
        network-based drug-disease relationships.

        Args:
            drug_targets: List of drug target genes/proteins
            disease_genes: List of disease-associated genes

        Returns:
            Dictionary with proximity metrics and statistical significance.
        """
        if self.current_graph is None:
            return {"error": "No knowledge graph built. Call build_knowledge_graph first."}

        result = self.current_graph.calculate_network_proximity(
            drug_targets, disease_genes, n_random=100
        )

        return {
            "drug_targets": result.drug_targets,
            "disease_genes": result.disease_genes,
            "observed_distance": round(result.observed_distance, 4) if result.observed_distance != float('inf') else None,
            "expected_distance": round(result.expected_distance, 4) if result.expected_distance != float('inf') else None,
            "z_score": round(result.z_score, 4),
            "p_value": round(result.p_value, 4),
            "is_significant": result.is_significant,
            "interpretation": result.interpretation
        }

    def predict_drug_synergy(
        self,
        target1: str,
        target2: str,
        disease_genes: list[str]
    ) -> dict[str, Any]:
        """
        Predict synergy potential for two drug targets in combination therapy.

        Based on Cheng et al. (2019) for network-based drug combination prediction.

        Args:
            target1: First drug target
            target2: Second drug target
            disease_genes: Disease-associated genes for context

        Returns:
            Dictionary with synergy scores and interpretation.
        """
        if self.current_graph is None:
            return {"error": "No knowledge graph built. Call build_knowledge_graph first."}

        result = self.current_graph.predict_synergy(target1, target2, disease_genes)

        return {
            "target1": result.target1,
            "target2": result.target2,
            "synergy_score": round(result.synergy_score, 4),
            "target_separation": result.target_separation,
            "module_coverage": round(result.module_coverage, 4),
            "complementarity": round(result.complementarity, 4),
            "interpretation": result.interpretation
        }

    def analyze_target_robustness(
        self,
        target: str,
        disease_genes: list[str]
    ) -> dict[str, Any]:
        """
        Analyze network robustness impact of targeting a specific node.

        Evaluates therapeutic potential by measuring disease impact vs
        global network disruption.

        Args:
            target: Drug target to analyze
            disease_genes: Disease-associated genes

        Returns:
            Dictionary with therapeutic index and safety assessment.
        """
        if self.current_graph is None:
            return {"error": "No knowledge graph built. Call build_knowledge_graph first."}

        result = self.current_graph.analyze_robustness(target, disease_genes)

        return {
            "target": result.target,
            "disease_impact": round(result.disease_impact, 4),
            "global_impact": round(result.global_impact, 4),
            "therapeutic_index": round(result.therapeutic_index, 4),
            "compensatory_paths": result.compensatory_paths,
            "interpretation": result.interpretation
        }

    def get_capabilities(self) -> dict[str, Any]:
        """Get a description of available capabilities."""
        return {
            "capabilities": [
                "Fetch protein interaction networks from STRING database",
                "Search PubMed for relevant biomedical literature",
                "Extract biological entities (genes, diseases, chemicals) from abstracts",
                "Extract typed relationships between entities using LLM",
                "Build evidence-backed knowledge graphs",
                "Predict novel protein-protein interactions using ML",
                "Infer relationship types for predicted interactions",
                "Query evidence for specific protein pairs",
                "Get neighborhood analysis for proteins",
                "Generate testable hypotheses for predicted interactions",
                # Graph Data Science capabilities
                "Compute centrality metrics (PageRank, betweenness, degree, closeness)",
                "Detect communities/modules using Louvain algorithm",
                "Find shortest paths between entities",
                "Run DIAMOnD disease module detection (Ghiassian et al. 2015)",
                "Calculate drug-disease network proximity (Guney et al. 2016)",
                "Predict drug synergy for combination therapy (Cheng et al. 2019)",
                "Analyze target robustness and therapeutic index",
            ],
            "data_sources": [
                "STRING (protein-protein interactions)",
                "PubMed (biomedical literature)",
                "PubTator (named entity recognition)",
            ],
            "models": [
                "Node2Vec + Logistic Regression (link prediction)",
                "Gemini (relationship extraction and inference)",
            ],
            "algorithms": [
                "PageRank, Betweenness, Degree, Closeness centrality",
                "Louvain community detection",
                "DIAMOnD disease module expansion",
                "Network proximity with z-score significance",
                "Synergy prediction with complementarity scoring",
                "Robustness analysis with therapeutic index",
            ],
        }

    # -------------------------------------------------------------------------
    # Tool Execution
    # -------------------------------------------------------------------------

    async def execute_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a tool by name with given parameters.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters

        Returns:
            Tool execution result
        """
        tool_map: dict[str, Callable[..., Any]] = {
            # Data fetching
            "get_string_network": self.get_string_network,
            "get_string_partners": self.get_string_partners,
            "search_literature": self.search_literature,
            "get_entity_annotations": self.get_entity_annotations,
            # Knowledge graph building
            "extract_relationships": self.extract_relationships,
            "build_knowledge_graph": self.build_knowledge_graph,
            # Link prediction & inference
            "predict_novel_links": self.predict_novel_links,
            "infer_novel_relationships": self.infer_novel_relationships,
            # Querying
            "query_evidence": self.query_evidence,
            "get_graph_summary": self.get_graph_summary,
            "get_protein_neighborhood": self.get_protein_neighborhood,
            "generate_hypothesis": self.generate_hypothesis,
            # Graph Data Science
            "compute_centrality": self.compute_centrality,
            "detect_communities": self.detect_communities,
            "find_path": self.find_path,
            "run_diamond_module": self.run_diamond_module,
            "calculate_drug_disease_proximity": self.calculate_drug_disease_proximity,
            "predict_drug_synergy": self.predict_drug_synergy,
            "analyze_target_robustness": self.analyze_target_robustness,
            # Meta
            "get_capabilities": self.get_capabilities,
        }

        if tool_name not in tool_map:
            return {"error": f"Unknown tool: {tool_name}"}

        tool_func = tool_map[tool_name]

        try:
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**parameters)
            else:
                result = tool_func(**parameters)

            return result
        except TypeError as e:
            return {"error": f"Invalid parameters for {tool_name}: {str(e)}"}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
