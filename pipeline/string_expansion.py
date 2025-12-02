"""
STRING database network expansion for protein-protein interactions.

This module orchestrates the first phase of the knowledge graph pipeline:
expanding a set of seed proteins into a broader interaction network using
the STRING database API.

Phase 1 in the pipeline flow:
    Seed Proteins -> STRING Expansion -> Expanded Network -> PubMed Query

Example:
    >>> from pipeline import PipelineConfig, PipelineReport
    >>> from pipeline.string_expansion import expand_string_network
    >>>
    >>> config = PipelineConfig()
    >>> report = PipelineReport.create()
    >>> result = await expand_string_network(
    ...     seed_proteins=["BRCA1", "TP53", "PTEN"],
    ...     config=config,
    ...     report=report,
    ... )
    >>> print(f"Expanded from {len(result.seed_proteins)} to {len(result.expanded_proteins)} proteins")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from clients.string_client import StringClient
from pipeline.config import PipelineConfig
from pipeline.metrics import PhaseMetrics, PipelineReport

logger = logging.getLogger(__name__)


# =============================================================================
# Result Data Structures
# =============================================================================


@dataclass
class STRINGExpansionResult:
    """
    Result of STRING network expansion.

    Contains the expanded protein network including original seeds,
    discovered interaction partners, and all interaction edges.

    Attributes:
        seed_proteins: Original list of seed proteins that were queried.
        expanded_proteins: All proteins in the expanded network (seeds + partners).
        interactions: List of STRING interaction records with scores.
        proteins_not_found: Seeds that could not be resolved in STRING.
        protein_annotations: Mapping of protein names to STRING annotations.
        phase_metrics: Metrics collected during this phase.
    """

    seed_proteins: list[str]
    expanded_proteins: list[str]
    interactions: list[dict]
    proteins_not_found: list[str]
    protein_annotations: dict[str, str] = field(default_factory=dict)
    phase_metrics: PhaseMetrics | None = None

    @property
    def partner_proteins(self) -> list[str]:
        """
        Get only the newly discovered partner proteins (excluding seeds).

        Returns:
            List of protein names that were discovered through network expansion.
        """
        seed_set = set(self.seed_proteins)
        return [p for p in self.expanded_proteins if p not in seed_set]

    @property
    def interaction_count(self) -> int:
        """Get the total number of interactions found."""
        return len(self.interactions)

    @property
    def expansion_ratio(self) -> float:
        """
        Calculate the network expansion ratio.

        Returns:
            Ratio of expanded network size to seed count.
        """
        if not self.seed_proteins:
            return 0.0
        return len(self.expanded_proteins) / len(self.seed_proteins)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "seed_proteins": self.seed_proteins,
            "expanded_proteins": self.expanded_proteins,
            "interactions": self.interactions,
            "proteins_not_found": self.proteins_not_found,
            "protein_annotations": self.protein_annotations,
            "interaction_count": self.interaction_count,
            "expansion_ratio": self.expansion_ratio,
        }


# =============================================================================
# Core Expansion Function
# =============================================================================


async def expand_string_network(
    seed_proteins: list[str],
    config: PipelineConfig,
    report: PipelineReport,
) -> STRINGExpansionResult:
    """
    Expand a protein network using STRING database interactions.

    This is Phase 1 of the knowledge graph pipeline. It takes a set of seed
    proteins and expands them into a broader interaction network by:

    1. Resolving protein identifiers to validate they exist in STRING
    2. Fetching interaction partners for each seed protein
    3. Collecting all interactions between proteins in the expanded network
    4. Recording phase metrics for observability

    Args:
        seed_proteins: List of protein gene symbols to expand (e.g., ["BRCA1", "TP53"]).
        config: Pipeline configuration with STRING parameters.
        report: Pipeline report for metrics tracking.

    Returns:
        STRINGExpansionResult containing:
            - seed_proteins: Original seeds that were found in STRING
            - expanded_proteins: All proteins (seeds + interaction partners)
            - interactions: List of interaction dicts with scores
            - proteins_not_found: Seeds that couldn't be resolved
            - phase_metrics: Metrics for this phase

    Raises:
        ValueError: If seed_proteins is empty.

    Example:
        >>> result = await expand_string_network(
        ...     ["BRCA1", "TP53"],
        ...     config=PipelineConfig(),
        ...     report=PipelineReport.create(),
        ... )
        >>> for interaction in result.interactions[:3]:
        ...     print(f"{interaction['preferredName_A']} <-> {interaction['preferredName_B']}")
    """
    if not seed_proteins:
        raise ValueError("seed_proteins cannot be empty")

    # Start phase metrics
    phase_name = "string_expansion"
    metrics = report.start_phase(phase_name)

    logger.info(
        "Starting STRING network expansion for %d seed proteins: %s",
        len(seed_proteins),
        ", ".join(seed_proteins[:5]) + ("..." if len(seed_proteins) > 5 else ""),
    )

    # Initialize result containers
    resolved_seeds: list[str] = []
    proteins_not_found: list[str] = []
    protein_annotations: dict[str, str] = {}
    all_interactions: list[dict] = []
    expanded_protein_set: set[str] = set()

    async with StringClient() as client:
        # ---------------------------------------------------------------------
        # Step 1: Resolve protein identifiers
        # ---------------------------------------------------------------------
        logger.debug("Resolving %d protein identifiers", len(seed_proteins))
        metrics.increment_api_calls()

        resolved = await client.resolve_identifiers(seed_proteins)

        # Build mapping of resolved proteins
        resolved_queries = set()
        for res in resolved:
            query_item = res.get("queryItem", "")
            preferred_name = res.get("preferredName", query_item)
            annotation = res.get("annotation", "")

            resolved_queries.add(query_item.upper())
            resolved_seeds.append(preferred_name)
            protein_annotations[preferred_name] = annotation
            expanded_protein_set.add(preferred_name)

        # Identify proteins that weren't found
        for protein in seed_proteins:
            if protein.upper() not in resolved_queries:
                proteins_not_found.append(protein)
                logger.warning("Protein not found in STRING: %s", protein)

        if proteins_not_found:
            metrics.add_error(
                f"{len(proteins_not_found)} proteins not found: {', '.join(proteins_not_found)}"
            )

        logger.info(
            "Resolved %d/%d proteins in STRING",
            len(resolved_seeds),
            len(seed_proteins),
        )

        # Exit early if no proteins were resolved
        if not resolved_seeds:
            metrics.complete()
            return STRINGExpansionResult(
                seed_proteins=[],
                expanded_proteins=[],
                interactions=[],
                proteins_not_found=proteins_not_found,
                protein_annotations={},
                phase_metrics=metrics,
            )

        # ---------------------------------------------------------------------
        # Step 2: Get interaction partners for network expansion
        # ---------------------------------------------------------------------
        logger.debug(
            "Fetching interaction partners (limit=%d, min_score=%d)",
            config.string_max_partners,
            config.string_min_score,
        )
        metrics.increment_api_calls()

        partners = await client.get_interaction_partners(
            proteins=resolved_seeds,
            limit=config.string_max_partners,
            min_score=config.string_min_score,
        )

        # Extract partner proteins and track interactions
        for interaction in partners:
            name_a = interaction.get("preferredName_A", "")
            name_b = interaction.get("preferredName_B", "")

            if name_a:
                expanded_protein_set.add(name_a)
            if name_b:
                expanded_protein_set.add(name_b)

            all_interactions.append(interaction)

        logger.info(
            "Found %d interaction partners, %d total interactions",
            len(expanded_protein_set) - len(resolved_seeds),
            len(partners),
        )

        # ---------------------------------------------------------------------
        # Step 3: Get full network between all expanded proteins
        # ---------------------------------------------------------------------
        if len(expanded_protein_set) > len(resolved_seeds):
            # Get interactions within the expanded network
            expanded_list = list(expanded_protein_set)
            logger.debug(
                "Fetching full network for %d proteins", len(expanded_list)
            )
            metrics.increment_api_calls()

            network = await client.get_network(
                proteins=expanded_list,
                min_score=config.string_min_score,
                network_type="functional",
            )

            # Merge network interactions (avoid duplicates)
            existing_pairs = set()
            for interaction in all_interactions:
                name_a = interaction.get("preferredName_A", "")
                name_b = interaction.get("preferredName_B", "")
                pair = tuple(sorted([name_a, name_b]))
                existing_pairs.add(pair)

            for interaction in network:
                name_a = interaction.get("preferredName_A", "")
                name_b = interaction.get("preferredName_B", "")
                pair = tuple(sorted([name_a, name_b]))

                if pair not in existing_pairs:
                    all_interactions.append(interaction)
                    existing_pairs.add(pair)

            logger.info(
                "Full network: %d total interactions after deduplication",
                len(all_interactions),
            )

    # Finalize metrics
    metrics.increment_items_processed(len(expanded_protein_set))
    metrics.complete()
    report.end_phase(phase_name)

    # Sort expanded proteins (seeds first, then partners alphabetically)
    seed_set = set(resolved_seeds)
    expanded_proteins = resolved_seeds + sorted(
        [p for p in expanded_protein_set if p not in seed_set]
    )

    result = STRINGExpansionResult(
        seed_proteins=resolved_seeds,
        expanded_proteins=expanded_proteins,
        interactions=all_interactions,
        proteins_not_found=proteins_not_found,
        protein_annotations=protein_annotations,
        phase_metrics=metrics,
    )

    logger.info(
        "STRING expansion complete: %d seeds -> %d proteins, %d interactions",
        len(resolved_seeds),
        len(expanded_proteins),
        len(all_interactions),
    )

    return result


# =============================================================================
# Utility Functions
# =============================================================================


def extract_high_confidence_partners(
    result: STRINGExpansionResult,
    min_score: float = 0.9,
) -> list[str]:
    """
    Extract partner proteins with high-confidence interactions.

    Filters the expanded network to include only proteins connected
    to seeds with very high STRING scores (indicating strong evidence).

    Args:
        result: STRINGExpansionResult from expand_string_network.
        min_score: Minimum combined score (0-1). Default 0.9.

    Returns:
        List of partner protein names with high-confidence connections.
    """
    seed_set = set(result.seed_proteins)
    high_confidence_partners = set()

    for interaction in result.interactions:
        score = interaction.get("score", 0)
        if score >= min_score:
            name_a = interaction.get("preferredName_A", "")
            name_b = interaction.get("preferredName_B", "")

            # If one is a seed, add the other as high-confidence partner
            if name_a in seed_set and name_b not in seed_set:
                high_confidence_partners.add(name_b)
            elif name_b in seed_set and name_a not in seed_set:
                high_confidence_partners.add(name_a)

    return sorted(high_confidence_partners)


def get_interaction_summary(result: STRINGExpansionResult) -> dict:
    """
    Generate a summary of the interaction network.

    Args:
        result: STRINGExpansionResult from expand_string_network.

    Returns:
        Dictionary with network statistics.
    """
    if not result.interactions:
        return {
            "total_interactions": 0,
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "high_confidence_count": 0,
        }

    scores = [i.get("score", 0) for i in result.interactions]
    high_confidence = [s for s in scores if s >= 0.7]

    return {
        "total_interactions": len(result.interactions),
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "high_confidence_count": len(high_confidence),
        "seeds_count": len(result.seed_proteins),
        "expanded_count": len(result.expanded_proteins),
        "not_found_count": len(result.proteins_not_found),
    }
