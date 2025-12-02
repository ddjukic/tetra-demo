"""
Parallel extraction pipeline for co-occurrence and LLM relationship mining.

This module runs two extraction strategies in parallel:
- Phase 4A: Fast co-occurrence extraction (Python-only, no external APIs)
- Phase 4B: LLM-based relationship extraction with semantic typing

Co-occurrence provides high-recall candidate edges while LLM extraction
adds semantic relationship types and confidence scores.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from extraction.relationship_extractor import RelationshipExtractor
from pipeline.config import PipelineConfig
from pipeline.metrics import PhaseMetrics, PipelineReport, TokenUsage

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CoOccurrenceEdge:
    """
    Represents a co-occurrence relationship between two entities.

    Co-occurrence edges are created when two entities appear in the same
    abstract. Weight is based on frequency and PMI (Pointwise Mutual Information).

    Attributes:
        source: Source entity identifier (normalized text or ID).
        target: Target entity identifier (normalized text or ID).
        weight: Co-occurrence weight (raw count or PMI score).
        pmids: List of PubMed IDs where co-occurrence was found.
        source_type: Entity type of source (Gene, Disease, Chemical, etc.).
        target_type: Entity type of target.
        pmi_score: Pointwise Mutual Information score (can be negative).
        raw_count: Raw co-occurrence count before normalization.
    """

    source: str
    target: str
    weight: float
    pmids: list[str]
    source_type: str = ""
    target_type: str = ""
    pmi_score: float = 0.0
    raw_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "pmids": self.pmids,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "pmi_score": self.pmi_score,
            "raw_count": self.raw_count,
        }


@dataclass
class ExtractionResult:
    """
    Combined results from parallel co-occurrence and LLM extraction.

    Contains results from both extraction phases along with metrics
    and error tracking for each phase.

    Attributes:
        cooccurrence_edges: List of co-occurrence edges (Phase 4A).
        llm_relationships: List of typed relationships from LLM (Phase 4B).
        cooccurrence_metrics: Phase metrics for co-occurrence extraction.
        llm_metrics: Phase metrics for LLM extraction.
        failed_extractions: List of failed extraction attempts with error info.
    """

    # Co-occurrence results (Phase 4A)
    cooccurrence_edges: list[CoOccurrenceEdge] = field(default_factory=list)

    # LLM extraction results (Phase 4B)
    llm_relationships: list[dict[str, Any]] = field(default_factory=list)

    # Metrics
    cooccurrence_metrics: PhaseMetrics | None = None
    llm_metrics: PhaseMetrics | None = None

    # Errors
    failed_extractions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_edges(self) -> int:
        """Total number of edges from both extraction methods."""
        return len(self.cooccurrence_edges) + len(self.llm_relationships)

    @property
    def unique_entities(self) -> set[str]:
        """Set of unique entities across all edges."""
        entities: set[str] = set()
        for edge in self.cooccurrence_edges:
            entities.add(edge.source)
            entities.add(edge.target)
        for rel in self.llm_relationships:
            if "entity1" in rel:
                entities.add(rel["entity1"])
            if "entity2" in rel:
                entities.add(rel["entity2"])
        return entities

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cooccurrence_edges": [e.to_dict() for e in self.cooccurrence_edges],
            "llm_relationships": self.llm_relationships,
            "cooccurrence_metrics": (
                self.cooccurrence_metrics.to_dict()
                if self.cooccurrence_metrics
                else None
            ),
            "llm_metrics": self.llm_metrics.to_dict() if self.llm_metrics else None,
            "failed_extractions": self.failed_extractions,
            "summary": {
                "total_cooccurrence_edges": len(self.cooccurrence_edges),
                "total_llm_relationships": len(self.llm_relationships),
                "unique_entities": len(self.unique_entities),
                "failed_count": len(self.failed_extractions),
            },
        }


# =============================================================================
# Co-occurrence Extraction (Phase 4A)
# =============================================================================


def _build_entity_key(entity_text: str, entity_id: str) -> str:
    """
    Build a normalized key for entity deduplication.

    Prefers entity ID when available for normalization.

    Args:
        entity_text: Raw entity text from annotation.
        entity_id: Normalized entity ID (may be empty).

    Returns:
        Normalized entity key.
    """
    if entity_id and entity_id.strip():
        return entity_id.strip()
    return entity_text.strip().lower()


def extract_cooccurrences(
    articles: list[dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
    min_count: int = 1,
) -> list[CoOccurrenceEdge]:
    """
    Extract entity co-occurrences from articles.

    For each article:
    1. Get entities from PubTator annotations
    2. Count pairs that appear in the same abstract
    3. Aggregate across all articles
    4. Calculate PMI: log2(P(a,b) / (P(a) * P(b)))

    PMI measures whether entities co-occur more often than expected by chance.
    Positive PMI indicates stronger-than-random association.

    Args:
        articles: List of article dicts with 'pmid' and 'abstract' keys.
        annotations: Dictionary mapping pmid -> list of annotation dicts.
                    Each annotation has: pmid, entity_id, entity_text, entity_type.
        min_count: Minimum co-occurrence count to include edge. Default 1.

    Returns:
        List of CoOccurrenceEdge objects sorted by weight (descending).
    """
    # Track co-occurrences: (entity1, entity2) -> {pmids, count, types}
    pair_stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"pmids": [], "count": 0, "source_type": "", "target_type": ""}
    )

    # Track entity frequencies for PMI
    entity_freq: dict[str, int] = defaultdict(int)
    total_documents = len(articles)

    if total_documents == 0:
        return []

    for article in articles:
        pmid = article.get("pmid", "")
        if not pmid:
            continue

        # Get annotations for this article
        article_annotations = annotations.get(pmid, [])
        if len(article_annotations) < 2:
            # Need at least 2 entities for co-occurrence
            continue

        # Build entity set for this document (deduplicated)
        entities_in_doc: dict[str, dict[str, str]] = {}
        for annot in article_annotations:
            entity_text = annot.get("entity_text", "")
            entity_id = annot.get("entity_id", "")
            entity_type = annot.get("entity_type", "")

            key = _build_entity_key(entity_text, entity_id)
            if key and key not in entities_in_doc:
                entities_in_doc[key] = {
                    "text": entity_text,
                    "id": entity_id,
                    "type": entity_type,
                }

        # Update entity frequencies (document frequency)
        for entity_key in entities_in_doc:
            entity_freq[entity_key] += 1

        # Generate all pairs and count co-occurrences
        entity_keys = list(entities_in_doc.keys())
        for e1_key, e2_key in combinations(sorted(entity_keys), 2):
            # Ensure consistent ordering (smaller key first)
            pair_key = (e1_key, e2_key) if e1_key < e2_key else (e2_key, e1_key)

            pair_stats[pair_key]["pmids"].append(pmid)
            pair_stats[pair_key]["count"] += 1

            # Store entity types (use first observed)
            if not pair_stats[pair_key]["source_type"]:
                e1_info = entities_in_doc[pair_key[0]]
                e2_info = entities_in_doc[pair_key[1]]
                pair_stats[pair_key]["source_type"] = e1_info["type"]
                pair_stats[pair_key]["target_type"] = e2_info["type"]

    # Calculate PMI and create edges
    edges: list[CoOccurrenceEdge] = []

    for (e1, e2), stats in pair_stats.items():
        raw_count = stats["count"]
        if raw_count < min_count:
            continue

        # Calculate PMI
        # P(a,b) = count(a,b) / total_docs
        # P(a) = count(a) / total_docs
        # P(b) = count(b) / total_docs
        # PMI = log2(P(a,b) / (P(a) * P(b)))

        p_ab = raw_count / total_documents
        p_a = entity_freq[e1] / total_documents
        p_b = entity_freq[e2] / total_documents

        # Avoid division by zero and log of zero
        if p_a > 0 and p_b > 0 and p_ab > 0:
            pmi = math.log2(p_ab / (p_a * p_b))
        else:
            pmi = 0.0

        # Weight combines raw count and PMI
        # Normalize PMI to [0, 1] range approximately with sigmoid
        pmi_normalized = 1 / (1 + math.exp(-pmi)) if abs(pmi) < 20 else (1 if pmi > 0 else 0)
        weight = raw_count * pmi_normalized

        edge = CoOccurrenceEdge(
            source=e1,
            target=e2,
            weight=weight,
            pmids=stats["pmids"],
            source_type=stats["source_type"],
            target_type=stats["target_type"],
            pmi_score=pmi,
            raw_count=raw_count,
        )
        edges.append(edge)

    # Sort by weight (descending)
    edges.sort(key=lambda e: e.weight, reverse=True)

    return edges


# =============================================================================
# LLM Relationship Extraction (Phase 4B)
# =============================================================================


def _build_entity_pairs_for_article(
    pmid: str,
    annotations: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """
    Build entity pairs for relationship extraction from annotations.

    Args:
        pmid: PubMed ID of the article.
        annotations: List of annotations for this article.

    Returns:
        List of (entity1, entity2) tuples for relationship classification.
    """
    # Deduplicate entities
    entities: dict[str, str] = {}  # key -> entity_text
    for annot in annotations:
        entity_text = annot.get("entity_text", "")
        entity_id = annot.get("entity_id", "")
        key = _build_entity_key(entity_text, entity_id)
        if key and key not in entities:
            entities[key] = entity_text

    # Generate pairs
    entity_texts = list(entities.values())
    pairs: list[tuple[str, str]] = []
    for i, e1 in enumerate(entity_texts):
        for e2 in entity_texts[i + 1 :]:
            pairs.append((e1, e2))

    return pairs


async def _extract_with_retry(
    article: dict[str, Any],
    annotations: list[dict[str, Any]],
    extractor: RelationshipExtractor,
    config: PipelineConfig,
    semaphore: asyncio.Semaphore,
) -> tuple[list[dict[str, Any]], TokenUsage | None, str | None]:
    """
    Extract relationships from a single article with retry logic.

    Uses exponential backoff for transient failures (429, 5xx).
    Respects semaphore for rate limiting.

    Args:
        article: Article dict with pmid and abstract.
        annotations: List of annotations for this article.
        extractor: RelationshipExtractor instance.
        config: Pipeline configuration with retry settings.
        semaphore: Asyncio semaphore for concurrency control.

    Returns:
        Tuple of (relationships, token_usage, error_message).
        error_message is None on success.
    """
    pmid = article.get("pmid", "unknown")
    abstract = article.get("abstract", "")

    if not abstract:
        return [], None, f"No abstract for PMID {pmid}"

    # Build entity pairs from annotations
    entity_pairs = _build_entity_pairs_for_article(pmid, annotations)
    if not entity_pairs:
        return [], None, None  # No pairs, not an error

    async with semaphore:
        last_error: str | None = None

        for attempt in range(config.mining_max_retries):
            try:
                start_time = time.time()

                # Call the extractor
                relationships = await extractor.extract_relationships(
                    abstract=abstract,
                    entity_pairs=entity_pairs,
                    pmid=pmid,
                )

                latency_ms = (time.time() - start_time) * 1000

                # Create token usage (approximate since we don't have direct access)
                # The extractor doesn't expose token counts, so we estimate
                token_usage = TokenUsage.create(
                    phase="relationship_mining",
                    step=f"pmid_{pmid}",
                    prompt_tokens=len(abstract) // 4,  # Rough estimate
                    completion_tokens=len(str(relationships)) // 4,
                    latency_ms=latency_ms,
                    model=config.mining_model,
                )

                return relationships, token_usage, None

            except Exception as e:
                last_error = str(e)
                error_str = str(e).lower()

                # Check if error is retryable
                is_rate_limit = "429" in error_str or "rate" in error_str
                is_server_error = any(
                    str(code) in error_str for code in config.mining_retry_on_codes
                )

                if (is_rate_limit or is_server_error) and attempt < config.mining_max_retries - 1:
                    delay = config.mining_base_delay * (2**attempt)
                    logger.warning(
                        f"Retrying PMID {pmid} after {delay}s (attempt {attempt + 1}): {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    # Non-retryable error or max retries exceeded
                    logger.error(f"Failed extraction for PMID {pmid}: {e}")
                    break

        return [], None, last_error


async def extract_relationships_parallel(
    articles: list[dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
    config: PipelineConfig,
    report: PipelineReport,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extract relationships using LLM with parallel execution.

    Uses existing RelationshipExtractor from extraction/relationship_extractor.py.
    Applies semaphore for rate limiting based on config.mining_max_concurrent.
    Implements retry with exponential backoff on failures.

    Args:
        articles: List of articles with pmid and abstract.
        annotations: Dictionary mapping pmid -> list of annotations.
        config: Pipeline configuration.
        report: Pipeline report for metrics tracking.

    Returns:
        Tuple of (relationships, failures).
        - relationships: List of extracted relationship dicts.
        - failures: List of failed extraction info dicts.
    """
    if not articles:
        return [], []

    # Create extractor with configured model
    extractor = RelationshipExtractor(model=config.mining_model)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(config.mining_max_concurrent)

    # Create tasks for all articles
    tasks = []
    for article in articles:
        pmid = article.get("pmid", "")
        article_annotations = annotations.get(pmid, [])
        tasks.append(
            _extract_with_retry(
                article=article,
                annotations=article_annotations,
                extractor=extractor,
                config=config,
                semaphore=semaphore,
            )
        )

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results and failures
    all_relationships: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for i, result in enumerate(results):
        pmid = articles[i].get("pmid", "unknown")

        if isinstance(result, Exception):
            failures.append({
                "pmid": pmid,
                "error": str(result),
                "error_type": type(result).__name__,
            })
            report.get_phase("relationship_mining")
            phase = report.phase_metrics.get("relationship_mining")
            if phase:
                phase.add_error(f"PMID {pmid}: {result}")

        elif isinstance(result, tuple):
            relationships, token_usage, error = result

            if error:
                failures.append({
                    "pmid": pmid,
                    "error": error,
                })
                phase = report.phase_metrics.get("relationship_mining")
                if phase:
                    phase.add_error(f"PMID {pmid}: {error}")

            if relationships:
                all_relationships.extend(relationships)

            if token_usage:
                report.add_token_usage("relationship_mining", token_usage)

    return all_relationships, failures


# =============================================================================
# Parallel Orchestration
# =============================================================================


async def run_parallel_extraction(
    articles: list[dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
    config: PipelineConfig,
    report: PipelineReport,
) -> ExtractionResult:
    """
    Run co-occurrence and LLM extraction in parallel.

    Phase 4A (fast, Python-only):
    - Count entity co-mentions per abstract
    - Weight by frequency
    - Calculate PMI score

    Phase 4B (LLM, parallel with semaphore):
    - Extract typed relationships from abstracts
    - Use asyncio.Semaphore for rate limiting
    - Retry with exponential backoff on 429/5xx

    Both phases run concurrently to minimize total wall-clock time.

    Args:
        articles: List of article dicts from PubMed with abstracts.
        annotations: Dictionary mapping pmid -> list of annotations from PubTator.
        config: Pipeline configuration.
        report: Pipeline report for metrics tracking.

    Returns:
        ExtractionResult with edges from both extraction methods.
    """
    result = ExtractionResult()

    # Start phase tracking
    cooccurrence_phase = report.start_phase("cooccurrence_extraction")
    llm_phase = report.start_phase("relationship_mining")

    logger.info(
        f"Starting parallel extraction: {len(articles)} articles, "
        f"{sum(len(v) for v in annotations.values())} annotations"
    )

    # Run co-occurrence extraction (synchronous but fast)
    async def run_cooccurrence() -> list[CoOccurrenceEdge]:
        """Run co-occurrence extraction in executor to not block."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: extract_cooccurrences(articles, annotations),
        )

    # Run both extractions in parallel
    cooccurrence_task = asyncio.create_task(run_cooccurrence())
    llm_task = asyncio.create_task(
        extract_relationships_parallel(articles, annotations, config, report)
    )

    # Wait for both to complete
    cooccurrence_edges, (llm_relationships, failures) = await asyncio.gather(
        cooccurrence_task,
        llm_task,
    )

    # Complete phase tracking
    cooccurrence_phase.complete()
    cooccurrence_phase.items_processed = len(articles)

    llm_phase.complete()
    llm_phase.items_processed = len(articles)
    llm_phase.api_calls = len(articles)

    # Populate result
    result.cooccurrence_edges = cooccurrence_edges
    result.llm_relationships = llm_relationships
    result.failed_extractions = failures
    result.cooccurrence_metrics = cooccurrence_phase
    result.llm_metrics = llm_phase

    logger.info(
        f"Parallel extraction complete: "
        f"{len(cooccurrence_edges)} co-occurrence edges, "
        f"{len(llm_relationships)} LLM relationships, "
        f"{len(failures)} failures"
    )

    # Update report summary
    report.relationships_extracted = len(llm_relationships)
    report.edges_created += len(cooccurrence_edges) + len(llm_relationships)

    return result


# =============================================================================
# Utility Functions
# =============================================================================


def merge_cooccurrence_and_llm_edges(
    cooccurrence_edges: list[CoOccurrenceEdge],
    llm_relationships: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge co-occurrence edges with LLM relationships.

    For edges that appear in both:
    - Use LLM relationship type and confidence
    - Add co-occurrence weight and PMI as additional evidence

    For co-occurrence-only edges:
    - Use relationship type "cooccurs_with"
    - Set confidence based on PMI

    For LLM-only edges:
    - Use as-is

    Args:
        cooccurrence_edges: Edges from co-occurrence extraction.
        llm_relationships: Relationships from LLM extraction.

    Returns:
        Merged list of relationship dicts.
    """
    # Build lookup for LLM relationships
    llm_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for rel in llm_relationships:
        e1 = rel.get("entity1", "").strip().lower()
        e2 = rel.get("entity2", "").strip().lower()
        key = (min(e1, e2), max(e1, e2))
        llm_lookup[key] = rel

    merged: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    # Process co-occurrence edges
    for edge in cooccurrence_edges:
        e1 = edge.source.lower()
        e2 = edge.target.lower()
        key = (min(e1, e2), max(e1, e2))
        seen_keys.add(key)

        if key in llm_lookup:
            # Merge with LLM data
            llm_rel = llm_lookup[key]
            merged_rel = {
                **llm_rel,
                "cooccurrence_weight": edge.weight,
                "pmi_score": edge.pmi_score,
                "raw_count": edge.raw_count,
                "evidence_pmids": edge.pmids,
            }
            merged.append(merged_rel)
        else:
            # Co-occurrence only
            merged_rel = {
                "entity1": edge.source,
                "entity2": edge.target,
                "relationship": "cooccurs_with",
                "confidence": min(0.9, 0.3 + 0.1 * edge.raw_count),
                "cooccurrence_weight": edge.weight,
                "pmi_score": edge.pmi_score,
                "raw_count": edge.raw_count,
                "pmids": edge.pmids,
                "source_type": edge.source_type,
                "target_type": edge.target_type,
            }
            merged.append(merged_rel)

    # Add LLM-only relationships
    for rel in llm_relationships:
        e1 = rel.get("entity1", "").strip().lower()
        e2 = rel.get("entity2", "").strip().lower()
        key = (min(e1, e2), max(e1, e2))

        if key not in seen_keys:
            merged.append(rel)
            seen_keys.add(key)

    return merged


def annotations_list_to_dict(
    annotations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Convert flat annotation list to pmid-keyed dictionary.

    Helper function for when annotations are returned as a flat list
    from PubTator API.

    Args:
        annotations: List of annotation dicts with 'pmid' key.

    Returns:
        Dictionary mapping pmid -> list of annotations.
    """
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annot in annotations:
        pmid = annot.get("pmid", "")
        if pmid:
            result[pmid].append(annot)
    return dict(result)
