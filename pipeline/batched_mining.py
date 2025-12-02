"""
Batched Relationship Mining with Intelligent Chunking.

This module provides an orchestrator that batches abstracts by token count
and mines relationships in parallel using structured output.

Architecture:
    BatchedMiningOrchestrator
        |
        +-- analyze_abstracts() -> token counts, chunking strategy
        |
        +-- chunk_abstracts(target_tokens=5000) -> list[AbstractChunk]
        |
        +-- mine_chunk(chunk) -> ChunkMiningResult
                +-- Uses structured output schema
                +-- Includes evidence sentences
                +-- Preserves PMID metadata

Key Features:
- Token-aware chunking for optimal batching (~5K tokens per batch)
- Aggressive parallelization (min 3 chunks for concurrency)
- Structured JSON output with response schema
- Exponential backoff retry on rate limits (429, 503)
- Evidence sentences preserved verbatim from abstracts
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import google.generativeai as genai

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AbstractChunk:
    """A batch of abstracts for mining."""

    chunk_id: int
    pmids: list[str]
    abstracts: list[dict[str, Any]]  # {pmid, title, abstract, year}
    entities: list[str]  # Unique entities in this chunk
    total_tokens: int


@dataclass
class ExtractedRelationship:
    """A single extracted relationship with evidence."""

    source_entity: str
    target_entity: str
    relation_type: str  # activates, inhibits, associated_with, etc.
    evidence_sentence: str  # The exact sentence supporting this
    confidence: float  # 0-1, model's confidence
    pmid: str  # Source paper

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relation_type": self.relation_type,
            "evidence_sentence": self.evidence_sentence,
            "confidence": self.confidence,
            "pmid": self.pmid,
        }


@dataclass
class ChunkMiningResult:
    """Results from mining a single chunk."""

    chunk_id: int
    relationships: list[ExtractedRelationship]
    pmids_processed: list[str]
    token_usage: dict[str, int]
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if mining was successful."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "relationships": [r.to_dict() for r in self.relationships],
            "pmids_processed": self.pmids_processed,
            "token_usage": self.token_usage,
            "errors": self.errors,
            "success": self.success,
        }


@dataclass
class BatchedMiningConfig:
    """Configuration for batched mining."""

    target_tokens_per_chunk: int = 5000
    min_chunks: int = 3  # Always parallelize at least this much
    max_concurrent: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.1
    min_confidence: float = 0.5

    # Relationship types to extract
    relationship_types: list[str] = field(
        default_factory=lambda: [
            "activates",
            "inhibits",
            "associated_with",
            "regulates",
            "binds_to",
            "treats",
            "causes",
            "interacts_with",
            "component_of",
            "produces",
        ]
    )


# =============================================================================
# Structured Output Schema
# =============================================================================

RELATIONSHIP_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "source_entity": {
                "type": "string",
                "description": "The source entity in the relationship",
            },
            "target_entity": {
                "type": "string",
                "description": "The target entity in the relationship",
            },
            "relation_type": {
                "type": "string",
                "description": "Type of relationship: activates, inhibits, associated_with, regulates, binds_to, treats, causes, interacts_with, component_of, produces",
            },
            "evidence_sentence": {
                "type": "string",
                "description": "The exact sentence from the abstract supporting this relationship",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score from 0.0 to 1.0",
            },
            "pmid": {
                "type": "string",
                "description": "The PubMed ID of the source paper",
            },
        },
        "required": [
            "source_entity",
            "target_entity",
            "relation_type",
            "evidence_sentence",
            "pmid",
        ],
    },
}


# =============================================================================
# Token Counting
# =============================================================================


def count_tokens_simple(text: str) -> int:
    """
    Simple token count approximation.

    Uses ~4 characters per token as approximation (typical for English).
    For production, use tiktoken with cl100k_base encoding.

    Args:
        text: Text to count tokens for.

    Returns:
        Approximate token count.
    """
    return len(text) // 4


def try_count_tokens_tiktoken(text: str) -> int:
    """
    Count tokens using tiktoken if available.

    Falls back to simple approximation if tiktoken not installed.

    Args:
        text: Text to count tokens for.

    Returns:
        Token count.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return count_tokens_simple(text)


# =============================================================================
# Prompt Builder
# =============================================================================


def build_extraction_prompt(chunk: AbstractChunk, config: BatchedMiningConfig) -> str:
    """
    Build the relationship extraction prompt for a chunk.

    Args:
        chunk: AbstractChunk containing abstracts and entities.
        config: Mining configuration.

    Returns:
        Formatted prompt string.
    """
    relationship_types_str = ", ".join(config.relationship_types)

    # Format abstracts with PMID markers
    abstracts_text = []
    for article in chunk.abstracts:
        pmid = article.get("pmid", "unknown")
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        year = article.get("year", "")

        abstract_block = f"""
[PMID: {pmid}] ({year})
Title: {title}
Abstract: {abstract}
"""
        abstracts_text.append(abstract_block)

    abstracts_formatted = "\n---\n".join(abstracts_text)

    # Format entities list
    entities_formatted = ", ".join(sorted(set(chunk.entities)))

    prompt = f"""You are a biomedical relationship extraction expert.

## TASK
Extract semantic relationships between biomedical entities from the following PubMed abstracts.

## ENTITIES TO LOOK FOR
{entities_formatted}

## RELATIONSHIP TYPES
{relationship_types_str}

## ABSTRACTS TO ANALYZE
{abstracts_formatted}

## OUTPUT REQUIREMENTS
For each relationship found:
1. source_entity: The source entity (must match one from the entities list or be clearly mentioned)
2. target_entity: The target entity (must match one from the entities list or be clearly mentioned)
3. relation_type: One of: {relationship_types_str}
4. evidence_sentence: The EXACT sentence from the abstract that supports this relationship (verbatim quote)
5. confidence: 0.0-1.0 based on evidence strength
   - 0.9+: Explicit causal statement (e.g., "X activates Y", "X inhibits Y")
   - 0.7-0.9: Clear implication or strong association
   - 0.5-0.7: Weak or indirect evidence
6. pmid: The PMID of the paper where this relationship was found

## RULES
1. Only extract relationships EXPLICITLY stated or strongly implied in the abstracts
2. Evidence must be VERBATIM quotes from the text
3. Use directional relationships (source -> relation -> target)
4. Do not invent relationships not supported by the text
5. Include the PMID for each relationship
6. If no relationships found, return an empty array []

Return a JSON array of relationship objects."""

    return prompt


# =============================================================================
# BatchedMiningOrchestrator
# =============================================================================


class BatchedMiningOrchestrator:
    """
    Orchestrates batched relationship mining with intelligent chunking.

    This orchestrator:
    1. Analyzes abstracts and their token counts
    2. Chunks abstracts by token count for optimal batching
    3. Spawns parallel mining agents with structured output
    4. Handles retries via exponential backoff
    """

    def __init__(self, config: BatchedMiningConfig | None = None):
        """
        Initialize the orchestrator.

        Args:
            config: Mining configuration. Uses defaults if not provided.
        """
        self.config = config or BatchedMiningConfig()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.config.model)

    def analyze_abstracts(self, articles: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Analyze abstracts and return token statistics.

        Args:
            articles: List of article dicts with 'pmid' and 'abstract' keys.

        Returns:
            Dictionary with token statistics and chunking recommendations.
        """
        if not articles:
            return {
                "total_abstracts": 0,
                "total_tokens": 0,
                "mean_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "recommended_chunks": 0,
                "tokens_per_chunk": 0,
            }

        token_counts = []
        for article in articles:
            abstract = article.get("abstract", "")
            title = article.get("title", "")
            text = f"{title} {abstract}"
            tokens = try_count_tokens_tiktoken(text)
            token_counts.append(tokens)

        total_tokens = sum(token_counts)
        mean_tokens = total_tokens / len(token_counts)

        # Calculate recommended chunks
        recommended_chunks = max(
            self.config.min_chunks,
            (total_tokens + self.config.target_tokens_per_chunk - 1)
            // self.config.target_tokens_per_chunk,
        )
        tokens_per_chunk = total_tokens / recommended_chunks if recommended_chunks > 0 else 0

        return {
            "total_abstracts": len(articles),
            "total_tokens": total_tokens,
            "mean_tokens": round(mean_tokens, 1),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "recommended_chunks": recommended_chunks,
            "tokens_per_chunk": round(tokens_per_chunk, 1),
        }

    def chunk_abstracts(
        self,
        articles: list[dict[str, Any]],
        annotations: dict[str, list[dict[str, Any]]],
    ) -> list[AbstractChunk]:
        """
        Chunk abstracts by token count.

        Algorithm:
        1. Count tokens for each abstract
        2. Greedily fill chunks up to target_tokens
        3. Ensure at least min_chunks for parallelization
        4. Attach relevant entities to each chunk

        Args:
            articles: List of article dicts with 'pmid', 'title', 'abstract'.
            annotations: Dictionary mapping pmid -> list of annotation dicts.

        Returns:
            List of AbstractChunk objects.
        """
        if not articles:
            return []

        # Calculate token counts for each article
        article_tokens = []
        for article in articles:
            abstract = article.get("abstract", "")
            title = article.get("title", "")
            text = f"{title} {abstract}"
            tokens = try_count_tokens_tiktoken(text)
            article_tokens.append((article, tokens))

        # Greedily build chunks
        chunks: list[AbstractChunk] = []
        current_articles: list[dict[str, Any]] = []
        current_tokens = 0
        chunk_id = 0

        for article, tokens in article_tokens:
            # Check if adding this article would exceed target
            if (
                current_tokens + tokens > self.config.target_tokens_per_chunk
                and current_articles
            ):
                # Finalize current chunk
                chunk = self._create_chunk(chunk_id, current_articles, annotations, current_tokens)
                chunks.append(chunk)
                chunk_id += 1
                current_articles = []
                current_tokens = 0

            current_articles.append(article)
            current_tokens += tokens

        # Add remaining articles
        if current_articles:
            chunk = self._create_chunk(chunk_id, current_articles, annotations, current_tokens)
            chunks.append(chunk)

        # Ensure minimum chunks for parallelization
        while len(chunks) < self.config.min_chunks:
            # Split the largest chunk
            largest_idx = max(range(len(chunks)), key=lambda i: len(chunks[i].abstracts))
            largest = chunks[largest_idx]

            # Can't split if largest chunk has fewer than 2 abstracts
            if len(largest.abstracts) < 2:
                break

            # Split in half
            mid = len(largest.abstracts) // 2
            first_half = largest.abstracts[:mid]
            second_half = largest.abstracts[mid:]

            # Create new chunks
            chunks[largest_idx] = self._create_chunk(
                largest.chunk_id,
                first_half,
                annotations,
                sum(try_count_tokens_tiktoken(a.get("abstract", "") + a.get("title", "")) for a in first_half),
            )
            new_chunk = self._create_chunk(
                len(chunks),
                second_half,
                annotations,
                sum(try_count_tokens_tiktoken(a.get("abstract", "") + a.get("title", "")) for a in second_half),
            )
            chunks.append(new_chunk)

        # Renumber chunk IDs
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = i

        return chunks

    def _create_chunk(
        self,
        chunk_id: int,
        articles: list[dict[str, Any]],
        annotations: dict[str, list[dict[str, Any]]],
        total_tokens: int,
    ) -> AbstractChunk:
        """
        Create an AbstractChunk from articles.

        Args:
            chunk_id: Chunk identifier.
            articles: List of articles for this chunk.
            annotations: All annotations dictionary.
            total_tokens: Total token count for this chunk.

        Returns:
            AbstractChunk object.
        """
        pmids = [a.get("pmid", "") for a in articles]

        # Collect unique entities for this chunk
        entities: set[str] = set()
        for pmid in pmids:
            for annot in annotations.get(pmid, []):
                entity_text = annot.get("entity_text", "")
                if entity_text:
                    entities.add(entity_text)

        return AbstractChunk(
            chunk_id=chunk_id,
            pmids=pmids,
            abstracts=articles,
            entities=sorted(entities),
            total_tokens=total_tokens,
        )

    async def mine_chunk(
        self,
        chunk: AbstractChunk,
        semaphore: asyncio.Semaphore,
    ) -> ChunkMiningResult:
        """
        Mine relationships from a single chunk using structured output.

        Uses retry logic on rate limit errors (429, 503).

        Args:
            chunk: AbstractChunk to mine.
            semaphore: Asyncio semaphore for concurrency control.

        Returns:
            ChunkMiningResult with extracted relationships.
        """
        async with semaphore:
            last_error: str | None = None

            for attempt in range(self.config.max_retries):
                try:
                    result = await self._extract_relationships(chunk)
                    return result

                except Exception as e:
                    last_error = str(e)
                    error_str = str(e).lower()

                    # Check if retryable
                    is_rate_limit = "429" in error_str or "rate" in error_str
                    is_server_error = any(
                        str(code) in error_str for code in [503, 500, 502]
                    )

                    if (is_rate_limit or is_server_error) and attempt < self.config.max_retries - 1:
                        delay = self.config.retry_delay * (2**attempt)
                        logger.warning(
                            f"Chunk {chunk.chunk_id} attempt {attempt + 1} failed, "
                            f"retrying in {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Chunk {chunk.chunk_id} failed after {attempt + 1} attempts: {e}"
                        )
                        break

            # Return error result
            return ChunkMiningResult(
                chunk_id=chunk.chunk_id,
                relationships=[],
                pmids_processed=chunk.pmids,
                token_usage={},
                errors=[last_error or "Unknown error"],
            )

    async def _extract_relationships(self, chunk: AbstractChunk) -> ChunkMiningResult:
        """
        Extract relationships using structured prompting.

        Args:
            chunk: AbstractChunk to process.

        Returns:
            ChunkMiningResult with extracted relationships.
        """
        prompt = build_extraction_prompt(chunk, self.config)

        start_time = time.time()

        # Use asyncio to run the synchronous Gemini API
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.config.temperature,
                    response_mime_type="application/json",
                    response_schema=RELATIONSHIP_SCHEMA,
                ),
            ),
        )

        latency_ms = (time.time() - start_time) * 1000

        # Parse response
        try:
            result_text = response.text or "[]"
            result_data = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Chunk {chunk.chunk_id} JSON parse error: {e}")
            result_data = []

        # Ensure result is a list
        if isinstance(result_data, dict):
            result_data = result_data.get("relationships", [])
        elif not isinstance(result_data, list):
            result_data = []

        # Convert to ExtractedRelationship objects
        relationships: list[ExtractedRelationship] = []
        for rel in result_data:
            confidence = rel.get("confidence", 0.5)
            if confidence < self.config.min_confidence:
                continue

            relationships.append(
                ExtractedRelationship(
                    source_entity=rel.get("source_entity", ""),
                    target_entity=rel.get("target_entity", ""),
                    relation_type=rel.get("relation_type", "").lower(),
                    evidence_sentence=rel.get("evidence_sentence", ""),
                    confidence=confidence,
                    pmid=rel.get("pmid", ""),
                )
            )

        # Token usage estimation (Gemini doesn't always expose this)
        token_usage = {
            "prompt_tokens": try_count_tokens_tiktoken(prompt),
            "completion_tokens": try_count_tokens_tiktoken(result_text) if result_text else 0,
            "latency_ms": round(latency_ms),
        }

        logger.info(
            f"Chunk {chunk.chunk_id}: extracted {len(relationships)} relationships "
            f"from {len(chunk.pmids)} abstracts ({latency_ms:.0f}ms)"
        )

        return ChunkMiningResult(
            chunk_id=chunk.chunk_id,
            relationships=relationships,
            pmids_processed=chunk.pmids,
            token_usage=token_usage,
            errors=[],
        )

    async def run(
        self,
        articles: list[dict[str, Any]],
        annotations: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Run the full batched mining pipeline.

        Args:
            articles: List of article dicts with 'pmid', 'title', 'abstract', 'year'.
            annotations: Dictionary mapping pmid -> list of annotation dicts.

        Returns:
            Dictionary with:
                - relationships: List of ExtractedRelationship objects
                - statistics: Mining statistics
                - errors: List of errors
        """
        start_time = time.time()

        # Analyze abstracts
        analysis = self.analyze_abstracts(articles)
        logger.info(
            f"Analyzing {analysis['total_abstracts']} abstracts "
            f"({analysis['total_tokens']} tokens, "
            f"recommended {analysis['recommended_chunks']} chunks)"
        )

        # Chunk abstracts
        chunks = self.chunk_abstracts(articles, annotations)
        logger.info(f"Created {len(chunks)} chunks for parallel mining")

        for chunk in chunks:
            logger.debug(
                f"  Chunk {chunk.chunk_id}: {len(chunk.abstracts)} abstracts, "
                f"{chunk.total_tokens} tokens, {len(chunk.entities)} entities"
            )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # Mine all chunks in parallel
        tasks = [self.mine_chunk(chunk, semaphore) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        all_relationships: list[ExtractedRelationship] = []
        all_errors: list[str] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        chunks_processed = 0

        for result in results:
            if isinstance(result, Exception):
                all_errors.append(str(result))
            elif isinstance(result, ChunkMiningResult):
                all_relationships.extend(result.relationships)
                all_errors.extend(result.errors)
                total_prompt_tokens += result.token_usage.get("prompt_tokens", 0)
                total_completion_tokens += result.token_usage.get("completion_tokens", 0)
                if result.success:
                    chunks_processed += 1

        duration = time.time() - start_time

        # Deduplicate relationships
        unique_relationships = self._deduplicate_relationships(all_relationships)

        statistics = {
            "total_abstracts": len(articles),
            "total_tokens_input": analysis["total_tokens"],
            "chunks_created": len(chunks),
            "chunks_processed": chunks_processed,
            "relationships_raw": len(all_relationships),
            "relationships_extracted": len(unique_relationships),
            "tokens_used": total_prompt_tokens + total_completion_tokens,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "duration_seconds": round(duration, 2),
            "throughput_abstracts_per_sec": round(len(articles) / duration, 1) if duration > 0 else 0,
        }

        logger.info(
            f"Batched mining complete: {len(unique_relationships)} relationships "
            f"from {len(articles)} abstracts in {duration:.2f}s"
        )

        return {
            "relationships": unique_relationships,
            "statistics": statistics,
            "errors": all_errors,
        }

    def _deduplicate_relationships(
        self,
        relationships: list[ExtractedRelationship],
    ) -> list[ExtractedRelationship]:
        """
        Deduplicate relationships keeping highest confidence.

        Args:
            relationships: List of relationships to deduplicate.

        Returns:
            Deduplicated list of relationships.
        """
        # Key: (source, target, relation_type)
        seen: dict[tuple[str, str, str], ExtractedRelationship] = {}

        for rel in relationships:
            key = (
                rel.source_entity.lower().strip(),
                rel.target_entity.lower().strip(),
                rel.relation_type.lower().strip(),
            )

            if key not in seen:
                seen[key] = rel
            elif rel.confidence > seen[key].confidence:
                # Keep higher confidence version
                seen[key] = rel

        return list(seen.values())


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_batched_mining(
    articles: list[dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
    config: BatchedMiningConfig | None = None,
) -> dict[str, Any]:
    """
    Convenience function to run batched mining.

    Args:
        articles: List of article dicts.
        annotations: Dictionary mapping pmid -> annotations.
        config: Optional mining configuration.

    Returns:
        Mining results dictionary.
    """
    orchestrator = BatchedMiningOrchestrator(config)
    return await orchestrator.run(articles, annotations)


def run_batched_mining_sync(
    articles: list[dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
    config: BatchedMiningConfig | None = None,
) -> dict[str, Any]:
    """
    Synchronous wrapper for batched mining.

    Args:
        articles: List of article dicts.
        annotations: Dictionary mapping pmid -> annotations.
        config: Optional mining configuration.

    Returns:
        Mining results dictionary.
    """
    return asyncio.run(run_batched_mining(articles, annotations, config))


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Batched Mining Module")
    print("=" * 50)
    print(f"Default config: {BatchedMiningConfig()}")
