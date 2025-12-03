"""
Batched Relationship Mining with LiteLLM Multi-Provider Support.

This module provides a unified batched mining orchestrator that:
- Uses token-aware chunking for optimal batching
- Supports multiple LLM providers via LiteLLM (Cerebras, Gemini, etc.)
- Includes provenance validation (PMID + evidence sentence matching)
- Uses configuration from extraction/config.toml

Architecture:
    BatchedLiteLLMMiner
        |
        +-- chunk_abstracts() -> list[AbstractChunk]
        |
        +-- mine_chunk(chunk) -> ChunkMiningResult
        |       +-- Uses LiteLLM with config-based provider routing
        |       +-- Structured output with PMID field
        |
        +-- validate_provenance(relationships, chunk) -> ValidationResult
                +-- Validates PMID exists in chunk
                +-- Validates evidence sentence appears in abstract

Usage:
    from extraction.batched_litellm_miner import BatchedLiteLLMMiner

    # Use default extractor (cerebras)
    miner = BatchedLiteLLMMiner()

    # Use specific extractor
    miner = BatchedLiteLLMMiner(extractor_name="gemini")

    # Run mining with validation
    results = await miner.run(articles, annotations)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from litellm import acompletion

from extraction.config_loader import (
    ExtractionConfig,
    get_config,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences for numbered indexing.

    Uses simple heuristics: split on . ? ! followed by space or end.
    Returns list of stripped sentences (non-empty).
    """
    if not text:
        return []

    sentences = []
    current = ""

    for i, char in enumerate(text):
        current += char
        # Check for sentence-ending punctuation
        if char in ".?!":
            # Look ahead to see if this is end of sentence
            next_char = text[i + 1] if i + 1 < len(text) else " "
            if next_char in " \n\t" or i + 1 >= len(text):
                stripped = current.strip()
                if stripped and len(stripped) > 10:  # Minimum sentence length
                    sentences.append(stripped)
                current = ""

    # Add any remaining text
    if current.strip() and len(current.strip()) > 10:
        sentences.append(current.strip())

    return sentences


def number_sentences(text: str) -> tuple[str, list[str]]:
    """
    Number sentences in text for reference.

    Returns:
        Tuple of (numbered_text, sentence_list)
        numbered_text: "[1] First sentence. [2] Second sentence."
        sentence_list: ["First sentence.", "Second sentence."]
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return text, []

    numbered_parts = []
    for i, sentence in enumerate(sentences, 1):
        numbered_parts.append(f"[{i}] {sentence}")

    return " ".join(numbered_parts), sentences


@dataclass
class AbstractChunk:
    """A batch of abstracts for mining."""

    chunk_id: int
    pmids: list[str]
    abstracts: list[dict[str, Any]]  # {pmid, title, abstract, year}
    entities: list[str]  # Unique entities in this chunk
    total_tokens: int
    sentence_map: dict[str, list[str]] = field(default_factory=dict)  # pmid -> sentences


@dataclass
class ExtractedRelationship:
    """A single extracted relationship with evidence."""

    entity1: str
    entity2: str
    relationship: str
    confidence: float
    pmid: str
    evidence_sentence_indices: list[int] = field(default_factory=list)
    evidence_text: str = ""  # Populated after sentence extraction

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entity1": self.entity1,
            "entity2": self.entity2,
            "relationship": self.relationship,
            "confidence": self.confidence,
            "pmid": self.pmid,
            "evidence_sentence_indices": self.evidence_sentence_indices,
            "evidence_text": self.evidence_text,
        }


@dataclass
class ValidationResult:
    """Result of provenance validation for a relationship."""

    relationship: ExtractedRelationship
    pmid_valid: bool
    evidence_valid: bool
    evidence_similarity: float  # 0.0-1.0 fuzzy match score
    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        """Check if provenance is fully valid."""
        return self.pmid_valid and self.evidence_valid

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "relationship": self.relationship.to_dict(),
            "pmid_valid": self.pmid_valid,
            "evidence_valid": self.evidence_valid,
            "evidence_similarity": self.evidence_similarity,
            "is_valid": self.is_valid,
            "error_message": self.error_message,
        }


@dataclass
class ChunkMiningResult:
    """Results from mining a single chunk."""

    chunk_id: int
    relationships: list[ExtractedRelationship]
    validation_results: list[ValidationResult]
    pmids_processed: list[str]
    token_usage: dict[str, int]
    latency_ms: float
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if mining was successful."""
        return len(self.errors) == 0

    @property
    def valid_relationships(self) -> list[ExtractedRelationship]:
        """Get only validated relationships."""
        return [v.relationship for v in self.validation_results if v.is_valid]

    @property
    def invalid_count(self) -> int:
        """Count of relationships that failed validation."""
        return sum(1 for v in self.validation_results if not v.is_valid)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "relationships": [r.to_dict() for r in self.relationships],
            "validation_results": [v.to_dict() for v in self.validation_results],
            "valid_relationships": [r.to_dict() for r in self.valid_relationships],
            "pmids_processed": self.pmids_processed,
            "token_usage": self.token_usage,
            "latency_ms": self.latency_ms,
            "errors": self.errors,
            "success": self.success,
            "invalid_count": self.invalid_count,
        }


@dataclass
class MiningStatistics:
    """Statistics from a complete mining run."""

    total_abstracts: int = 0
    total_chunks: int = 0
    chunks_processed: int = 0
    total_relationships: int = 0
    valid_relationships: int = 0
    invalid_relationships: int = 0
    pmid_failures: int = 0
    evidence_failures: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    wall_clock_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_abstracts": self.total_abstracts,
            "total_chunks": self.total_chunks,
            "chunks_processed": self.chunks_processed,
            "total_relationships": self.total_relationships,
            "valid_relationships": self.valid_relationships,
            "invalid_relationships": self.invalid_relationships,
            "pmid_failures": self.pmid_failures,
            "evidence_failures": self.evidence_failures,
            "validation_rate": (
                self.valid_relationships / self.total_relationships
                if self.total_relationships > 0
                else 0.0
            ),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_latency_ms": self.total_latency_ms,
            "wall_clock_ms": self.wall_clock_ms,
            "throughput_tok_per_sec": (
                (self.total_prompt_tokens + self.total_completion_tokens)
                / self.total_latency_ms
                * 1000
                if self.total_latency_ms > 0
                else 0.0
            ),
        }


# =============================================================================
# Token Counting
# =============================================================================


def count_tokens_simple(text: str) -> int:
    """Simple token count approximation (~4 chars per token)."""
    return len(text) // 4


def try_count_tokens_tiktoken(text: str) -> int:
    """Count tokens using tiktoken if available, else approximate."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return count_tokens_simple(text)


# =============================================================================
# Provenance Validation
# =============================================================================


def find_evidence_in_abstract(
    evidence_text: str,
    abstract: str,
    threshold: float = 0.7,
) -> tuple[bool, float]:
    """
    Check if evidence sentence appears in abstract using fuzzy matching.

    Args:
        evidence_text: The evidence sentence from the model
        abstract: The full abstract text
        threshold: Minimum similarity score to consider a match (0.0-1.0)

    Returns:
        Tuple of (is_valid, similarity_score)
    """
    if not evidence_text or not abstract:
        return False, 0.0

    # Normalize texts
    evidence_lower = evidence_text.lower().strip()
    abstract_lower = abstract.lower()

    # Quick exact substring check
    if evidence_lower in abstract_lower:
        return True, 1.0

    # Try sentence-level fuzzy matching
    # Split abstract into sentences (simple split on . ? !)
    sentences = []
    current = ""
    for char in abstract:
        current += char
        if char in ".?!":
            if current.strip():
                sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    best_ratio = 0.0
    for sentence in sentences:
        ratio = SequenceMatcher(None, evidence_lower, sentence.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio

    return best_ratio >= threshold, best_ratio


def validate_relationship_provenance(
    relationship: ExtractedRelationship,
    chunk: AbstractChunk,
    evidence_threshold: float = 0.7,
) -> ValidationResult:
    """
    Validate that a relationship's provenance is correct.

    Checks:
    1. PMID exists in the chunk's PMID list
    2. Evidence sentence appears in the corresponding abstract

    Args:
        relationship: The extracted relationship to validate
        chunk: The chunk that was used for extraction
        evidence_threshold: Minimum similarity for evidence matching

    Returns:
        ValidationResult with validation status
    """
    # Check PMID validity
    pmid_valid = relationship.pmid in chunk.pmids

    if not pmid_valid:
        return ValidationResult(
            relationship=relationship,
            pmid_valid=False,
            evidence_valid=False,
            evidence_similarity=0.0,
            error_message=f"PMID {relationship.pmid} not in chunk PMIDs: {chunk.pmids}",
        )

    # Find the abstract for this PMID
    abstract_text = ""
    for article in chunk.abstracts:
        if article.get("pmid") == relationship.pmid:
            abstract_text = article.get("abstract", "")
            break

    if not abstract_text:
        return ValidationResult(
            relationship=relationship,
            pmid_valid=True,
            evidence_valid=False,
            evidence_similarity=0.0,
            error_message=f"No abstract found for PMID {relationship.pmid}",
        )

    # Check evidence validity
    evidence_valid, similarity = find_evidence_in_abstract(
        relationship.evidence_text,
        abstract_text,
        threshold=evidence_threshold,
    )

    error_msg = None
    if not evidence_valid:
        error_msg = (
            f"Evidence not found in abstract (similarity: {similarity:.2f}): "
            f"'{relationship.evidence_text[:100]}...'"
        )

    return ValidationResult(
        relationship=relationship,
        pmid_valid=True,
        evidence_valid=evidence_valid,
        evidence_similarity=similarity,
        error_message=error_msg,
    )


def validate_relationship_provenance_indexed(
    relationship: ExtractedRelationship,
    chunk: AbstractChunk,
) -> ValidationResult:
    """
    Validate provenance using sentence indices (not fuzzy text matching).

    Checks:
    1. PMID exists in the chunk's PMID list
    2. All sentence indices are valid (within bounds)

    This is more reliable than fuzzy matching since we extract
    sentences by index rather than having the LLM reproduce text.

    Args:
        relationship: The extracted relationship to validate
        chunk: The chunk that was used for extraction

    Returns:
        ValidationResult with validation status
    """
    # Check PMID validity
    pmid_valid = relationship.pmid in chunk.pmids

    if not pmid_valid:
        return ValidationResult(
            relationship=relationship,
            pmid_valid=False,
            evidence_valid=False,
            evidence_similarity=0.0,
            error_message=f"PMID {relationship.pmid} not in chunk PMIDs: {chunk.pmids}",
        )

    # Get sentences for this PMID
    sentences = chunk.sentence_map.get(relationship.pmid, [])

    if not sentences:
        return ValidationResult(
            relationship=relationship,
            pmid_valid=True,
            evidence_valid=False,
            evidence_similarity=0.0,
            error_message=f"No sentences found for PMID {relationship.pmid}",
        )

    # Check if sentence indices are provided
    indices = relationship.evidence_sentence_indices
    if not indices:
        return ValidationResult(
            relationship=relationship,
            pmid_valid=True,
            evidence_valid=False,
            evidence_similarity=0.0,
            error_message="No evidence sentence indices provided",
        )

    # Check if all indices are valid (1-based)
    invalid_indices = []
    valid_count = 0
    for idx in indices:
        if isinstance(idx, int) and 1 <= idx <= len(sentences):
            valid_count += 1
        else:
            invalid_indices.append(idx)

    # Calculate validity score based on how many indices are valid
    evidence_similarity = valid_count / len(indices) if indices else 0.0
    evidence_valid = len(invalid_indices) == 0 and valid_count > 0

    error_msg = None
    if invalid_indices:
        error_msg = (
            f"Invalid sentence indices {invalid_indices} "
            f"(PMID {relationship.pmid} has {len(sentences)} sentences)"
        )
    elif valid_count == 0:
        error_msg = "No valid sentence indices"

    return ValidationResult(
        relationship=relationship,
        pmid_valid=True,
        evidence_valid=evidence_valid,
        evidence_similarity=evidence_similarity,
        error_message=error_msg,
    )


# =============================================================================
# BatchedLiteLLMMiner
# =============================================================================


class BatchedLiteLLMMiner:
    """
    Batched relationship mining with LiteLLM multi-provider support.

    Features:
    - Token-aware chunking for optimal batching
    - Config-based LLM provider selection (Cerebras, Gemini, etc.)
    - Provenance validation for all extracted relationships
    - Parallel chunk processing with semaphore control
    """

    def __init__(
        self,
        extractor_name: str | None = None,
        config: ExtractionConfig | None = None,
        evidence_threshold: float = 0.7,
        chunk_tokens: int | None = None,
    ):
        """
        Initialize the batched miner.

        Args:
            extractor_name: Name of extractor in config (e.g., "cerebras", "gemini")
            config: Optional config override
            evidence_threshold: Minimum similarity for evidence validation (0.0-1.0)
            chunk_tokens: Override for tokens per chunk (default from config)
        """
        self._config = config or get_config()
        self._extractor_name = extractor_name or self._config.DEFAULT_EXTRACTOR
        self._extractor_config = self._config.get_extractor(self._extractor_name)
        self._evidence_threshold = evidence_threshold
        self._chunk_tokens = chunk_tokens or self._config.BATCHED.TARGET_TOKENS_PER_CHUNK

        # Configure API keys
        self._configure_api_keys()

        logger.info(
            f"Initialized BatchedLiteLLMMiner: "
            f"extractor={self._extractor_name}, model={self._extractor_config.MODEL}"
        )

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._extractor_config.MODEL

    def _configure_api_keys(self) -> None:
        """Configure API keys for the selected provider."""
        model = self._extractor_config.MODEL

        if model.startswith("cerebras/"):
            if not os.environ.get("CEREBRAS_API_KEY"):
                logger.warning("CEREBRAS_API_KEY not set")
        elif model.startswith("openrouter/"):
            if not os.environ.get("OPENROUTER_API_KEY"):
                logger.warning("OPENROUTER_API_KEY not set")
            os.environ.setdefault("OR_SITE_URL", "https://tetra-kg.example.com")
            os.environ.setdefault("OR_APP_NAME", "tetra-kg-batched-miner")
        elif model.startswith("gemini/"):
            if not os.environ.get("GOOGLE_API_KEY"):
                logger.warning("GOOGLE_API_KEY not set")

    def chunk_abstracts(
        self,
        articles: list[dict[str, Any]],
        annotations: dict[str, list[dict[str, Any]]],
    ) -> list[AbstractChunk]:
        """
        Chunk abstracts by token count for optimal batching.

        Args:
            articles: List of article dicts with 'pmid', 'title', 'abstract', 'year'
            annotations: Dictionary mapping pmid -> list of annotation dicts

        Returns:
            List of AbstractChunk objects
        """
        if not articles:
            return []

        target_tokens = self._chunk_tokens
        min_chunks = self._config.BATCHED.MIN_CHUNKS

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
            if current_tokens + tokens > target_tokens and current_articles:
                chunk = self._create_chunk(
                    chunk_id, current_articles, annotations, current_tokens
                )
                chunks.append(chunk)
                chunk_id += 1
                current_articles = []
                current_tokens = 0

            current_articles.append(article)
            current_tokens += tokens

        # Add remaining articles
        if current_articles:
            chunk = self._create_chunk(
                chunk_id, current_articles, annotations, current_tokens
            )
            chunks.append(chunk)

        # Ensure minimum chunks for parallelization
        while len(chunks) < min_chunks and len(chunks) > 0:
            largest_idx = max(range(len(chunks)), key=lambda i: len(chunks[i].abstracts))
            largest = chunks[largest_idx]

            if len(largest.abstracts) < 2:
                break

            mid = len(largest.abstracts) // 2
            first_half = largest.abstracts[:mid]
            second_half = largest.abstracts[mid:]

            chunks[largest_idx] = self._create_chunk(
                largest.chunk_id,
                first_half,
                annotations,
                sum(
                    try_count_tokens_tiktoken(a.get("abstract", "") + a.get("title", ""))
                    for a in first_half
                ),
            )
            new_chunk = self._create_chunk(
                len(chunks),
                second_half,
                annotations,
                sum(
                    try_count_tokens_tiktoken(a.get("abstract", "") + a.get("title", ""))
                    for a in second_half
                ),
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
        """Create an AbstractChunk from articles."""
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

    def _build_prompt(self, chunk: AbstractChunk) -> tuple[str, str]:
        """
        Build system and user prompts for a chunk.

        Numbers sentences in each abstract and populates chunk.sentence_map.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Format abstracts with PMID markers and numbered sentences
        abstracts_text = []
        for article in chunk.abstracts:
            pmid = article.get("pmid", "unknown")
            title = article.get("title", "")
            abstract = article.get("abstract", "")
            year = article.get("year", "")

            # Number sentences and store mapping
            numbered_abstract, sentences = number_sentences(abstract)
            chunk.sentence_map[pmid] = sentences

            abstract_block = f"""[PMID: {pmid}] ({year})
Title: {title}
Abstract: {numbered_abstract}"""
            abstracts_text.append(abstract_block)

        abstracts_formatted = "\n\n---\n\n".join(abstracts_text)
        entities_formatted = ", ".join(sorted(set(chunk.entities)))
        relationship_types = ", ".join(self._config.SCHEMA.RELATIONSHIP_TYPES)

        system_prompt = self._config.BATCHED_PROMPT.SYSTEM
        user_prompt = self._config.BATCHED_PROMPT.USER_TEMPLATE.format(
            entities=entities_formatted,
            relationship_types=relationship_types,
            abstracts=abstracts_formatted,
        )

        return system_prompt, user_prompt

    async def mine_chunk(
        self,
        chunk: AbstractChunk,
        semaphore: asyncio.Semaphore,
    ) -> ChunkMiningResult:
        """
        Mine relationships from a single chunk using LiteLLM.

        Includes provenance validation for all extracted relationships.

        Args:
            chunk: AbstractChunk to mine
            semaphore: Asyncio semaphore for concurrency control

        Returns:
            ChunkMiningResult with extracted and validated relationships
        """
        async with semaphore:
            last_error: str | None = None
            max_retries = self._config.BATCHED.MAX_RETRIES
            retry_delay_ms = self._config.BATCHED.RETRY_DELAY_MS

            for attempt in range(max_retries):
                try:
                    result = await self._extract_from_chunk(chunk)
                    return result

                except Exception as e:
                    last_error = str(e)
                    error_str = str(e).lower()

                    is_rate_limit = "429" in error_str or "rate" in error_str
                    is_server_error = any(
                        str(code) in error_str for code in [503, 500, 502]
                    )

                    if (is_rate_limit or is_server_error) and attempt < max_retries - 1:
                        delay = (retry_delay_ms / 1000) * (2**attempt)
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

            return ChunkMiningResult(
                chunk_id=chunk.chunk_id,
                relationships=[],
                validation_results=[],
                pmids_processed=chunk.pmids,
                token_usage={},
                latency_ms=0.0,
                errors=[last_error or "Unknown error"],
            )

    async def _extract_from_chunk(self, chunk: AbstractChunk) -> ChunkMiningResult:
        """Extract relationships from a chunk using LiteLLM."""
        system_prompt, user_prompt = self._build_prompt(chunk)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        start_time = time.time()

        # Get provider-specific response format (with PMID field)
        response_format = self._config.get_batched_response_format(self._extractor_name)
        extra_params = self._config.get_extra_params(self._extractor_name)

        response = await acompletion(
            model=self.model,
            messages=messages,
            temperature=self._extractor_config.TEMPERATURE,
            max_tokens=self._config.BATCHED.MAX_TOKENS,  # Use batched config for higher output
            timeout=self._extractor_config.TIMEOUT,
            response_format=response_format,
            **extra_params,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract token usage
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Parse response
        result_text = response.choices[0].message.content or "{}"
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Chunk {chunk.chunk_id} JSON parse error: {e}")
            result = {"relationships": []}

        # Handle both array and object with 'relationships' key
        if isinstance(result, list):
            raw_relationships = result
        elif isinstance(result, dict) and "relationships" in result:
            raw_relationships = result["relationships"]
        else:
            raw_relationships = []

        # Convert to ExtractedRelationship objects and validate
        min_confidence = self._config.BATCHED.MIN_CONFIDENCE
        relationships: list[ExtractedRelationship] = []
        validation_results: list[ValidationResult] = []

        for rel in raw_relationships:
            confidence = rel.get("confidence", 0.5)
            if confidence < min_confidence:
                continue

            pmid = rel.get("pmid", "")
            sentence_indices = rel.get("evidence_sentence_indices", [])

            # Extract actual sentences from sentence_map
            evidence_text = ""
            sentences_for_pmid = chunk.sentence_map.get(pmid, [])
            valid_sentences = []
            for idx in sentence_indices:
                # Indices are 1-based in the prompt
                if isinstance(idx, int) and 1 <= idx <= len(sentences_for_pmid):
                    valid_sentences.append(sentences_for_pmid[idx - 1])
            evidence_text = " ".join(valid_sentences)

            extracted = ExtractedRelationship(
                entity1=rel.get("entity1", ""),
                entity2=rel.get("entity2", ""),
                relationship=rel.get("relationship", "").lower(),
                confidence=confidence,
                pmid=pmid,
                evidence_sentence_indices=sentence_indices,
                evidence_text=evidence_text,
            )
            relationships.append(extracted)

            # Validate provenance (now uses index-based validation)
            validation = validate_relationship_provenance_indexed(
                extracted, chunk
            )
            validation_results.append(validation)

        logger.info(
            f"Chunk {chunk.chunk_id}: extracted {len(relationships)} relationships "
            f"({sum(1 for v in validation_results if v.is_valid)} valid) "
            f"from {len(chunk.pmids)} abstracts ({latency_ms:.0f}ms)"
        )

        return ChunkMiningResult(
            chunk_id=chunk.chunk_id,
            relationships=relationships,
            validation_results=validation_results,
            pmids_processed=chunk.pmids,
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            latency_ms=latency_ms,
            errors=[],
        )

    async def run(
        self,
        articles: list[dict[str, Any]],
        annotations: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Run the full batched mining pipeline with provenance validation.

        Args:
            articles: List of article dicts with 'pmid', 'title', 'abstract', 'year'
            annotations: Dictionary mapping pmid -> list of annotation dicts

        Returns:
            Dictionary with:
                - relationships: All extracted relationships
                - valid_relationships: Only validated relationships
                - validation_results: Full validation details
                - statistics: Mining statistics
                - errors: List of errors
        """
        start_time = time.time()

        # Chunk abstracts
        chunks = self.chunk_abstracts(articles, annotations)
        logger.info(
            f"Created {len(chunks)} chunks from {len(articles)} abstracts "
            f"for {self._extractor_name} extraction"
        )

        for chunk in chunks:
            logger.debug(
                f"  Chunk {chunk.chunk_id}: {len(chunk.abstracts)} abstracts, "
                f"{chunk.total_tokens} tokens, {len(chunk.entities)} entities"
            )

        # Create semaphore for concurrency control
        max_concurrent = self._config.BATCHED.MAX_CONCURRENT
        semaphore = asyncio.Semaphore(max_concurrent)

        # Mine all chunks in parallel
        tasks = [self.mine_chunk(chunk, semaphore) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results and build statistics
        all_relationships: list[ExtractedRelationship] = []
        valid_relationships: list[ExtractedRelationship] = []
        all_validation_results: list[ValidationResult] = []
        all_errors: list[str] = []

        stats = MiningStatistics(
            total_abstracts=len(articles),
            total_chunks=len(chunks),
        )

        for result in results:
            if isinstance(result, Exception):
                all_errors.append(str(result))
                continue

            if isinstance(result, ChunkMiningResult):
                all_relationships.extend(result.relationships)
                all_validation_results.extend(result.validation_results)
                all_errors.extend(result.errors)

                if result.success:
                    stats.chunks_processed += 1
                    stats.total_prompt_tokens += result.token_usage.get("prompt_tokens", 0)
                    stats.total_completion_tokens += result.token_usage.get(
                        "completion_tokens", 0
                    )
                    stats.total_latency_ms += result.latency_ms

                    # Count validation outcomes
                    for v in result.validation_results:
                        if v.is_valid:
                            valid_relationships.append(v.relationship)
                        else:
                            if not v.pmid_valid:
                                stats.pmid_failures += 1
                            elif not v.evidence_valid:
                                stats.evidence_failures += 1

        stats.total_relationships = len(all_relationships)
        stats.valid_relationships = len(valid_relationships)
        stats.invalid_relationships = stats.total_relationships - stats.valid_relationships
        stats.wall_clock_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Batched mining complete: {stats.valid_relationships}/{stats.total_relationships} "
            f"valid relationships from {stats.chunks_processed}/{stats.total_chunks} chunks "
            f"in {stats.wall_clock_ms:.0f}ms"
        )

        if stats.pmid_failures > 0 or stats.evidence_failures > 0:
            logger.warning(
                f"Validation failures: {stats.pmid_failures} PMID, "
                f"{stats.evidence_failures} evidence"
            )

        return {
            "relationships": [r.to_dict() for r in all_relationships],
            "valid_relationships": [r.to_dict() for r in valid_relationships],
            "validation_results": [v.to_dict() for v in all_validation_results],
            "statistics": stats.to_dict(),
            "errors": all_errors,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_batched_mining(
    articles: list[dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
    extractor_name: str | None = None,
    evidence_threshold: float = 0.7,
) -> dict[str, Any]:
    """
    Convenience function to run batched mining with validation.

    Args:
        articles: List of article dicts
        annotations: Dictionary mapping pmid -> annotations
        extractor_name: Name of extractor (default from config)
        evidence_threshold: Minimum similarity for evidence validation

    Returns:
        Mining results dictionary with validation
    """
    miner = BatchedLiteLLMMiner(
        extractor_name=extractor_name,
        evidence_threshold=evidence_threshold,
    )
    return await miner.run(articles, annotations)


def create_batched_miner(
    extractor_name: str | None = None,
    evidence_threshold: float = 0.7,
) -> BatchedLiteLLMMiner:
    """
    Factory function to create a batched miner.

    Args:
        extractor_name: Name of extractor in config (e.g., "cerebras", "gemini")
        evidence_threshold: Minimum similarity for evidence validation

    Returns:
        Configured BatchedLiteLLMMiner instance
    """
    return BatchedLiteLLMMiner(
        extractor_name=extractor_name,
        evidence_threshold=evidence_threshold,
    )
