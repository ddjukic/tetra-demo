"""
LLM-based relationship extraction from biomedical abstracts.

This module uses Google's Gemini models to extract typed relationships
between biological entities from PubMed abstracts.
"""

import asyncio
import json
import logging
import os
from typing import Optional, Any

import google.generativeai as genai

logger = logging.getLogger(__name__)

# Configure Google Generative AI
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))


class RelationshipExtractor:
    """
    Extract typed relationships from abstracts using LLM.

    Uses Google's Gemini models with structured JSON output to identify
    relationship types between co-occurring entity pairs in biomedical text.
    """

    EXTRACTION_PROMPT = '''You are a biomedical relationship extractor.

Given an abstract and entity pairs that co-occur, determine the relationship type.

Abstract:
{abstract}

Entity pairs to classify:
{entity_pairs}

For each pair, classify as one of:
- ACTIVATES: Entity1 activates/increases/promotes Entity2
- INHIBITS: Entity1 inhibits/decreases/blocks Entity2
- ASSOCIATED_WITH: Entities are associated (correlation)
- REGULATES: Entity1 regulates Entity2 (direction unclear)
- BINDS_TO: Physical binding interaction
- COOCCURS_WITH: Mentioned together but no clear relationship

Return JSON array:
[{{"entity1": "...", "entity2": "...", "relationship": "...", "confidence": 0.0-1.0, "evidence_text": "..."}}]

Only include pairs where you can identify a relationship. Return an empty array if no relationships are found.
Important: Return ONLY valid JSON, no additional text.'''

    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Initialize the relationship extractor.

        Args:
            model: Gemini model to use for extraction
        """
        self.model = genai.GenerativeModel(model)

    async def extract_relationships(
        self,
        abstract: str,
        entity_pairs: list[tuple[str, str]],
        pmid: str
    ) -> list[dict[str, Any]]:
        """
        Extract relationships for co-occurring entity pairs in an abstract.

        Args:
            abstract: The abstract text to analyze
            entity_pairs: List of (entity1, entity2) tuples to classify
            pmid: PubMed ID for reference

        Returns:
            List of extracted relationships with keys:
                - entity1: First entity
                - entity2: Second entity
                - relationship: Relationship type (uppercase)
                - confidence: Confidence score (0-1)
                - evidence_text: Supporting text from abstract
                - pmid: Source PMID
        """
        if not entity_pairs:
            return []

        # Format entity pairs for the prompt
        pairs_text = "\n".join(f"- {e1} and {e2}" for e1, e2 in entity_pairs)

        prompt = self.EXTRACTION_PROMPT.format(
            abstract=abstract,
            entity_pairs=pairs_text
        )

        try:
            # Use asyncio to run the synchronous Gemini API
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=1500,
                        response_mime_type="application/json",
                    ),
                )
            )

            result_text = response.text or "[]"
            result = json.loads(result_text)

            # Handle both array and object with 'relationships' key
            if isinstance(result, list):
                relationships = result
            elif isinstance(result, dict) and "relationships" in result:
                relationships = result["relationships"]
            else:
                relationships = []

            # Add PMID to each relationship
            for rel in relationships:
                rel["pmid"] = pmid
                # Normalize relationship type to lowercase
                if "relationship" in rel:
                    rel["relationship"] = rel["relationship"].lower()

            return relationships

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response for PMID {pmid}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error extracting relationships from PMID {pmid}: {e}")
            return []

    async def batch_extract(
        self,
        abstracts: list[dict[str, Any]],
        max_concurrent: int = 5
    ) -> list[dict[str, Any]]:
        """
        Batch extraction with semaphore concurrency control.

        Args:
            abstracts: List of dictionaries with keys:
                - pmid: PubMed ID
                - abstract: Abstract text
                - entity_pairs: List of (entity1, entity2) tuples
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of all extracted relationships from all abstracts
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(item: dict[str, Any]) -> list[dict[str, Any]]:
            async with semaphore:
                pmid = item.get("pmid", "unknown")
                abstract = item.get("abstract", "")
                entity_pairs = item.get("entity_pairs", [])

                if not abstract or not entity_pairs:
                    return []

                return await self.extract_relationships(abstract, entity_pairs, pmid)

        # Process all abstracts concurrently with semaphore limit
        tasks = [process_one(item) for item in abstracts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and filter out exceptions
        all_relationships = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch extraction error: {result}")
            elif isinstance(result, list):
                all_relationships.extend(result)

        return all_relationships


class EntityPairExtractor:
    """
    Extract entity pairs from text for relationship classification.

    Uses NER to identify entities and returns all co-occurring pairs.
    This is a simpler approach than full relationship extraction.
    """

    ENTITY_EXTRACTION_PROMPT = '''Extract all biomedical entities (genes, proteins, diseases, chemicals) from this text.

Text:
{text}

Return JSON:
{{"entities": [{{"name": "...", "type": "gene|protein|disease|chemical"}}]}}

Only include clearly named entities, not general terms.'''

    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Initialize the entity pair extractor.

        Args:
            model: Gemini model to use for extraction
        """
        self.model = genai.GenerativeModel(model)

    async def extract_entities(self, text: str) -> list[dict[str, str]]:
        """
        Extract named entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            List of entity dictionaries with name and type
        """
        prompt = self.ENTITY_EXTRACTION_PROMPT.format(text=text)

        try:
            # Use asyncio to run the synchronous Gemini API
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=500,
                        response_mime_type="application/json",
                    ),
                )
            )

            result_text = response.text or "{}"
            result = json.loads(result_text)

            return result.get("entities", [])

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    async def get_entity_pairs(
        self,
        text: str,
        filter_types: Optional[list[str]] = None
    ) -> list[tuple[str, str]]:
        """
        Extract all entity pairs from text.

        Args:
            text: Text to extract from
            filter_types: Only include entities of these types (e.g., ["gene", "protein"])

        Returns:
            List of (entity1, entity2) tuples
        """
        entities = await self.extract_entities(text)

        if filter_types:
            entities = [e for e in entities if e.get("type") in filter_types]

        # Generate all pairs
        pairs = []
        names = [e["name"] for e in entities]
        for i, e1 in enumerate(names):
            for e2 in names[i + 1:]:
                pairs.append((e1, e2))

        return pairs
