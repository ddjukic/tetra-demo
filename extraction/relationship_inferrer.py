"""
LLM-based relationship inference for ML-predicted edges.

This module uses Google's Gemini models to infer relationship types for
edges predicted by the ML link predictor that lack literature support.
"""

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any

import google.generativeai as genai

if TYPE_CHECKING:
    from models.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Configure Google Generative AI
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))


class RelationshipInferrer:
    """
    Infer relationship types for ML-predicted edges based on graph context.

    Uses the local neighborhood of each protein in the knowledge graph
    to provide context for LLM-based inference of the likely relationship type.
    """

    INFERENCE_PROMPT = '''You are a biomedical knowledge expert analyzing a predicted protein interaction.

A machine learning model predicts that {protein_a} and {protein_b} likely interact (probability: {ml_score:.2f}),
but there is no direct literature evidence.

Known interactions for {protein_a}:
{protein_a_interactions}

Known interactions for {protein_b}:
{protein_b_interactions}

Based on this context:
1. What type of relationship would you hypothesize? (ACTIVATES, INHIBITS, BINDS_TO, REGULATES, COMPLEX_MEMBER, PATHWAY_NEIGHBOR)
2. Confidence level: LOW, MEDIUM, HIGH
3. Reasoning (2-3 sentences explaining your hypothesis)
4. What experiments would validate this?

Return JSON:
{{"hypothesized_relationship": "...", "confidence": "...", "reasoning": "...", "validation_experiments": ["...", "..."]}}

Important: Return ONLY valid JSON, no additional text.'''

    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Initialize the relationship inferrer.

        Args:
            model: Gemini model to use for inference
        """
        self.model = genai.GenerativeModel(model)

    async def infer_relationship(
        self,
        protein_a: str,
        protein_b: str,
        ml_score: float,
        graph: "KnowledgeGraph"
    ) -> dict[str, Any]:
        """
        Infer relationship type for a novel predicted edge.

        Uses the known interactions of both proteins to provide context
        for the LLM to hypothesize the relationship type.

        Args:
            protein_a: First protein identifier
            protein_b: Second protein identifier
            ml_score: ML prediction probability (0-1)
            graph: KnowledgeGraph containing known interactions

        Returns:
            Dictionary with keys:
                - hypothesized_relationship: Inferred relationship type
                - confidence: LOW, MEDIUM, or HIGH
                - reasoning: Explanation for the hypothesis
                - validation_experiments: Suggested experiments
                - protein_a: First protein
                - protein_b: Second protein
                - ml_score: Original ML score
        """
        # Get interaction summaries for context
        protein_a_interactions = graph.get_entity_interactions_summary(protein_a)
        protein_b_interactions = graph.get_entity_interactions_summary(protein_b)

        prompt = self.INFERENCE_PROMPT.format(
            protein_a=protein_a,
            protein_b=protein_b,
            ml_score=ml_score,
            protein_a_interactions=protein_a_interactions,
            protein_b_interactions=protein_b_interactions
        )

        try:
            # Use asyncio to run the synchronous Gemini API
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1000,
                        response_mime_type="application/json",
                    ),
                )
            )

            result_text = response.text or "{}"
            result = json.loads(result_text)

            # Normalize and add metadata
            result["protein_a"] = protein_a
            result["protein_b"] = protein_b
            result["ml_score"] = ml_score

            # Normalize relationship type to lowercase
            if "hypothesized_relationship" in result:
                result["hypothesized_relationship"] = result["hypothesized_relationship"].lower()

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response for {protein_a}-{protein_b}: {e}")
            return {
                "protein_a": protein_a,
                "protein_b": protein_b,
                "ml_score": ml_score,
                "hypothesized_relationship": "unknown",
                "confidence": "LOW",
                "reasoning": "Failed to parse LLM response",
                "validation_experiments": [],
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Error inferring relationship for {protein_a}-{protein_b}: {e}")
            return {
                "protein_a": protein_a,
                "protein_b": protein_b,
                "ml_score": ml_score,
                "hypothesized_relationship": "unknown",
                "confidence": "LOW",
                "reasoning": f"Error during inference: {str(e)}",
                "validation_experiments": [],
                "error": str(e)
            }

    async def batch_infer(
        self,
        predictions: list[dict[str, Any]],
        graph: "KnowledgeGraph",
        max_concurrent: int = 3
    ) -> list[dict[str, Any]]:
        """
        Batch inference for multiple predicted edges.

        Args:
            predictions: List of prediction dicts with keys:
                - source: Source protein
                - target: Target protein
                - ml_score: ML prediction score
            graph: KnowledgeGraph for context
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of inference results
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(pred: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self.infer_relationship(
                    pred.get("source", ""),
                    pred.get("target", ""),
                    pred.get("ml_score", 0.0),
                    graph
                )

        tasks = [process_one(pred) for pred in predictions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch inference error: {result}")
            elif isinstance(result, dict):
                valid_results.append(result)

        return valid_results


class HypothesisGenerator:
    """
    Generate testable hypotheses from ML predictions and graph context.

    Combines relationship inference with experimental design suggestions.
    """

    HYPOTHESIS_PROMPT = '''You are a drug discovery scientist generating testable hypotheses.

Based on a predicted interaction and its context, generate a structured hypothesis.

Prediction: {protein_a} and {protein_b} may interact (ML score: {ml_score:.2f})
Inferred relationship: {inferred_relationship}
Reasoning: {reasoning}

Known context:
{context}

Generate a structured hypothesis:
1. Hypothesis statement (one clear, testable statement)
2. Biological mechanism (proposed mechanism based on context)
3. Supporting evidence (what from the context supports this)
4. Counter-evidence (what might argue against this)
5. Validation approach (primary experiment to test)
6. Impact if true (significance for drug discovery)

Return JSON:
{{
    "hypothesis": "...",
    "mechanism": "...",
    "supporting_evidence": ["...", "..."],
    "counter_evidence": ["...", "..."],
    "validation_approach": "...",
    "impact": "...",
    "confidence_score": 0.0-1.0
}}'''

    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Initialize hypothesis generator.

        Args:
            model: Gemini model for hypothesis generation
        """
        self.model = genai.GenerativeModel(model)

    async def generate_hypothesis(
        self,
        inference_result: dict[str, Any],
        graph: "KnowledgeGraph"
    ) -> dict[str, Any]:
        """
        Generate a structured hypothesis from an inference result.

        Args:
            inference_result: Output from RelationshipInferrer.infer_relationship
            graph: KnowledgeGraph for additional context

        Returns:
            Structured hypothesis dictionary
        """
        protein_a = inference_result.get("protein_a", "")
        protein_b = inference_result.get("protein_b", "")

        # Get richer context
        context_a = graph.get_entity_interactions_summary(protein_a)
        context_b = graph.get_entity_interactions_summary(protein_b)
        context = f"{context_a}\n\n{context_b}"

        prompt = self.HYPOTHESIS_PROMPT.format(
            protein_a=protein_a,
            protein_b=protein_b,
            ml_score=inference_result.get("ml_score", 0.0),
            inferred_relationship=inference_result.get("hypothesized_relationship", "unknown"),
            reasoning=inference_result.get("reasoning", "No reasoning provided"),
            context=context
        )

        try:
            # Use asyncio to run the synchronous Gemini API
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.4,
                        max_output_tokens=1500,
                        response_mime_type="application/json",
                    ),
                )
            )

            result_text = response.text or "{}"
            result = json.loads(result_text)

            # Add source information
            result["protein_a"] = protein_a
            result["protein_b"] = protein_b
            result["ml_score"] = inference_result.get("ml_score", 0.0)
            result["inferred_relationship"] = inference_result.get("hypothesized_relationship", "unknown")

            return result

        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            return {
                "protein_a": protein_a,
                "protein_b": protein_b,
                "hypothesis": f"Failed to generate hypothesis: {str(e)}",
                "error": str(e)
            }
