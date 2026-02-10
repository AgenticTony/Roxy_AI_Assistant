"""Confidence-based router for local/cloud LLM decisions.

Routes requests to local or cloud LLMs based on confidence scoring
and user privacy preferences.
"""

from __future__ import annotations

import logging

from .llm_clients import CloudLLMClient, ConfidenceScorer, LLMResponse, OllamaClient
from .privacy import PrivacyGateway

logger = logging.getLogger(__name__)


class ConfidenceRouter:
    """
    Routes requests between local and cloud LLMs.

    Algorithm:
    1. Assess local LLM's confidence in handling the request
    2. If confidence >= threshold, use local LLM
    3. If confidence < threshold:
       - Check privacy consent
       - Redact PII
       - Send to cloud LLM
       - Restore PII in response
       - Log the cloud request
    """

    def __init__(
        self,
        local_client: OllamaClient,
        cloud_client: CloudLLMClient,
        privacy: PrivacyGateway,
        confidence_threshold: float = 0.7,
    ) -> None:
        """Initialize confidence router.

        Args:
            local_client: Client for local Ollama LLM.
            cloud_client: Client for cloud LLM provider.
            privacy: Privacy gateway for PII handling.
            confidence_threshold: Minimum confidence for local processing.
        """
        self.local_client = local_client
        self.cloud_client = cloud_client
        self.privacy = privacy
        self.threshold = confidence_threshold

        # Create confidence scorer with smaller model
        self.scorer = ConfidenceScorer(
            client=local_client,
            model=local_client.router_model
            if hasattr(local_client, "router_model")
            else "qwen3:0.6b",
        )

        # Statistics
        self._total_requests = 0
        self._local_requests = 0
        self._cloud_requests = 0
        self._blocked_requests = 0

    async def route(
        self,
        request: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Route request to appropriate LLM based on confidence.

        Args:
            request: The user request to process.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse from either local or cloud provider.
        """
        self._total_requests += 1

        logger.info(f"Routing request: {request[:100]}...")

        # Assess confidence for local handling
        confidence = await self._assess_confidence(request)
        logger.debug(f"Local confidence score: {confidence:.2f}")

        # Check if we can handle locally
        if confidence >= self.threshold:
            logger.info(f"Using local LLM (confidence: {confidence:.2f} >= {self.threshold})")
            self._local_requests += 1
            return await self._use_local(request, temperature, max_tokens)

        # Need cloud - check consent
        can_use, message = await self.privacy.can_use_cloud()

        if not can_use:
            logger.info(f"Cloud access denied: {message}")
            self._blocked_requests += 1
            # Fall back to local even with low confidence
            logger.warning("Using local LLM due to cloud access denial")
            self._local_requests += 1
            return await self._use_local(request, temperature, max_tokens)

        # Use cloud with privacy protection
        logger.info(f"Using cloud LLM (confidence: {confidence:.2f} < {self.threshold})")
        self._cloud_requests += 1
        return await self._use_cloud(request, temperature, max_tokens)

    async def _assess_confidence(self, request: str) -> float:
        """
        Assess local LLM's confidence on handling the request.

        Uses the smaller router model for fast confidence scoring.

        Args:
            request: The user request to assess.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        try:
            return await self.scorer.score(request)
        except Exception as e:
            logger.error(f"Error assessing confidence: {e}")
            # Default to middle confidence on error
            return 0.5

    async def _use_local(
        self,
        request: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Use local LLM for request.

        Args:
            request: The user request.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            LLMResponse from local provider.
        """
        try:
            return await self.local_client.generate(
                prompt=request,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Local LLM error: {e}")
            raise

    async def _use_cloud(
        self,
        request: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Use cloud LLM for request with privacy protection.

        Args:
            request: The user request.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            LLMResponse from cloud provider with PII restored.
        """
        # Redact PII from request
        redaction_result = self.privacy.redact(request)

        if redaction_result.was_redacted:
            logger.info(
                f"Redacted {len(redaction_result.pii_matches)} PII instances before cloud call"
            )

        try:
            # Send redacted request to cloud
            response = await self.cloud_client.generate(
                prompt=redaction_result.redacted_text,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Restore PII in response
            if redaction_result.was_redacted:
                response.content = self.privacy.restore(
                    response.content, redaction_result.pii_matches
                )
                logger.debug("Restored PII in cloud response")

            # Log the cloud request
            await self.privacy.log_cloud_request(
                original_prompt=request,
                redacted_prompt=redaction_result.redacted_text,
                provider=self.cloud_client.provider,
                model=self.cloud_client.model,
                response_summary=response.content[:200],
            )

            return response

        except Exception as e:
            logger.error(f"Cloud LLM error: {e}")
            raise

    def get_statistics(self) -> dict[str, int | float]:
        """
        Get routing statistics.

        Returns:
            Dictionary with routing metrics.
        """
        return {
            "total_requests": self._total_requests,
            "local_requests": self._local_requests,
            "cloud_requests": self._cloud_requests,
            "blocked_requests": self._blocked_requests,
            "local_rate": self._local_requests / self._total_requests
            if self._total_requests > 0
            else 0.0,
            "cloud_rate": self._cloud_requests / self._total_requests
            if self._total_requests > 0
            else 0.0,
        }

    def reset_statistics(self) -> None:
        """Reset routing statistics."""
        self._total_requests = 0
        self._local_requests = 0
        self._cloud_requests = 0
        self._blocked_requests = 0
