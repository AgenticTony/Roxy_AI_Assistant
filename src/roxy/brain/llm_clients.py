"""LLM client implementations for Roxy.

Provides unified interface for local (Ollama) and cloud LLM providers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

import httpx
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from .rate_limiter import RateLimiter, RateLimiterAware

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from any LLM client."""

    content: str
    model: str
    provider: Literal["local", "cloud"]
    confidence: float | None = None
    tokens_used: int | None = None
    latency_ms: int | None = None

    def __str__(self) -> str:
        return f"LLMResponse(provider={self.provider}, model={self.model}, content_length={len(self.content)})"


class LLMClient:
    """Protocol for LLM clients.

    All LLM clients must implement the generate() method with this signature.
    """

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        raise NotImplementedError


class OllamaClient(LLMClient):
    """
    Client for local Ollama LLM.

    Uses OpenAI-compatible API with base_url pointing to Ollama server.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        router_model: str = "qwen3:0.6b",
        timeout: float = 120.0,
    ) -> None:
        """Initialize Ollama client.

        Args:
            host: Ollama server host URL.
            model: Main model name to use.
            router_model: Smaller model for fast confidence scoring.
            timeout: Request timeout in seconds.
        """
        self.host = host
        self.model = model
        self.router_model = router_model
        self.timeout = timeout

        # Create HTTP client with connection pooling
        self._http_client = httpx.AsyncClient(
            base_url=host,
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

        # Create OpenAI client pointing to Ollama
        self._client = AsyncOpenAI(
            base_url=f"{host}/v1",
            api_key="ollama",  # Ollama doesn't require a key
            http_client=self._http_client,
        )

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from Ollama.

        Args:
            prompt: The prompt to send to the LLM.
            temperature: Sampling temperature (0.0-2.0). None uses model default.
            max_tokens: Maximum tokens to generate. None uses model default.

        Returns:
            LLMResponse with generated content and metadata.

        Raises:
            APIConnectionError: If connection to Ollama fails.
            APIError: If the API request fails.
        """
        start_time = time.time()

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else None

            logger.debug(
                f"Ollama response: model={self.model}, tokens={tokens_used}, latency={latency_ms}ms"
            )

            return LLMResponse(
                content=content,
                model=self.model,
                provider="local",
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )

        except APIConnectionError as e:
            logger.error(f"Failed to connect to Ollama at {self.host}: {e}")
            raise
        except APIError as e:
            logger.error(f"Ollama API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Ollama client: {e}")
            raise APIError(f"Unexpected error: {e}")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http_client.aclose()

    async def __aenter__(self) -> OllamaClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


class CloudLLMClient(LLMClient, RateLimiterAware):
    """
    Client for cloud LLM providers.

    Supports multiple providers via OpenAI-compatible API:
    - Z.ai (ChatGLM)
    - OpenRouter

    Includes persistent rate limiting that survives restarts.
    """

    def __init__(
        self,
        provider: Literal["zai", "openrouter"] = "zai",
        model: str = "glm-4.7",
        api_key: str = "",
        base_url: str = "",
        timeout: float = 60.0,
        rate_limiter: RateLimiter | None = None,
        rate_limit_enabled: bool = True,
    ) -> None:
        """Initialize cloud LLM client.

        Args:
            provider: Cloud provider name.
            model: Model name to use.
            api_key: API key for the provider.
            base_url: Base URL for the provider API.
            timeout: Request timeout in seconds.
            rate_limiter: Optional custom rate limiter. If None, creates default.
            rate_limit_enabled: If False, bypass rate limit checks.
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._rate_limit_enabled = rate_limit_enabled

        # Initialize rate limiter support
        RateLimiterAware.__init__(self, rate_limiter)

        # Set default base URL if not provided
        if not base_url:
            if provider == "zai":
                self.base_url = "https://api.z.ai/api/paas/v4"
            elif provider == "openrouter":
                self.base_url = "https://openrouter.ai/api/v1"

        # Create HTTP client with connection pooling
        self._http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

        # Create OpenAI client
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            http_client=self._http_client,
        )

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
    ) -> LLMResponse:
        """Generate a response from cloud LLM.

        Args:
            prompt: The prompt to send to the LLM.
            temperature: Sampling temperature. None uses model default.
            max_tokens: Maximum tokens to generate. None uses model default.
            max_retries: Maximum number of retries on rate limit errors.

        Returns:
            LLMResponse with generated content and metadata.

        Raises:
            APIConnectionError: If connection to provider fails.
            APIError: If the API request fails after retries.
            RateLimitError: If rate limit is exceeded after retries.
        """
        if not self.api_key:
            self._record_request_result(False, None, "no_api_key")
            raise APIError("API key is required for cloud LLM provider")

        # Check local rate limit before making request
        if self._rate_limit_enabled:
            allowed, message = self.check_rate_limit(self.provider)
            if not allowed:
                logger.warning(f"Local rate limit exceeded: {message}")
                self._record_request_result(False, None, "local_rate_limit")
                raise RateLimitError(message)

        start_time = time.time()
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                latency_ms = int((time.time() - start_time) * 1000)

                content = response.choices[0].message.content or ""
                tokens_used = response.usage.total_tokens if response.usage else None

                logger.info(
                    f"Cloud LLM response: provider={self.provider}, model={self.model}, "
                    f"tokens={tokens_used}, latency={latency_ms}ms"
                )

                # Record successful request
                self._record_request_result(True, None, None)

                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider="cloud",
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                )

            except RateLimitError as e:
                last_error = e
                # Record rate limit hit
                self._record_request_result(False, 429, "api_rate_limit")

                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = 2**attempt
                    logger.warning(
                        f"Rate limited, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    await self._http_client.aclose()
                    self._http_client = httpx.AsyncClient(timeout=self.timeout)
                    self._client = AsyncOpenAI(
                        base_url=self.base_url,
                        api_key=self.api_key,
                        http_client=self._http_client,
                    )
                    continue
                logger.error(f"Rate limit exceeded after {max_retries} retries")
                raise

            except APIConnectionError as e:
                last_error = e
                self._record_request_result(False, None, "connection_error")
                logger.error(f"Failed to connect to {self.provider} at {self.base_url}: {e}")
                raise

            except APIError as e:
                last_error = e
                self._record_request_result(False, None, "api_error")
                logger.error(f"{self.provider} API error: {e}")
                raise

            except Exception as e:
                last_error = e
                self._record_request_result(False, None, "unknown_error")
                logger.error(f"Unexpected error in cloud LLM client: {e}")
                raise APIError(f"Unexpected error: {e}")

        # Should not reach here, but just in case
        raise last_error or APIError("Unknown error in cloud LLM client")

    def _record_request_result(
        self,
        success: bool,
        status_code: int | None,
        error_reason: str | None,
    ) -> None:
        """Record request result in rate limiter.

        Args:
            success: Whether the request succeeded.
            status_code: HTTP status code if applicable.
            error_reason: Error reason for logging.
        """
        if self._rate_limit_enabled:
            self.record_request(
                provider=self.provider,
                model=self.model,
                success=success,
                status_code=status_code,
            )

        if error_reason and not success:
            logger.debug(f"Request failed: {error_reason}")

    def get_rate_limit_stats(self) -> dict:
        """Get current rate limit statistics.

        Returns:
            Dictionary with rate limit statistics for this provider.
        """
        return self._rate_limiter.get_stats(self.provider)

    def get_failed_requests(self, hours: int = 1) -> list[dict]:
        """Get recent failed requests for debugging.

        Args:
            hours: Lookback period in hours.

        Returns:
            List of failed request records as dictionaries.
        """

        records = self._rate_limiter.get_failed_requests(self.provider, hours)
        return [r.to_dict() for r in records]

    def reset_rate_limits(self) -> None:
        """Reset rate limit tracking for this provider."""
        self._rate_limiter.reset(self.provider)
        logger.info(f"Reset rate limits for provider: {self.provider}")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http_client.aclose()

    async def __aenter__(self) -> CloudLLMClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


class ConfidenceScorer:
    """
    Uses a smaller, faster LLM to assess confidence in handling a request.

    This helps decide whether to use local or cloud LLM for the actual response.
    """

    def __init__(
        self,
        client: OllamaClient,
        model: str = "qwen3:0.6b",
    ) -> None:
        """Initialize confidence scorer.

        Args:
            client: Ollama client to use for scoring.
            model: Smaller model name for fast scoring.
        """
        self.client = client
        self.model = model
        self._scoring_prompt = """Analyze the following user request and rate your confidence (0.0 to 1.0) that a local AI assistant can handle it effectively.

Request: "{request}"

Consider:
- Does this require recent information not in your training data?
- Does this require complex reasoning or calculations?
- Does this require access to external services?
- Is this within the general knowledge domain?

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

    async def score(self, request: str) -> float:
        """
        Assess confidence in handling the request locally.

        Args:
            request: The user request to assess.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        prompt = self._scoring_prompt.format(request=request)

        try:
            response = await self.client._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=10,
            )

            content = response.choices[0].message.content or "0.5"

            # Extract number from response
            import re

            match = re.search(r"0?\.\d+|1\.0|0|1", content)
            if match:
                confidence = float(match.group())
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                logger.debug(f"Confidence score for request: {confidence}")
                return confidence

            logger.warning(f"Could not parse confidence score from response: {content}")
            return 0.5

        except Exception as e:
            logger.error(f"Error getting confidence score: {e}")
            return 0.5  # Default to middle confidence on error
