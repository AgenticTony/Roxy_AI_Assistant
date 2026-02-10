"""Protocol definitions for dependency injection in Roxy's brain.

These protocols define the interfaces that can be injected into the Orchestrator,
enabling loose coupling and easier testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .llm_clients import LLMResponse


@dataclass
class PrivacyRedactionResult:
    """Result of PII redaction from privacy gateway."""

    redacted_text: str
    pii_matches: list[Any]
    was_redacted: bool = False


class PrivacyGatewayProtocol(Protocol):
    """
    Protocol for privacy gateway implementations.

    A privacy gateway handles PII detection, redaction, restoration,
    and user consent management for cloud LLM access.
    """

    async def can_use_cloud(self) -> tuple[bool, str | None]:
        """Check if cloud LLM can be used based on consent mode."""
        ...

    def redact(self, text: str) -> PrivacyRedactionResult:
        """Detect and redact PII from text."""
        ...

    def restore(self, text: str, pii_matches: list[Any]) -> str:
        """Restore PII placeholders with original values."""
        ...

    async def log_cloud_request(
        self,
        original_prompt: str,
        redacted_prompt: str,
        provider: str,
        model: str,
        response_summary: str,
    ) -> None:
        """Log cloud request to file for audit."""
        ...

    @property
    def consent_mode(self) -> Any:
        """Get the current consent mode."""
        ...


class LocalLLMClientProtocol(Protocol):
    """
    Protocol for local LLM client implementations.

    A local LLM client generates responses using an on-device model.
    """

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the local LLM."""
        ...

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        ...

    @property
    def router_model(self) -> str:
        """Get the model name used for fast confidence scoring."""
        ...

    @property
    def host(self) -> str:
        """Get the Ollama server host URL."""
        ...

    @property
    def model(self) -> str:
        """Get the main model name."""
        ...


class CloudLLMClientProtocol(Protocol):
    """
    Protocol for cloud LLM client implementations.

    A cloud LLM client generates responses using external API providers.
    """

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the cloud LLM."""
        ...

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        ...

    @property
    def provider(self) -> str:
        """Get the cloud provider name."""
        ...

    @property
    def model(self) -> str:
        """Get the model name."""
        ...


class ConfidenceRouterProtocol(Protocol):
    """
    Protocol for confidence router implementations.

    A confidence router decides between local and cloud LLMs
    based on confidence scoring and privacy considerations.
    """

    async def route(
        self,
        request: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Route request to appropriate LLM based on confidence."""
        ...

    def get_statistics(self) -> dict[str, int | float]:
        """Get routing statistics."""
        ...


class MemoryManagerProtocol(Protocol):
    """
    Protocol for memory manager implementations.

    A memory manager provides access to session context,
    conversation history, and long-term memory.
    """

    async def initialize(self) -> None:
        """Initialize all memory tiers."""
        ...

    async def build_context_for_llm(self, query: str) -> dict[str, Any]:
        """Build complete context for LLM prompt."""
        ...

    async def add_to_session(self, role: str, content: str) -> None:
        """Add a message to the current session."""
        ...

    async def remember(self, key: str, value: str) -> None:
        """Store a fact in long-term memory."""
        ...

    async def recall(self, query: str, limit: int = 5) -> list[str]:
        """Retrieve relevant long-term memories."""
        ...

    async def end_conversation(self) -> None:
        """End the current conversation session."""
        ...


class SkillRegistryProtocol(Protocol):
    """
    Protocol for skill registry implementations.

    A skill registry manages skill discovery, registration,
    and matching intents to skills.
    """

    def find_skill(
        self,
        intent: str,
        parameters: dict[str, Any],
    ) -> tuple[Any, float]:
        """Find the best matching skill for the given intent.

        Returns:
            Tuple of (skill, confidence_score) or (None, 0.0) if no match.
        """
        ...

    def list_skills(self) -> list[dict[str, Any]]:
        """List all registered skills.

        Returns:
            List of skill dicts with 'name' and 'description' keys.
        """
        ...

    async def initialize(self) -> None:
        """Initialize the skill registry."""
        ...
