"""Roxy brain module.

Contains the core orchestration logic for Roxy's AI assistant.
"""

from roxy.brain.llm_clients import (
    LLMClient,
    LLMResponse,
    OllamaClient,
    CloudLLMClient,
    ConfidenceScorer,
)
from roxy.brain.privacy import (
    PrivacyGateway,
    PIIMatch,
    RedactionResult,
    ConsentMode,
)
from roxy.brain.router import ConfidenceRouter
from roxy.brain.orchestrator import RoxyOrchestrator, OrchestratorConfig

__all__ = [
    "LLMClient",
    "LLMResponse",
    "OllamaClient",
    "CloudLLMClient",
    "ConfidenceScorer",
    "PrivacyGateway",
    "PIIMatch",
    "RedactionResult",
    "ConsentMode",
    "ConfidenceRouter",
    "RoxyOrchestrator",
    "OrchestratorConfig",
]
