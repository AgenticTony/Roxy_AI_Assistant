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
from roxy.brain.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitRecord,
    RateLimiterAware,
)
from roxy.brain.router import ConfidenceRouter
from roxy.brain.orchestrator import RoxyOrchestrator, OrchestratorConfig
from roxy.brain.protocols import (
    PrivacyGatewayProtocol,
    LocalLLMClientProtocol,
    CloudLLMClientProtocol,
    ConfidenceRouterProtocol,
    MemoryManagerProtocol,
    SkillRegistryProtocol,
)
from roxy.brain.factory import create_orchestrator, create_orchestrator_sync

__all__ = [
    # LLM clients
    "LLMClient",
    "LLMResponse",
    "OllamaClient",
    "CloudLLMClient",
    "ConfidenceScorer",
    # Privacy
    "PrivacyGateway",
    "PIIMatch",
    "RedactionResult",
    "ConsentMode",
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitRecord",
    "RateLimiterAware",
    # Routing
    "ConfidenceRouter",
    # Orchestrator
    "RoxyOrchestrator",
    "OrchestratorConfig",
    # Protocols for DI
    "PrivacyGatewayProtocol",
    "LocalLLMClientProtocol",
    "CloudLLMClientProtocol",
    "ConfidenceRouterProtocol",
    "MemoryManagerProtocol",
    "SkillRegistryProtocol",
    # Factory functions
    "create_orchestrator",
    "create_orchestrator_sync",
]
