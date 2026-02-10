"""Roxy brain module.

Contains the core orchestration logic for Roxy's AI assistant.
"""

from roxy.brain.factory import create_orchestrator, create_orchestrator_sync
from roxy.brain.llm_clients import (
    CloudLLMClient,
    ConfidenceScorer,
    LLMClient,
    LLMResponse,
    OllamaClient,
)
from roxy.brain.orchestrator import OrchestratorConfig, RoxyOrchestrator
from roxy.brain.privacy import (
    ConsentMode,
    PIIMatch,
    PrivacyGateway,
    RedactionResult,
)
from roxy.brain.protocols import (
    CloudLLMClientProtocol,
    ConfidenceRouterProtocol,
    LocalLLMClientProtocol,
    MemoryManagerProtocol,
    PrivacyGatewayProtocol,
    SkillRegistryProtocol,
)
from roxy.brain.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimiterAware,
    RateLimitRecord,
)
from roxy.brain.router import ConfidenceRouter
from roxy.brain.tool_adapter import IntentClassifier, SkillToolAdapter

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
    # Tool adapter for function calling
    "SkillToolAdapter",
    "IntentClassifier",
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
