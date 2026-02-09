"""Factory for creating RoxyOrchestrator instances with dependency injection.

This module provides factory functions for creating fully configured
orchestrators with all dependencies properly injected. This enables
easy testing and swapping of implementations.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import RoxyConfig
from ..skills.registry import SkillRegistry
from ..memory import MemoryManager
from .orchestrator import RoxyOrchestrator
from .llm_clients import OllamaClient, CloudLLMClient
from .privacy import PrivacyGateway
from .rate_limiter import RateLimiter, RateLimitConfig
from .router import ConfidenceRouter

logger = logging.getLogger(__name__)


async def create_orchestrator(
    config: RoxyConfig,
    skill_registry: SkillRegistry | None = None,
) -> RoxyOrchestrator:
    """
    Create a fully configured RoxyOrchestrator with all dependencies.

    This factory function creates all dependencies from the config
    and injects them into the orchestrator. This is the recommended
    way to create orchestrators in production code.

    Args:
        config: Roxy configuration.
        skill_registry: Optional skill registry. Uses singleton if None.

    Returns:
        Fully configured and initialized RoxyOrchestrator.

    Example:
        >>> from roxy.config import RoxyConfig
        >>> config = RoxyConfig.load()
        >>> orchestrator = await create_orchestrator(config)
    """
    # Get or use provided skill registry
    registry = skill_registry or SkillRegistry.get_instance()

    # Create LLM clients
    local_client = OllamaClient(
        host=config.llm_local.host,
        model=config.llm_local.model,
        router_model=config.llm_local.router_model,
    )
    logger.debug(f"Created local LLM client: {config.llm_local.model}")

    # Create rate limiter with persistent storage
    rate_limiter = RateLimiter(
        config=RateLimitConfig(
            storage_path=str(Path(config.data_dir) / "rate_limits.json"),
        )
    )
    logger.debug(f"Created rate limiter with storage: {Path(config.data_dir) / 'rate_limits.json'}")

    cloud_client = CloudLLMClient(
        provider=config.llm_cloud.provider,
        model=config.llm_cloud.model,
        api_key=config.llm_cloud.api_key,
        base_url=config.llm_cloud.base_url,
        rate_limiter=rate_limiter,
    )
    logger.debug(f"Created cloud LLM client: {config.llm_cloud.provider}/{config.llm_cloud.model}")

    # Create privacy gateway
    privacy = PrivacyGateway(
        redact_patterns=config.privacy.redact_patterns,
        consent_mode=config.privacy.cloud_consent,
        log_path=Path(config.data_dir) / "cloud_requests.log",
    )
    logger.debug(f"Created privacy gateway with consent mode: {config.privacy.cloud_consent}")

    # Create confidence router
    router = ConfidenceRouter(
        local_client=local_client,
        cloud_client=cloud_client,
        privacy=privacy,
        confidence_threshold=config.llm_cloud.confidence_threshold,
    )
    logger.debug(f"Created confidence router with threshold: {config.llm_cloud.confidence_threshold}")

    # Create memory manager
    memory = MemoryManager(
        config=config.memory,
        ollama_host=config.llm_local.host,
    )
    logger.debug("Created memory manager")

    # Create orchestrator with all dependencies injected
    orchestrator = RoxyOrchestrator(
        config=config,
        skill_registry=registry,
        local_client=local_client,
        cloud_client=cloud_client,
        privacy=privacy,
        router=router,
        memory=memory,
    )
    logger.info("Created orchestrator with dependency injection")

    # Initialize the orchestrator
    await orchestrator.initialize()

    return orchestrator


def create_orchestrator_sync(
    config: RoxyConfig,
    skill_registry: SkillRegistry | None = None,
) -> RoxyOrchestrator:
    """
    Synchronous version of create_orchestrator for non-async contexts.

    Note: This still requires an async context to initialize the orchestrator.
    The returned orchestrator must have initialize() called before use.

    Args:
        config: Roxy configuration.
        skill_registry: Optional skill registry. Uses singleton if None.

    Returns:
        RoxyOrchestrator instance (not yet initialized).

    Example:
        >>> from roxy.config import RoxyConfig
        >>> config = RoxyConfig.load()
        >>> orchestrator = create_orchestrator_sync(config)
        >>> await orchestrator.initialize()
    """
    # Get or use provided skill registry
    registry = skill_registry or SkillRegistry.get_instance()

    # Create LLM clients
    local_client = OllamaClient(
        host=config.llm_local.host,
        model=config.llm_local.model,
        router_model=config.llm_local.router_model,
    )

    # Create rate limiter with persistent storage
    rate_limiter = RateLimiter(
        config=RateLimitConfig(
            storage_path=str(Path(config.data_dir) / "rate_limits.json"),
        )
    )

    cloud_client = CloudLLMClient(
        provider=config.llm_cloud.provider,
        model=config.llm_cloud.model,
        api_key=config.llm_cloud.api_key,
        base_url=config.llm_cloud.base_url,
        rate_limiter=rate_limiter,
    )

    # Create privacy gateway
    privacy = PrivacyGateway(
        redact_patterns=config.privacy.redact_patterns,
        consent_mode=config.privacy.cloud_consent,
        log_path=Path(config.data_dir) / "cloud_requests.log",
    )

    # Create confidence router
    router = ConfidenceRouter(
        local_client=local_client,
        cloud_client=cloud_client,
        privacy=privacy,
        confidence_threshold=config.llm_cloud.confidence_threshold,
    )

    # Create memory manager
    memory = MemoryManager(
        config=config.memory,
        ollama_host=config.llm_local.host,
    )

    # Create orchestrator with all dependencies injected
    return RoxyOrchestrator(
        config=config,
        skill_registry=registry,
        local_client=local_client,
        cloud_client=cloud_client,
        privacy=privacy,
        router=router,
        memory=memory,
    )
