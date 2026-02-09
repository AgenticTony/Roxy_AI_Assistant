"""Roxy orchestrator - Main brain for the AI assistant.

Orchestrates LLM interactions, skill dispatch, and memory management.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agno.agent.agent import Agent
from agno.models.openai.chat import OpenAIChat

from ..config import RoxyConfig
from ..skills.base import (
    SkillContext,
    SkillResult,
    RoxySkill,
    Permission,
)
from ..skills.registry import SkillRegistry
from ..memory import MemoryManager
from .llm_clients import OllamaClient, CloudLLMClient, LLMResponse
from .privacy import PrivacyGateway, ConsentMode
from .router import ConfidenceRouter
from .protocols import (
    PrivacyGatewayProtocol,
    LocalLLMClientProtocol,
    CloudLLMClientProtocol,
    ConfidenceRouterProtocol,
    MemoryManagerProtocol,
    SkillRegistryProtocol,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration specific to the orchestrator."""

    use_real_memory: bool = True  # Use real MemoryManager
    max_conversation_history: int = 50
    enable_skill_registry: bool = True  # Skills are now integrated
    skill_confidence_threshold: float = 0.5  # Minimum confidence for skill dispatch


class RoxyOrchestrator:
    """
    Main orchestrator for Roxy's brain.

    Uses Agno framework to manage agent, tools, and conversation flow.
    Routes requests between local and cloud LLMs based on confidence.
    Dispatches to skills when appropriate.

    The orchestrator supports dependency injection for all major dependencies,
    enabling loose coupling and easier testing. When dependencies are not
    provided, they are created from the config for backward compatibility.
    """

    def __init__(
        self,
        config: RoxyConfig,
        skill_registry: SkillRegistryProtocol | None = None,
        local_client: LocalLLMClientProtocol | None = None,
        cloud_client: CloudLLMClientProtocol | None = None,
        privacy: PrivacyGatewayProtocol | None = None,
        router: ConfidenceRouterProtocol | None = None,
        memory: MemoryManagerProtocol | None = None,
    ) -> None:
        """Initialize the orchestrator with dependency injection support.

        Args:
            config: Roxy configuration.
            skill_registry: Optional skill registry. Uses singleton if None.
            local_client: Optional local LLM client. Created from config if None.
            cloud_client: Optional cloud LLM client. Created from config if None.
            privacy: Optional privacy gateway. Created from config if None.
            router: Optional confidence router. Created from dependencies if None.
            memory: Optional memory manager. Created from config if None.

        Note:
            For backward compatibility, when dependencies are not provided,
            they are created from the config. This allows existing code to
            work without changes while enabling dependency injection for
            testing and advanced use cases.
        """
        self.config = config

        # Get or use provided skill registry
        self.skill_registry = skill_registry or SkillRegistry.get_instance()

        # Initialize or inject LLM clients
        self.local_client = local_client or OllamaClient(
            host=config.llm_local.host,
            model=config.llm_local.model,
            router_model=config.llm_local.router_model,
        )
        self.cloud_client = cloud_client or CloudLLMClient(
            provider=config.llm_cloud.provider,
            model=config.llm_cloud.model,
            api_key=config.llm_cloud.api_key,
            base_url=config.llm_cloud.base_url,
        )

        # Initialize or inject privacy gateway (shared with skills)
        self.privacy = privacy or PrivacyGateway(
            redact_patterns=config.privacy.redact_patterns,
            consent_mode=config.privacy.cloud_consent,
            log_path=f"{config.data_dir}/cloud_requests.log",
        )

        # Initialize or inject confidence router
        self.router = router or ConfidenceRouter(
            local_client=self.local_client,
            cloud_client=self.cloud_client,
            privacy=self.privacy,
            confidence_threshold=config.llm_cloud.confidence_threshold,
        )

        # Initialize or inject memory manager
        self.memory = memory or MemoryManager(
            config=config.memory,
            ollama_host=config.llm_local.host,
        )

        # Agno agent (will be enhanced with tools/skills later)
        self.agent = self._create_agent()

        # Conversation state
        self.conversation_history: list[dict] = []

        # Performance tracking
        self._timing: dict[str, list[float]] = {}

        logger.info("Roxy orchestrator initialized")

    def _create_agent(self) -> Agent:
        """Create the Agno agent with appropriate model.

        Returns:
            Configured Agno Agent instance.
        """
        # Create OpenAIChat model pointing to Ollama
        model = OpenAIChat(
            id=self.config.llm_local.model,
            base_url=f"{self.config.llm_local.host}/v1",
            api_key="ollama",  # Ollama doesn't require a key
        )

        # Create agent
        agent = Agent(
            name=self.config.name,
            model=model,
            instructions=[self._get_system_prompt()],
        )

        return agent

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI agent.

        Returns:
            System prompt string.
        """
        return f"""You are {self.config.name}, a privacy-first, local-first AI assistant for macOS.

Your capabilities:
- Answer questions using your knowledge
- Help with tasks using available skills
- Remember important information the user shares
- Control macOS applications and system settings
- Search the web when needed

Your principles:
- Privacy first: Never share user data without consent
- Local first: Prefer local processing over cloud
- Helpful and friendly: Be a capable assistant
- Honest about limitations: If you can't do something, say so

When responding:
- Be concise but thorough
- Use markdown for formatting
- If you need cloud access, explain why
- Offer follow-up suggestions when relevant"""

    async def initialize(self) -> None:
        """Initialize the orchestrator and its dependencies.

        This must be called before using the orchestrator to ensure
        the memory system is properly initialized.
        """
        logger.info("Initializing Roxy orchestrator...")
        await self.memory.initialize()
        logger.info("Roxy orchestrator initialization complete")

    async def process(self, user_input: str) -> str:
        """
        Main processing pipeline for user input.

        This implements the following flow:
        1. Try to match a skill from the registry
        2. If skill matches with sufficient confidence, execute it
        3. Otherwise, fall through to LLM processing
        4. Store interaction in memory
        5. Update conversation history

        Args:
            user_input: The user's input text.

        Returns:
            Response text to display/speak to user.
        """
        logger.info(f"Processing user input: {user_input[:100]}...")
        start_time = time.time()

        try:
            # Build memory context for this interaction
            memory_start = time.time()
            _memory_context = await self.memory.build_context_for_llm(user_input)
            memory_elapsed = (time.time() - memory_start) * 1000
            self._track_timing("memory_context_build", memory_elapsed)

            # Try to find a matching skill
            skill_find_start = time.time()
            skill, skill_confidence = self.skill_registry.find_skill(
                intent=user_input,
                parameters={},  # Parameters will be extracted by skill if needed
            )
            skill_find_elapsed = (time.time() - skill_find_start) * 1000
            self._track_timing("skill_find", skill_find_elapsed)

            response_text: str

            # If we found a skill with good confidence, use it
            skill_threshold = 0.5  # Minimum confidence for skill dispatch
            if skill and skill_confidence >= skill_threshold:
                logger.info(f"Using skill: {skill.name} (confidence: {skill_confidence:.2f})")
                skill_start = time.time()

                # Create skill context with all dependencies
                context = SkillContext(
                    user_input=user_input,
                    intent=user_input,  # Use full input as intent for skill
                    parameters={},  # Skills will extract their own parameters
                    memory=self.memory,
                    config=self.config,
                    conversation_history=self.conversation_history.copy(),
                    local_llm_client=self.local_client,
                )

                # Inject privacy gateway into skills that need it
                if hasattr(skill, 'privacy_gateway') and skill.privacy_gateway is None:
                    skill.privacy_gateway = self.privacy

                # Execute the skill with lifecycle hooks
                result: SkillResult = await skill._execute_with_hooks(context)

                skill_elapsed = (time.time() - skill_start) * 1000
                self._track_timing("skill_execution", skill_elapsed)
                logger.debug(f"Skill executed in {skill_elapsed:.0f}ms")

                response_text = result.response_text

                # Store interaction in memory
                store_start = time.time()
                await self._store_interaction(user_input, response_text)
                store_elapsed = (time.time() - store_start) * 1000
                self._track_timing("memory_store", store_elapsed)

            else:
                # No skill matched - use LLM processing
                logger.debug(f"No matching skill (best: {skill.name if skill else 'None'} @ {skill_confidence:.2f}), using LLM")

                # Route to appropriate LLM (local or cloud based on confidence)
                llm_start = time.time()
                llm_response = await self.router.route(user_input)
                llm_elapsed = (time.time() - llm_start) * 1000
                self._track_timing("llm_generate", llm_elapsed)

                # Track local vs cloud specifically
                provider = "local" if llm_response.provider == "local" else "cloud"
                self._track_timing(f"llm_{provider}", llm_elapsed)

                response_text = llm_response.content

                # Store interaction in memory
                store_start = time.time()
                await self._store_interaction(user_input, response_text)
                store_elapsed = (time.time() - store_start) * 1000
                self._track_timing("memory_store", store_elapsed)

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            # Trim history if needed
            trim_start = time.time()
            max_history = self.config.memory.session_max_messages
            if len(self.conversation_history) > max_history:
                self.conversation_history = self.conversation_history[-max_history:]
            trim_elapsed = (time.time() - trim_start) * 1000
            self._track_timing("history_trim", trim_elapsed)

            elapsed = (time.time() - start_time) * 1000
            self._track_timing("process_total", elapsed)
            logger.debug(f"Processed in {elapsed:.0f}ms")

            return response_text

        except Exception as e:
            logger.error(f"Error processing user input: {e}", exc_info=True)
            return f"I'm sorry, I encountered an error: {str(e)}"

    def _track_timing(self, operation: str, duration_ms: float) -> None:
        """Track timing for performance monitoring.

        Args:
            operation: Name of the operation being timed.
            duration_ms: Duration in milliseconds.
        """
        if operation not in self._timing:
            self._timing[operation] = []
        self._timing[operation].append(duration_ms)

        # Keep only last 100 measurements
        if len(self._timing[operation]) > 100:
            self._timing[operation] = self._timing[operation][-100:]

    def get_timing_stats(self) -> dict[str, dict[str, float]]:
        """Get timing statistics for all tracked operations.

        Returns:
            Dict with operation names as keys, each containing:
                - avg: Average duration in ms
                - min: Minimum duration in ms
                - max: Maximum duration in ms
                - count: Number of measurements
        """
        stats = {}
        for operation, timings in self._timing.items():
            if timings:
                stats[operation] = {
                    "avg": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "count": len(timings),
                }
        return stats

    async def _store_interaction(self, user_input: str, response: str) -> None:
        """Store interaction in memory.

        Args:
            user_input: User's input text.
            response: Assistant's response text.
        """
        try:
            # Store in session history
            await self.memory.add_to_session("user", user_input)
            await self.memory.add_to_session("assistant", response)

            # Extract and store any facts the user shared
            # This is a simple placeholder - real implementation would use NLP
            await self._extract_and_store_facts(user_input)

        except Exception as e:
            logger.error(f"Error storing interaction: {e}")

    async def _extract_and_store_facts(self, text: str) -> None:
        """
        Extract and store facts from user input.

        This is a simple implementation. A more sophisticated version
        would use NLP to better identify factual statements.

        Args:
            text: User input text.
        """
        # Simple patterns for fact extraction
        # In production, this would use the LLM to extract structured facts
        fact_patterns = [
            ("my name is", "user_name"),
            ("i live in", "user_location"),
            ("my email is", "user_email"),  # Will be redacted before cloud
            ("remember that", "general_fact"),
        ]

        for pattern, key in fact_patterns:
            if pattern in text.lower():
                # Extract the relevant part
                idx = text.lower().find(pattern)
                fact = text[idx + len(pattern):].strip()

                # Clean up the fact
                fact = fact.rstrip(".!?").strip()

                if fact:
                    await self.memory.remember(key, fact)
                    logger.debug(f"Stored fact: {key} = {fact}")

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string.

        Returns:
            ISO format timestamp.
        """
        from datetime import datetime
        return datetime.now().isoformat()

    def register_skill(self, skill: RoxySkill) -> None:
        """
        Register a skill with the orchestrator and skill registry.

        Skills are registered with the skill_registry for dispatch during
        request processing. The Agno agent integration is handled through
        the skill registry's find_skill method.

        Args:
            skill: Skill instance to register.
        """
        # Register with the skill registry for dispatch
        self.skill_registry.register(skill)

        # Note: Agno agent function calling integration can be added
        # by passing tool functions to the agent during creation or
        # by using the @agno.tools.function decorator. The current
        # implementation uses the skill registry for dispatch which
        # provides more control over skill execution and lifecycle.

        logger.info(f"Registered skill: {skill.name}")

    async def get_memory(self, query: str, limit: int = 5) -> list[str]:
        """
        Retrieve relevant memories.

        Args:
            query: Search query.
            limit: Maximum number of results to return.

        Returns:
            List of relevant memory contents.
        """
        return await self.memory.recall(query, limit=limit)

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get orchestrator statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "conversation_length": len(self.conversation_history),
            "routing_stats": self.router.get_statistics(),
            "timing_stats": self.get_timing_stats(),
            "skills_registered": len(self.skill_registry.list_skills()),
            "config": {
                "local_model": self.config.llm_local.model,
                "cloud_provider": self.config.llm_cloud.provider.value,
                "cloud_model": self.config.llm_cloud.model,
                "confidence_threshold": self.config.llm_cloud.confidence_threshold,
            },
        }

    async def shutdown(self) -> None:
        """Cleanup resources before shutdown."""
        logger.info("Shutting down orchestrator...")
        await self.local_client.close()
        await self.cloud_client.close()
        await self.memory.end_conversation()
        logger.info("Orchestrator shutdown complete")
