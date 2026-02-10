"""Roxy orchestrator - Main brain for the AI assistant.

Orchestrates LLM interactions, skill dispatch, and memory management.
"""
# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-statements,broad-exception-caught
# pylint: disable=protected-access

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# pylint: disable=import-error
from agno.agent.agent import Agent  # noqa: I001
from agno.models.openai.chat import OpenAIChat  # noqa: I001

from ..config import RoxyConfig
from ..memory import MemoryManager
from ..skills.base import RoxySkill, SkillContext, SkillResult
from ..skills.registry import SkillRegistry
from .llm_clients import CloudLLMClient, OllamaClient
from .privacy import PrivacyGateway
from .protocols import (
    CloudLLMClientProtocol,
    ConfidenceRouterProtocol,
    LocalLLMClientProtocol,
    MemoryManagerProtocol,
    PrivacyGatewayProtocol,
    SkillRegistryProtocol,
)
from .router import ConfidenceRouter
from .tool_adapter import IntentClassifier, SkillToolAdapter

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

        # Initialize tool adapter for function calling
        self.tool_adapter = SkillToolAdapter(
            skill_registry=self.skill_registry,
            orchestrator_context=self,
        )

        # Initialize intent classifier for fast classification
        self.intent_classifier = IntentClassifier(
            local_client=self.local_client,
            model=config.llm_local.router_model,
        )

        # Agno agent with function calling support
        self.agent = self._create_agent()

        # Conversation state
        self.conversation_history: list[dict] = []

        # Performance tracking
        self._timing: dict[str, list[float]] = {}

        logger.info("Roxy orchestrator initialized")

    def _create_agent(self) -> Agent:
        """Create the Agno agent with appropriate model and tools.

        Returns:
            Configured Agno Agent instance with registered skill tools.
        """
        # Create OpenAIChat model pointing to Ollama
        model = OpenAIChat(
            id=self.config.llm_local.model,
            base_url=f"{self.config.llm_local.host}/v1",
            api_key="ollama",  # Ollama doesn't require a key
        )

        # Get all skill tools for function calling
        tools = self.tool_adapter.get_all_tools()

        # Create agent with tools
        agent = Agent(
            name=self.config.name,
            model=model,
            instructions=[self._get_system_prompt()],
            tools=tools,  # Register skills as Agno tools
        )

        logger.info("Created Agno agent with %s skill tools", len(tools))

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
        Main processing pipeline for user input with Agno function calling.

        This implements the following flow:
        1. Use intent classifier for fast pre-processing
        2. Send request to Agno agent with all skill tools available
        3. If LLM returns a tool_call → execute the corresponding skill
        4. If LLM returns text response (no tool call) → use as general conversation
        5. Store interaction in memory
        6. Update conversation history

        Args:
            user_input: The user's input text.

        Returns:
            Response text to display/speak to user.
        """
        logger.info("Processing user input: %s...", user_input[:100])
        start_time = time.time()

        try:
            # Build memory context for this interaction
            memory_start = time.time()
            _memory_context = await self.memory.build_context_for_llm(user_input)
            memory_elapsed = (time.time() - memory_start) * 1000
            self._track_timing("memory_context_build", memory_elapsed)

            # Fast intent classification with the router model
            classify_start = time.time()
            classification = await self.intent_classifier.classify(user_input)
            classify_elapsed = (time.time() - classify_start) * 1000
            self._track_timing("intent_classification", classify_elapsed)

            logger.debug(
                "Intent: %s (confidence: %.2f)",
                classification['intent'],
                classification['confidence']
            )

            response_text: str

            # Check if intent confidence is high enough for direct skill dispatch
            # (bypassing full LLM function calling for obvious commands)
            if classification['confidence'] >= 0.8:
                intent = classification['intent']
                if intent != 'general_conversation':
                    # Direct skill dispatch for high-confidence matches
                    skill = self.skill_registry.get_skill(intent)
                    if skill:
                        logger.info("Direct skill dispatch: %s", intent)
                        skill_start = time.time()

                        # Create skill context with classified parameters
                        context = SkillContext(
                            user_input=user_input,
                            intent=classification['intent'],
                            parameters=classification['parameters'],
                            memory=self.memory,
                            config=self.config,
                            conversation_history=self.conversation_history.copy(),
                            local_llm_client=self.local_client,
                        )

                        # Inject privacy gateway if needed
                        if hasattr(skill, 'privacy_gateway') and skill.privacy_gateway is None:
                            skill.privacy_gateway = self.privacy

                        # Execute the skill
                        result: SkillResult = await skill._execute_with_hooks(context)

                        skill_elapsed = (time.time() - skill_start) * 1000
                        self._track_timing("skill_execution", skill_elapsed)

                        response_text = result.response_text

                        # Store interaction and continue
                        await self._store_interaction(user_input, response_text)
                        self._update_conversation_history(user_input, response_text)
                        self._trim_history()

                        elapsed = (time.time() - start_time) * 1000
                        self._track_timing("process_total", elapsed)
                        logger.debug("Processed in %.0fms (direct dispatch)", elapsed)

                        return response_text

            # Use Agno agent with function calling for complex/ambiguous requests
            logger.debug("Using Agno agent with function calling")
            agent_start = time.time()

            # Prepare messages for the agent
            messages = [{"role": "user", "content": user_input}]

            # Add conversation history for context
            if self.conversation_history:
                messages = self.conversation_history + messages

            # Run the agent - it will automatically decide whether to use tools
            try:
                # Agno's Agent.run() handles tool calling internally
                response = await self.agent.arun(
                    input=user_input,
                    conversation_history=self.conversation_history.copy(),
                )

                # Extract the response content
                if hasattr(response, 'content'):
                    response_text = response.content
                elif isinstance(response, str):
                    response_text = response
                else:
                    response_text = str(response)

            except Exception as agent_error:
                logger.error("Agno agent error: %s", agent_error)
                # Fall back to direct LLM routing
                logger.info("Falling back to direct LLM routing")
                llm_response = await self.router.route(user_input)
                response_text = llm_response.content

            agent_elapsed = (time.time() - agent_start) * 1000
            self._track_timing("agent_execution", agent_elapsed)

            # Store interaction in memory
            store_start = time.time()
            await self._store_interaction(user_input, response_text)
            store_elapsed = (time.time() - store_start) * 1000
            self._track_timing("memory_store", store_elapsed)

            # Update conversation history
            self._update_conversation_history(user_input, response_text)
            self._trim_history()

            elapsed = (time.time() - start_time) * 1000
            self._track_timing("process_total", elapsed)
            logger.debug("Processed in %.0fms (function calling)", elapsed)

            return response_text

        except Exception as e:
            logger.error("Error processing user input: %s", e, exc_info=True)
            return f"I'm sorry, I encountered an error: {str(e)}"

    def _update_conversation_history(self, user_input: str, response: str) -> None:
        """Update conversation history with new messages.

        Args:
            user_input: User's input text.
            response: Assistant's response text.
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})

    def _trim_history(self) -> None:
        """Trim conversation history to max size."""
        max_history = self.config.memory.session_max_messages
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

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
            logger.error("Error storing interaction: %s", e)

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
                    logger.debug("Stored fact: %s = %s", key, fact)

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string.

        Returns:
            ISO format timestamp.
        """
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

        logger.info("Registered skill: %s", skill.name)

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
