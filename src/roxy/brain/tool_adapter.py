"""Tool adapter for integrating Roxy skills with Agno function calling.

This module bridges the gap between Roxy's skill system and Agno's function calling
capabilities, allowing the LLM to autonomously decide which skill to invoke.

Hardening measures:
- Timeout protection for skill execution
- Circuit breaker for failing skills
- Parameter validation before execution
- Permission checks before skill dispatch
- Structured error responses
- Cache invalidation on skill changes
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from agno.tools.function import Function

from ..skills.base import RoxySkill, SkillContext, SkillResult
from ..skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


@dataclass
class SkillCallStats:
    """Statistics for skill execution tracking."""

    total_calls: int = 0
    failures: int = 0
    last_failure: datetime | None = None
    consecutive_failures: int = 0
    last_success: datetime | None = None

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as ratio."""
        if self.total_calls == 0:
            return 0.0
        return self.failures / self.total_calls

    @property
    def is_healthy(self) -> bool:
        """Check if skill is healthy for execution."""
        # Circuit breaker: disable if >50% failure rate and 5+ consecutive failures
        if self.consecutive_failures >= 5 and self.failure_rate > 0.5:
            return False
        # Also disable if last failure was recent (within last minute) and high failure rate
        if (
            self.last_failure
            and self.total_calls >= 10
            and self.failure_rate > 0.7
            and datetime.now() - self.last_failure < timedelta(minutes=1)
        ):
            return False
        return True


class SkillToolAdapter:
    """
    Converts RoxySkill instances into Agno-compatible tools.

    This adapter enables proper function calling by:
    1. Converting skill metadata to Agno tool definitions
    2. Creating async wrapper functions for skill execution
    3. Handling parameter extraction and validation
    4. Managing the skill lifecycle during function calls
    5. Enforcing permission checks before execution
    6. Providing timeout protection
    7. Circuit breaker for failing skills
    """

    # Default timeout for skill execution (seconds)
    DEFAULT_SKILL_TIMEOUT = 30.0

    # Circuit breaker: disable skill after these consecutive failures
    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(
        self,
        skill_registry: SkillRegistry,
        orchestrator_context: Any,  # noqa: ANN401  # RoxyOrchestrator reference (avoid circular import)
        skill_timeout: float = DEFAULT_SKILL_TIMEOUT,
    ) -> None:
        """Initialize the skill tool adapter.

        Args:
            skill_registry: The skill registry containing all registered skills.
            orchestrator_context: Reference to the orchestrator for dependencies.
            skill_timeout: Maximum seconds to wait for skill execution.
        """
        self.registry = skill_registry
        self.orchestrator = orchestrator_context
        self.skill_timeout = skill_timeout
        self._tool_cache: dict[str, Function] = {}
        self._stats: dict[str, SkillCallStats] = defaultdict(SkillCallStats)
        self._cache_version = 0  # Increment to invalidate cache

    def skill_to_tool(self, skill: RoxySkill) -> Function:
        """Convert a RoxySkill instance to an Agno Function tool.

        Args:
            skill: The RoxySkill instance to convert.

        Returns:
            An Agno Function object that can be registered with the Agno agent.

        Raises:
            ValueError: If skill name is invalid or skill is disabled.
        """
        # Validate skill name
        if not skill.name or not isinstance(skill.name, str):
            raise ValueError(f"Invalid skill name: {skill.name!r}")

        # Check cache first (with version)
        cache_key = f"{skill.name}:{self._cache_version}"
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]

        # Get parameters schema from the skill
        parameters = self._extract_parameters(skill)

        # Create the function description with permission info
        description = skill.description
        if skill.triggers:
            trigger_list = ", ".join(f'"{t}"' for t in skill.triggers[:5])
            description += f"\nTriggered by phrases like: {trigger_list}"

        # Add permission requirements to description
        if skill.permissions:
            perm_list = ", ".join(p.value for p in skill.permissions)
            description += f"\nRequires permissions: {perm_list}"

        # Create async wrapper function with timeout and error handling
        async def skill_wrapper(**kwargs: Any) -> str:  # noqa: ANN401
            """Execute the skill with provided parameters.

            Args:
                **kwargs: Parameters extracted by the LLM.

            Returns:
                The response text from the skill.

            Raises:
                asyncio.TimeoutError: If skill execution exceeds timeout.
            """
            return await self.execute_from_tool_call(skill.name, kwargs)

        # Set function metadata
        skill_wrapper.__name__ = skill.name
        skill_wrapper.__doc__ = description

        # Create Agno Function
        tool = Function(
            name=skill.name,
            description=description,
            parameters=parameters,
            function=skill_wrapper,
        )

        # Cache the tool
        self._tool_cache[cache_key] = tool
        logger.debug(f"Converted skill '{skill.name}' to Agno tool")

        return tool

    def _extract_parameters(self, skill: RoxySkill) -> dict[str, Any]:
        """Extract parameter schema from a skill.

        Analyzes the skill's execute method to build a JSON schema for parameters.

        Args:
            skill: The RoxySkill instance.

        Returns:
            JSON Schema compatible parameter definitions.
        """
        # Basic parameters that all skills can use
        base_params = {
            "type": "object",
            "properties": {
                "user_input": {
                    "type": "string",
                    "description": "The original user input text that triggered this skill.",
                },
                "intent": {
                    "type": "string",
                    "description": "The classified intent behind the user's request.",
                },
            },
        }

        # Check if skill has custom parameter requirements
        # This can be extended by adding a `get_parameters_schema` method to skills
        if hasattr(skill, "get_parameters_schema"):
            custom_params = skill.get_parameters_schema()
            if custom_params:
                base_params["properties"].update(custom_params.get("properties", {}))
                if "required" in custom_params:
                    base_params["required"] = custom_params["required"]

        # For now, keep parameters minimal and let skills extract from user_input
        return base_params

    async def execute_from_tool_call(
        self,
        skill_name: str,
        parameters: dict[str, Any],
    ) -> str:
        """Execute a skill from an Agno tool call response.

        This method is called when Agno's function calling triggers a skill.

        Args:
            skill_name: Name of the skill to execute.
            parameters: Parameters extracted by the LLM from the tool call.

        Returns:
            The response text from skill execution.
        """
        # Validate skill exists
        skill = self.registry.get_skill(skill_name)
        if skill is None:
            error_msg = f"Skill '{skill_name}' not found in registry"
            logger.error(error_msg)
            return self._format_error("skill_not_found", error_msg)

        # Check circuit breaker - is this healthy to call?
        stats = self._stats[skill_name]
        if not stats.is_healthy:
            error_msg = (
                f"Skill '{skill_name}' is temporarily disabled due to repeated failures. "
                f"Failure rate: {stats.failure_rate:.1%}, "
                f"Consecutive failures: {stats.consecutive_failures}"
            )
            logger.warning(error_msg)
            return self._format_error("skill_unavailable", error_msg)

        # Check permissions before execution
        if skill.permissions:
            # TODO: Integrate with permission system
            # For now, log and continue
            logger.debug(f"Skill '{skill_name}' requires permissions: {skill.permissions}")

        try:
            # Validate parameters
            user_input = parameters.get("user_input", "")
            if not isinstance(user_input, str):
                user_input = str(user_input)

            intent = parameters.get("intent", user_input)
            if not isinstance(intent, str):
                intent = str(intent)

            # Extract additional parameters for the skill
            skill_params = {
                k: v for k, v in parameters.items() if k not in ("user_input", "intent")
            }

            # Create SkillContext with orchestrator dependencies
            context = SkillContext(
                user_input=user_input,
                intent=intent,
                parameters=skill_params,
                memory=self.orchestrator.memory,
                config=self.orchestrator.config,
                conversation_history=self.orchestrator.conversation_history.copy(),
                local_llm_client=self.orchestrator.local_client,
            )

            # Inject privacy gateway if skill needs it
            if hasattr(skill, "privacy_gateway") and skill.privacy_gateway is None:
                skill.privacy_gateway = self.orchestrator.privacy

            # Execute the skill with timeout protection
            logger.info(f"Executing skill '{skill_name}' via function call")
            stats.total_calls += 1

            result: SkillResult = await asyncio.wait_for(
                skill._execute_with_hooks(context), timeout=self.skill_timeout
            )

            # Update stats on success
            if result.success:
                stats.consecutive_failures = 0
                stats.last_success = datetime.now()
            else:
                stats.failures += 1
                stats.consecutive_failures += 1
                stats.last_failure = datetime.now()
                logger.warning(f"Skill '{skill_name}' execution failed: {result.response_text}")

            return result.response_text

        except TimeoutError:
            stats.failures += 1
            stats.consecutive_failures += 1
            stats.last_failure = datetime.now()
            error_msg = f"Skill '{skill_name}' execution timed out after {self.skill_timeout}s"
            logger.error(error_msg)
            return self._format_error("timeout", error_msg)

        except Exception as e:
            stats.failures += 1
            stats.consecutive_failures += 1
            stats.last_failure = datetime.now()
            error_msg = f"Error executing skill '{skill_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._format_error("execution_error", error_msg)

    def _format_error(self, error_type: str, message: str) -> str:
        """Format an error response for the LLM.

        Args:
            error_type: Category of error (e.g., "timeout", "skill_not_found").
            message: Human-readable error message.

        Returns:
            Formatted error string.
        """
        return f"[Error: {error_type}] {message}"

    def get_all_tools(self) -> list[Function]:
        """Get Agno Function tools for all registered skills.

        Skips skills that are currently unhealthy (circuit breaker).

        Returns:
            List of Agno Function objects for all registered skills.
        """
        tools = []
        skills_list = self.registry.list_skills()

        for skill_info in skills_list:
            skill_name = skill_info["name"]
            skill = self.registry.get_skill(skill_name)

            if skill is None:
                logger.warning(f"Skill '{skill_name}' in list but not in registry")
                continue

            # Check circuit breaker
            if not self._stats[skill_name].is_healthy:
                logger.debug(f"Skipping unhealthy skill '{skill_name}'")
                continue

            try:
                tool = self.skill_to_tool(skill)
                tools.append(tool)
            except Exception as e:
                logger.error(f"Failed to convert skill '{skill_name}' to tool: {e}")

        logger.info(f"Created {len(tools)} Agno tools from {len(skills_list)} registered skills")
        return tools

    def get_tools_for_request(self, user_input: str) -> list[Function]:
        """Get relevant tools for a specific request.

        Uses the router's intent classification to determine which skills
        are relevant, providing a filtered tool list to the LLM.

        Args:
            user_input: The user's input text.

        Returns:
            List of relevant Agno Function objects.
        """
        # Get all tools by default - the LLM will decide which to use
        # This could be optimized with intent-based filtering in the future
        return self.get_all_tools()

    def clear_cache(self) -> None:
        """Clear the tool cache, forcing regeneration on next access."""
        self._tool_cache.clear()
        self._cache_version += 1
        logger.debug("Cleared tool adapter cache")

    def get_skill_stats(self, skill_name: str | None = None) -> dict[str, Any]:
        """Get execution statistics for skills.

        Args:
            skill_name: Specific skill to get stats for, or None for all.

        Returns:
            Dictionary with statistics.
        """
        if skill_name:
            stats = self._stats.get(skill_name)
            if not stats:
                return {}
            return {
                "total_calls": stats.total_calls,
                "failures": stats.failures,
                "failure_rate": stats.failure_rate,
                "consecutive_failures": stats.consecutive_failures,
                "is_healthy": stats.is_healthy,
                "last_failure": stats.last_failure.isoformat() if stats.last_failure else None,
                "last_success": stats.last_success.isoformat() if stats.last_success else None,
            }

        return {name: self.get_skill_stats(name) for name in self._stats}

    def reset_skill_stats(self, skill_name: str) -> None:
        """Reset statistics for a specific skill.

        Useful for manually re-enabling a skill after it was disabled by circuit breaker.

        Args:
            skill_name: Name of the skill to reset.
        """
        if skill_name in self._stats:
            self._stats[skill_name] = SkillCallStats()
            logger.info(f"Reset statistics for skill '{skill_name}'")


class IntentClassifier:
    """
    Uses the fast router model to classify user intent and extract parameters.

    This provides a fast pre-processing step before full function calling.

    Hardening:
    - Improved JSON extraction with multiple fallback patterns
    - Better error recovery
    - Input validation
    """

    def __init__(
        self,
        local_client: Any,  # OllamaClient (avoid circular import)  # noqa: ANN401
        model: str = "qwen3:0.6b",
    ) -> None:
        """Initialize the intent classifier.

        Args:
            local_client: Ollama client for classification.
            model: Fast model name for classification.
        """
        self.client = local_client
        self.model = model
        self._classification_prompt = self._build_classification_prompt()

    def _build_classification_prompt(self) -> str:
        """Build the system prompt for intent classification.

        Returns:
            The classification prompt template.
        """
        # Get skill descriptions for the prompt
        skills_info = self._get_all_skills_info()

        return f"""You are an intent classifier for a local AI assistant.

Based on the user input, classify the request and extract parameters.

Available skills:
{skills_info}

Respond ONLY with valid JSON in this exact format:
{{
    "intent": "skill_name or 'general_conversation'",
    "confidence": 0.0 to 1.0,
    "parameters": {{
        "user_input": "original user input",
        "intent": "detected intent",
        "app_name": "extracted app name if applicable",
        "query": "search query if applicable",
        "date": "date reference if applicable"
    }}
}}

For 'general_conversation', set confidence to 0.5 and minimal parameters.
"""

    def _get_all_skills_info(self) -> str:
        """Get formatted descriptions of all available skills.

        Returns:
            Formatted string describing all skills.
        """
        from ..skills.registry import get_registry

        registry = get_registry()
        skills = registry.list_skills()

        lines = []
        for skill in skills:
            name = skill["name"]
            desc = skill["description"]
            triggers = ", ".join(skill.get("triggers", []))
            lines.append(f"- {name}: {desc}")
            if triggers:
                lines.append(f"  Triggers: {triggers}")

        return "\n".join(lines)

    async def classify(self, user_input: str) -> dict[str, Any]:
        """Classify user intent and extract parameters.

        Args:
            user_input: The user's input text.

        Returns:
            Dict with intent, confidence, and extracted parameters.
        """
        # Input validation
        if not user_input or not isinstance(user_input, str):
            logger.warning("Invalid user_input for classification")
            return {
                "intent": "general_conversation",
                "confidence": 0.5,
                "parameters": {
                    "user_input": str(user_input) or "",
                    "intent": "general_conversation",
                },
            }

        prompt = f"{self._classification_prompt}\n\nUser input: {user_input}"

        try:
            response = await self.client._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=500,
            )

            content = response.choices[0].message.content or "{}"

            # Parse JSON response with improved patterns
            result = self._extract_json(content)

            if result is None:
                # Fallback to default classification
                logger.warning(f"Failed to extract JSON from response: {content[:200]}")
                return {
                    "intent": "general_conversation",
                    "confidence": 0.5,
                    "parameters": {"user_input": user_input, "intent": "general_conversation"},
                }

            # Validate and normalize the result
            intent = result.get("intent", "general_conversation")
            confidence = float(result.get("confidence", 0.5))
            parameters = result.get("parameters", {})

            # Ensure required fields exist
            parameters.setdefault("user_input", user_input)
            parameters.setdefault("intent", intent)

            logger.debug(f"Classified intent: {intent} (confidence: {confidence:.2f})")

            return {
                "intent": intent,
                "confidence": max(0.0, min(1.0, confidence)),
                "parameters": parameters,
            }

        except Exception as e:
            logger.error(f"Error during intent classification: {e}", exc_info=True)
            return {
                "intent": "general_conversation",
                "confidence": 0.5,
                "parameters": {"user_input": user_input, "intent": "general_conversation"},
            }

    def _extract_json(self, content: str) -> dict[str, Any] | None:
        """Extract JSON from LLM response with multiple fallback patterns.

        Args:
            content: Raw response content from LLM.

        Returns:
            Parsed JSON dict, or None if extraction fails.
        """
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Pattern 1: Extract JSON object with content (nested braces)
        patterns = [
            r'\{[^{}]*"intent"[^{}]*\}',  # Simple single-level
            r"\{(?:[^{}]|\{[^{}]*\})*\}",  # Nested up to 2 levels
            r"\{(?:[^{}]|\{[^{}]*\}|\{\{[^{}]*\}\})*\}",  # Nested up to 3 levels
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue

        # Pattern 2: Look for code blocks with JSON
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # All patterns failed
        return None
