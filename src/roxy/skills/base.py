"""Base skill class and dataclasses for Roxy's skill system.

All skills must inherit from RoxySkill and implement the execute() method.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Protocol, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from roxy.brain.llm_clients import LLMClient
    from roxy.config import RoxyConfig


# Hook types for lifecycle events
class HookType(str, Enum):
    """Types of skill lifecycle hooks."""

    BEFORE_EXECUTE = "before_execute"
    AFTER_EXECUTE = "after_execute"


# Hook function signature - use string for forward reference
HookFunction = Callable[[str, "SkillContext"], None]


# Global hook registry - stores hooks registered for each hook type
class _HookRegistry:
    """Global registry for skill lifecycle hooks.

    Hooks are stored per-skill-name and per-hook-type.
    """

    def __init__(self) -> None:
        self._hooks: dict[tuple[str, HookType], list[HookFunction]] = defaultdict(list)

    def register(self, skill_name: str, hook_type: HookType, hook_fn: HookFunction) -> None:
        """Register a hook function for a specific skill and hook type.

        Args:
            skill_name: Name of the skill ("*" for all skills).
            hook_type: Type of hook to register for.
            hook_fn: Function to call when hook is triggered.
        """
        key = (skill_name, hook_type)
        self._hooks[key].append(hook_fn)
        logger.debug(f"Registered {hook_type.value} hook for skill '{skill_name}': {hook_fn.__name__}")

    def get_hooks(self, skill_name: str, hook_type: HookType) -> list[HookFunction]:
        """Get all hooks for a specific skill and hook type.

        Args:
            skill_name: Name of the skill.
            hook_type: Type of hook to get.

        Returns:
            List of hook functions, including global hooks (registered with "*").
        """
        hooks = []
        # Add global hooks first (registered with "*")
        global_key = ("*", hook_type)
        hooks.extend(self._hooks.get(global_key, []))
        # Add skill-specific hooks
        specific_key = (skill_name, hook_type)
        hooks.extend(self._hooks.get(specific_key, []))
        return hooks

    def clear(self, skill_name: str | None = None) -> None:
        """Clear hooks for a skill or all hooks.

        Args:
            skill_name: Name of skill to clear hooks for. If None, clears all.
        """
        if skill_name is None:
            self._hooks.clear()
        else:
            keys_to_remove = [k for k in self._hooks if k[0] == skill_name]
            for key in keys_to_remove:
                del self._hooks[key]


# Global hook registry instance
_hook_registry = _HookRegistry()


class Permission(str, Enum):
    """Permissions that skills can declare.

    Used for permission checks before skill execution.
    """

    FILESYSTEM = "filesystem"
    NETWORK = "network"
    SHELL = "shell"
    MICROPHONE = "microphone"
    NOTIFICATIONS = "notifications"
    APPLESCRIPT = "applescript"
    CLOUD_LLM = "cloud_llm"
    CALENDAR = "calendar"
    EMAIL = "email"
    CONTACTS = "contacts"


@dataclass
class SkillContext:
    """
    Context passed to every skill execution.

    Contains all information a skill might need to execute its task.
    """

    user_input: str
    intent: str
    parameters: dict[str, Any]
    memory: "MemoryManager"
    config: "RoxyConfig"
    conversation_history: list[dict] = field(default_factory=list)
    local_llm_client: "LLMClient | None" = None

    # Hook execution metadata (populated during execution)
    skill_name: str | None = None
    hook_metadata: dict[str, Any] = field(default_factory=dict)

    def get_history(self, limit: int = 10) -> list[dict]:
        """Get recent conversation history."""
        return self.conversation_history[-limit:]

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def _run_hooks(self, hook_type: HookType) -> None:
        """Run all registered hooks for this skill and hook type.

        Args:
            hook_type: Type of hook to run.
        """
        if self.skill_name is None:
            return

        hooks = _hook_registry.get_hooks(self.skill_name, hook_type)
        for hook_fn in hooks:
            try:
                hook_fn(self.skill_name, self)
            except Exception as e:
                logger.error(f"Hook {hook_fn.__name__} failed: {e}", exc_info=True)


@dataclass
class SkillResult:
    """
    Result returned by every skill execution.

    Provides standardized output format from all skills.
    """

    success: bool
    response_text: str
    data: dict[str, Any] | None = None
    speak: bool = True
    follow_up: str | None = None

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"SkillResult({status}: {self.response_text[:100]}...)"


# MemoryManager Protocol - interface for memory system
class MemoryManager(Protocol):
    """
    Protocol for the three-tier memory system.

    This protocol defines the interface that the memory system must implement.
    The brain module uses this protocol to interact with memory without
    depending on specific implementations.
    """

    async def get_session_context(self) -> list[dict]:
        """Get current conversation context from session memory.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        ...

    async def search_history(self, query: str, limit: int = 5) -> list[dict]:
        """Search conversation history using semantic search.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of relevant conversation entries.
        """
        ...

    async def remember(self, key: str, value: str) -> None:
        """Store a long-term fact in persistent memory.

        Args:
            key: Identifier for the fact.
            value: The fact to remember.
        """
        ...

    async def recall(self, query: str) -> list[str]:
        """Retrieve relevant long-term memories.

        Args:
            query: Search query for memories.

        Returns:
            List of relevant memory contents.
        """
        ...

    async def get_user_preferences(self) -> dict[str, Any]:
        """Get stored user preferences.

        Returns:
            Dictionary of user preferences.
        """
        ...


# Stub implementation for testing before real MemoryManager is ready
class StubMemoryManager:
    """
    In-memory stub for testing before real MemoryManager is available.

    Implements the MemoryManager protocol with simple in-memory storage.
    Will be replaced by real implementation from memory-builder teammate.
    """

    def __init__(self) -> None:
        """Initialize stub memory with empty storage."""
        self._session: list[dict] = []
        self._history: list[dict] = []
        self._longterm: dict[str, str] = {}
        self._preferences: dict[str, Any] = {}

    async def get_session_context(self) -> list[dict]:
        """Get current session context."""
        return self._session.copy()

    async def search_history(self, query: str, limit: int = 5) -> list[dict]:
        """Simple keyword search over history."""
        query_lower = query.lower()
        results = [
            entry for entry in self._history
            if query_lower in str(entry).lower()
        ]
        return results[:limit]

    async def remember(self, key: str, value: str) -> None:
        """Store a fact in memory."""
        self._longterm[key] = value
        logger.debug(f"Stored in memory: {key} = {value[:50]}...")

    async def recall(self, query: str) -> list[str]:
        """Recall facts matching query."""
        query_lower = query.lower()
        results = [
            value for key, value in self._longterm.items()
            if query_lower in key.lower() or query_lower in value.lower()
        ]
        return results

    async def get_user_preferences(self) -> dict[str, Any]:
        """Get user preferences."""
        return self._preferences.copy()

    def add_to_session(self, role: str, content: str) -> None:
        """Add message to session (helper method)."""
        self._session.append({"role": role, "content": content})

    def add_to_history(self, entry: dict) -> None:
        """Add entry to history (helper method)."""
        self._history.append(entry)

    def set_preference(self, key: str, value: Any) -> None:
        """Set user preference (helper method)."""
        self._preferences[key] = value


class RoxySkill(ABC):
    """
    Base class for all Roxy skills.

    Every capability in Roxy must be implemented as a skill that inherits
    from this class. Skills declare their permissions and can be checked
    before execution.
    """

    # Skill metadata - must be overridden by subclasses
    name: str = "base_skill"
    description: str = "A Roxy skill"
    triggers: list[str] = []
    permissions: list[Permission] = []
    requires_cloud: bool = False

    def __init__(self) -> None:
        """Initialize skill."""
        if not self.name:
            raise ValueError(f"{self.__class__.__name__} must define a 'name' attribute")

    def before_execute(self, context: SkillContext) -> None:
        """
        Lifecycle hook called before skill execution.

        Override this method to implement pre-execution logic like:
        - Validation checks
        - Resource initialization
        - Logging
        - Permission checks

        Args:
            context: SkillContext with user input and dependencies.

        Note:
            This is a synchronous method. For async operations,
            use the registered hooks via register_hook() instead.
        """
        pass

    def after_execute(self, context: SkillContext, result: SkillResult) -> SkillResult:
        """
        Lifecycle hook called after skill execution.

        Override this method to implement post-execution logic like:
        - Result transformation
        - Cleanup
        - Metrics collection
        - Response enrichment

        Args:
            context: SkillContext with user input and dependencies.
            result: The SkillResult returned by execute().

        Returns:
            SkillResult (may be modified from the input result).

        Note:
            This is a synchronous method. For async operations,
            use the registered hooks via register_hook() instead.
        """
        return result

    @abstractmethod
    async def execute(self, context: SkillContext) -> SkillResult:
        """
        Execute the skill.

        This method must be implemented by all skills.

        Args:
            context: SkillContext with user input and dependencies.

        Returns:
            SkillResult with response and status.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.execute() must be implemented")

    def can_handle(self, intent: str, parameters: dict) -> float:
        """
        Return confidence 0.0-1.0 that this skill handles the intent.

        Override this method for custom intent matching logic.
        Default implementation checks for keyword matches in triggers.
        Longer, more specific matches get higher confidence.

        Args:
            intent: Classified intent string.
            parameters: Extracted parameters from user input.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not self.triggers:
            return 0.0

        intent_lower = intent.lower()
        best_confidence = 0.0

        for trigger in self.triggers:
            trigger_lower = trigger.lower()
            if trigger_lower in intent_lower:
                # Calculate confidence based on match specificity
                # Longer triggers = more specific = higher confidence
                # Base confidence of 0.5, plus bonus for longer triggers
                match_ratio = len(trigger) / max(len(intent), 1)
                confidence = 0.5 + (match_ratio * 0.4)
                best_confidence = max(best_confidence, confidence)

        return best_confidence

    def check_permissions(self, granted_permissions: set[Permission]) -> bool:
        """
        Check if required permissions are granted.

        Args:
            granted_permissions: Set of permissions the user has granted.

        Returns:
            True if all required permissions are granted.
        """
        return all(p in granted_permissions for p in self.permissions)

    async def _execute_with_hooks(self, context: SkillContext) -> SkillResult:
        """
        Execute the skill with lifecycle hooks.

        This method:
        1. Sets the skill name in context
        2. Calls before_execute hooks (instance method + registered)
        3. Calls execute()
        4. Calls after_execute hooks (instance method + registered)

        Args:
            context: SkillContext with user input and dependencies.

        Returns:
            SkillResult with response and status.
        """
        # Set skill name in context for hooks
        context.skill_name = self.name

        # Run before_execute hooks
        self.before_execute(context)
        context._run_hooks(HookType.BEFORE_EXECUTE)

        # Execute the skill
        result = await self.execute(context)

        # Run after_execute hooks (allow modification)
        result = self.after_execute(context, result)
        context.hook_metadata["result"] = result
        context._run_hooks(HookType.AFTER_EXECUTE)

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


# Public API for hook registration
def register_hook(
    skill_name: str,
    hook_type: HookType,
    hook_fn: HookFunction | None = None,
) -> Callable[[HookFunction], HookFunction] | None:
    """
    Register a lifecycle hook for a skill.

    Can be used as a decorator or as a function.

    Args:
        skill_name: Name of the skill to register hook for. Use "*" for all skills.
        hook_type: Type of hook (BEFORE_EXECUTE or AFTER_EXECUTE).
        hook_fn: Optional hook function. If None, returns a decorator.

    Returns:
        If hook_fn is None, returns a decorator function.
        Otherwise, returns None.

    Examples:
        # As a decorator
        @register_hook("app_launcher", HookType.BEFORE_EXECUTE)
        def log_app_launch(skill_name: str, context: SkillContext) -> None:
            logger.info(f"Launching app: {context.user_input}")

        # As a function
        def log_app_launch(skill_name: str, context: SkillContext) -> None:
            logger.info(f"Launching app: {context.user_input}")
        register_hook("app_launcher", HookType.BEFORE_EXECUTE, log_app_launch)

        # Global hook (runs for all skills)
        @register_hook("*", HookType.AFTER_EXECUTE)
        def track_all_skill_executions(skill_name: str, context: SkillContext) -> None:
            metrics.track_skill_execution(skill_name)
    """
    def decorator(fn: HookFunction) -> HookFunction:
        _hook_registry.register(skill_name, hook_type, fn)
        return fn

    if hook_fn is not None:
        _hook_registry.register(skill_name, hook_type, hook_fn)
        return None
    return decorator


def clear_hooks(skill_name: str | None = None) -> None:
    """
    Clear hooks for a specific skill or all hooks.

    Args:
        skill_name: Name of skill to clear hooks for. If None, clears all hooks.

    Examples:
        # Clear hooks for a specific skill
        clear_hooks("app_launcher")

        # Clear all hooks
        clear_hooks()
    """
    _hook_registry.clear(skill_name)


def get_hooks(skill_name: str, hook_type: HookType) -> list[HookFunction]:
    """
    Get all registered hooks for a skill and hook type.

    Args:
        skill_name: Name of the skill.
        hook_type: Type of hook to get.

    Returns:
        List of hook functions.
    """
    return _hook_registry.get_hooks(skill_name, hook_type)
