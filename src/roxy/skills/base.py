"""Base skill class and dataclasses for Roxy's skill system.

All skills must inherit from RoxySkill and implement the execute() method.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from roxy.config import RoxyConfig


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

    def get_history(self, limit: int = 10) -> list[dict]:
        """Get recent conversation history."""
        return self.conversation_history[-limit:]

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
