"""Unified memory manager for Roxy's three-tier memory system.

Provides a single interface to session context, conversation history,
and long-term memory.

This is the main entry point for all memory operations in Roxy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from roxy.config import MemoryConfig, RoxyConfig
from roxy.memory.history import ConversationHistory
from roxy.memory.longterm import LongTermMemory
from roxy.memory.session import SessionMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified interface to Roxy's three-tier memory system.

    Provides access to session context, conversation history,
    and long-term memory through a single interface.

    The MemoryManager implements the exact interface specified in CLAUDE.md:
    - get_session_context(): Get current conversation messages
    - search_history(query, limit): Semantic search over all conversations
    - remember(key, value): Store long-term memory
    - recall(query): Retrieve relevant long-term memories
    - get_user_preferences(): Get stored user preferences
    - start_conversation(): Start a new conversation session
    - end_conversation(): End current conversation session
    - build_context_for_llm(query): Build complete context for LLM prompts

    Example:
        >>> from roxy.config import RoxyConfig
        >>> config = RoxyConfig.load()
        >>> memory = MemoryManager(config.memory)
        >>> await memory.initialize()
        >>> await memory.add_to_session("user", "Hello Roxy")
        >>> await memory.remember("name", "Anthony")
    """

    def __init__(
        self,
        config: MemoryConfig,
        ollama_host: str | None = None,
        use_mem0: bool = True,
    ) -> None:
        """Initialize MemoryManager with all three memory tiers.

        Args:
            config: Memory configuration from RoxyConfig
            ollama_host: Optional Ollama host override. Uses config default if None.
            use_mem0: If False, use in-memory fallback instead of Mem0 (for testing)
        """
        self._config = config

        # Load RoxyConfig for defaults
        roxy_config = RoxyConfig.load()
        self._ollama_host = ollama_host or roxy_config.llm_local.host

        # Initialize tier 1: Session memory (in-memory, no init needed)
        self._session = SessionMemory(max_messages=config.session_max_messages)

        # Initialize tier 2: Conversation history (will be initialized in initialize())
        self._history = ConversationHistory(
            db_path=config.history_db,
            ollama_host=self._ollama_host,
        )

        # Initialize tier 3: Long-term memory (will be initialized in initialize())
        self._longterm = LongTermMemory(
            data_dir=str(Path(config.history_db).parent / "mem0"),
            ollama_host=self._ollama_host,
            llm_model=config.mem0_llm_model,
            embed_model=config.mem0_embedder_model,
            use_mem0=use_mem0,
        )

        # Current conversation tracking
        self._current_conversation_id: str | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all memory tiers.

        Must be called before other methods. Initializes
        ConversationHistory and LongTermMemory (SessionMemory
        requires no initialization).
        """
        if self._initialized:
            return

        logger.info("Initializing MemoryManager...")

        # Initialize conversation history (Tier 2)
        await self._history.initialize()
        logger.info("ConversationHistory initialized")

        # Initialize long-term memory (Tier 3)
        await self._longterm.initialize()
        logger.info("LongTermMemory initialized")

        # Start a new conversation session (skip initialized check)
        await self._start_conversation_internal()

        self._initialized = True
        logger.info("MemoryManager initialization complete")

    async def _start_conversation_internal(self) -> str:
        """Internal method to start conversation without initialized check.

        Used during initialization to avoid circular dependency.
        """
        # End current conversation if exists
        if self._current_conversation_id:
            await self._history.end_conversation(self._current_conversation_id)

        # Start new conversation
        self._current_conversation_id = await self._history.start_conversation()

        # Clear session context
        await self._session.clear()

        logger.info(f"Started new conversation: {self._current_conversation_id}")
        return self._current_conversation_id

    # ===== Tier 1: Session Context =====

    async def get_session_context(self) -> list[dict]:
        """Get current conversation messages from session.

        Returns the current in-memory conversation context.

        Returns:
            List of message dicts from the current session, each containing:
                - role: "user" or "assistant"
                - content: The message text
                - timestamp: Unix timestamp

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        return await self._session.get_messages()

    async def add_to_session(self, role: str, content: str) -> None:
        """Add a message to the current session.

        Args:
            role: "user" or "assistant"
            content: Message content

        Raises:
            RuntimeError: If MemoryManager not initialized
            ValueError: If role is invalid
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        await self._session.add_message(role, content)

        # Also add to conversation history
        if self._current_conversation_id:
            await self._history.add_message(self._current_conversation_id, role, content)

    async def clear_session(self) -> None:
        """Clear the current session context.

        Note: This does not affect the conversation history or long-term memory.
        Only clears the in-memory session context.

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        await self._session.clear()

    # ===== Tier 2: Conversation History =====

    async def search_history(self, query: str, limit: int = 5) -> list[dict]:
        """Semantic search over conversation history.

        Searches across all previous conversations using vector embeddings.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching message dicts with similarity scores:
                [
                    {
                        "id": str,
                        "conversation_id": str,
                        "role": str,
                        "content": str,
                        "timestamp": float,
                        "similarity": float
                    },
                    ...
                ]

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        return await self._history.search(query, limit=limit)

    async def get_conversation_history(self, conversation_id: str) -> dict | None:
        """Get full conversation by ID.

        Args:
            conversation_id: ID of conversation

        Returns:
            Conversation dict with all messages, or None if not found:
                {
                    "id": str,
                    "started_at": float,
                    "ended_at": float | None,
                    "metadata": dict | None,
                    "messages": list[dict]
                }

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        return await self._history.get_conversation(conversation_id)

    async def list_conversations(self, limit: int = 10, offset: int = 0) -> list[dict]:
        """List conversations, most recent first.

        Args:
            limit: Maximum conversations to return
            offset: Pagination offset

        Returns:
            List of conversation summaries:
                [
                    {
                        "id": str,
                        "started_at": float,
                        "ended_at": float | None,
                        "metadata": dict | None,
                        "message_count": int
                    },
                    ...
                ]

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        return await self._history.list_conversations(limit=limit, offset=offset)

    # ===== Tier 3: Long-Term Memory =====

    async def remember(self, key: str, value: str) -> None:
        """Store a fact in long-term memory.

        Args:
            key: Memory identifier
            value: Content to remember

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        await self._longterm.remember(key, value, category="general")

    async def recall(self, query: str, limit: int = 5) -> list[str]:
        """Retrieve relevant long-term memories.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of relevant memory contents

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        return await self._longterm.recall(query, limit=limit)

    async def get_user_preferences(self) -> dict:
        """Get all stored user preferences.

        Returns:
            Dict of preference key-value pairs

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        return await self._longterm.get_user_preferences()

    async def set_preference(self, key: str, value: str) -> None:
        """Store a user preference.

        Args:
            key: Preference name
            value: Preference value

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        await self._longterm.set_preference(key, value)

    async def get_preference(self, key: str, default: str | None = None) -> str | None:
        """Get a specific user preference.

        Args:
            key: Preference name
            default: Default value if not found

        Returns:
            Preference value or default

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        return await self._longterm.get_preference(key, default=default)

    # ===== Conversation Lifecycle =====

    async def start_conversation(self) -> str:
        """Start a new conversation session.

        Creates a new conversation ID and initializes tracking.
        Clears the current session context.

        Returns:
            New conversation ID (UUID string)

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        return await self._start_conversation_internal()

    async def end_conversation(self) -> None:
        """End the current conversation session.

        Saves the current session to history and clears
        the in-memory session context.

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        if self._current_conversation_id:
            await self._history.end_conversation(self._current_conversation_id)
            logger.info(f"Ended conversation: {self._current_conversation_id}")
            self._current_conversation_id = None

        # Clear session context
        await self._session.clear()

    # ===== Context Building =====

    async def build_context_for_llm(self, query: str) -> dict[str, Any]:
        """Build complete context for LLM prompt.

        Combines session context, relevant history, and
        relevant long-term memories.

        Args:
            query: The current user query (used for relevance search)

        Returns:
            Dict with:
                - session_messages: Current conversation messages
                - relevant_history: Relevant past conversations
                - relevant_memories: Relevant long-term memories
                - user_preferences: User preferences
                - current_conversation_id: Current conversation ID

        Raises:
            RuntimeError: If MemoryManager not initialized
        """
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

        # Get session context
        session_messages = await self._session.get_messages()

        # Search relevant history
        relevant_history = await self._history.search(query, limit=3)

        # Recall relevant memories
        relevant_memories = await self._longterm.recall(query, limit=3)

        # Get user preferences
        user_preferences = await self._longterm.get_user_preferences()

        return {
            "session_messages": session_messages,
            "relevant_history": relevant_history,
            "relevant_memories": relevant_memories,
            "user_preferences": user_preferences,
            "current_conversation_id": self._current_conversation_id,
        }

    # ===== Properties =====

    @property
    def current_conversation_id(self) -> str | None:
        """Get the current conversation ID."""
        return self._current_conversation_id

    @property
    def is_initialized(self) -> bool:
        """Check if MemoryManager is initialized."""
        return self._initialized
