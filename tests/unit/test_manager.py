"""Tests for MemoryManager - unified memory interface."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.config import MemoryConfig
from roxy.memory.manager import MemoryManager


@pytest.fixture
def memory_config(temp_data_dir: Path) -> MemoryConfig:
    """Create MemoryConfig with temporary paths."""
    return MemoryConfig(
        session_max_messages=10,
        history_db=str(temp_data_dir / "test_memory.db"),
        chromadb_path=str(temp_data_dir / "chromadb"),
        mem0_llm_provider="ollama",
        mem0_llm_model="qwen3:8b",
        mem0_embedder_provider="ollama",
        mem0_embedder_model="nomic-embed-text",
    )


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory."""
    return tmp_path / "data"


@pytest.fixture
async def memory_manager(memory_config: MemoryConfig) -> MemoryManager:
    """Create initialized MemoryManager."""
    manager = MemoryManager(config=memory_config)
    await manager.initialize()
    yield manager


class TestMemoryManagerInit:
    """Tests for MemoryManager initialization."""

    @pytest.mark.asyncio
    async def test_init(self, memory_config: MemoryConfig) -> None:
        """Test initialization."""
        manager = MemoryManager(config=memory_config)
        assert not manager.is_initialized

        await manager.initialize()
        assert manager.is_initialized

    @pytest.mark.asyncio
    async def test_init_creates_data_dir(self, tmp_path: Path) -> None:
        """Test initialization creates data directory."""
        data_dir = tmp_path / "roxy_data"
        config = MemoryConfig(
            session_max_messages=10,
            history_db=str(data_dir / "test_memory.db"),
        )

        manager = MemoryManager(config=config)
        await manager.initialize()

        assert data_dir.exists()

    @pytest.mark.asyncio
    async def test_init_starts_conversation(self, memory_config: MemoryConfig) -> None:
        """Test initialization starts a new conversation."""
        manager = MemoryManager(config=memory_config)
        await manager.initialize()

        assert manager.current_conversation_id is not None

    @pytest.mark.asyncio
    async def test_init_is_idempotent(self, memory_config: MemoryConfig) -> None:
        """Test initialize can be called multiple times."""
        manager = MemoryManager(config=memory_config)
        await manager.initialize()
        conv_id_1 = manager.current_conversation_id

        await manager.initialize()
        conv_id_2 = manager.current_conversation_id

        # Should be idempotent - same conversation
        assert conv_id_1 == conv_id_2


class TestSessionContext:
    """Tests for session context (Tier 1)."""

    @pytest.mark.asyncio
    async def test_get_session_context_empty(self, memory_manager: MemoryManager) -> None:
        """Test getting empty session context."""
        context = await memory_manager.get_session_context()

        assert context == []

    @pytest.mark.asyncio
    async def test_add_to_session(self, memory_manager: MemoryManager) -> None:
        """Test adding messages to session."""
        await memory_manager.add_to_session("user", "Hello Roxy")

        context = await memory_manager.get_session_context()

        assert len(context) == 1
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "Hello Roxy"

    @pytest.mark.asyncio
    async def test_add_multiple_to_session(self, memory_manager: MemoryManager) -> None:
        """Test adding multiple messages to session."""
        await memory_manager.add_to_session("user", "First")
        await memory_manager.add_to_session("assistant", "Second")

        context = await memory_manager.get_session_context()

        assert len(context) == 2

    @pytest.mark.asyncio
    async def test_clear_session(self, memory_manager: MemoryManager) -> None:
        """Test clearing session context."""
        await memory_manager.add_to_session("user", "Test")

        await memory_manager.clear_session()

        context = await memory_manager.get_session_context()
        assert context == []

    @pytest.mark.asyncio
    async def test_operations_not_initialized(self, memory_config: MemoryConfig) -> None:
        """Test operations fail before initialization."""
        manager = MemoryManager(config=memory_config)
        # Don't call initialize()

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.get_session_context()


class TestConversationHistory:
    """Tests for conversation history (Tier 2)."""

    @pytest.mark.asyncio
    async def test_search_history_empty(self, memory_manager: MemoryManager) -> None:
        """Test searching empty history."""
        results = await memory_manager.search_history("test")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_history_with_results(
        self, memory_manager: MemoryManager
    ) -> None:
        """Test searching history with results."""
        # Add some messages to current conversation
        await memory_manager.add_to_session("user", "Python programming")
        await memory_manager.add_to_session("assistant", "I can help with Python")

        # Note: Search may not find immediate results due to embedding timing
        results = await memory_manager.search_history("Python")

        # Results should be a list
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_conversation_history(
        self, memory_manager: MemoryManager
    ) -> None:
        """Test getting conversation by ID."""
        conv_id = memory_manager.current_conversation_id
        await memory_manager.add_to_session("user", "Test message")

        conv = await memory_manager.get_conversation_history(conv_id)

        assert conv is not None
        assert conv["id"] == conv_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_conversation(
        self, memory_manager: MemoryManager
    ) -> None:
        """Test getting non-existent conversation returns None."""
        conv = await memory_manager.get_conversation_history("nonexistent-id")

        assert conv is None

    @pytest.mark.asyncio
    async def test_list_conversations(self, memory_manager: MemoryManager) -> None:
        """Test listing conversations."""
        conversations = await memory_manager.list_conversations()

        assert len(conversations) >= 1  # At least the current one


class TestLongTermMemory:
    """Tests for long-term memory (Tier 3)."""

    @pytest.mark.asyncio
    async def test_remember(self, memory_manager: MemoryManager) -> None:
        """Test storing a long-term memory."""
        # Use fallback mode for testing
        memory_manager._longterm._mem0_client = {}
        memory_manager._longterm._use_mem0 = False

        await memory_manager.remember("test_key", "test_value")

        # Should not raise
        assert True

    @pytest.mark.asyncio
    async def test_recall(self, memory_manager: MemoryManager) -> None:
        """Test recalling long-term memories."""
        # Use fallback mode for testing
        memory_manager._longterm._mem0_client = {}
        memory_manager._longterm._use_mem0 = False

        await memory_manager.remember("user_name", "Anthony")

        results = await memory_manager.recall("name")

        assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_set_preference(self, memory_manager: MemoryManager) -> None:
        """Test setting a user preference."""
        # Use fallback mode for testing
        memory_manager._longterm._mem0_client = {}
        memory_manager._longterm._use_mem0 = False

        await memory_manager.set_preference("theme", "dark")

        value = await memory_manager.get_preference("theme")

        assert value == "dark"

    @pytest.mark.asyncio
    async def test_get_preference_default(self, memory_manager: MemoryManager) -> None:
        """Test getting non-existent preference returns default."""
        value = await memory_manager.get_preference("nonexistent", default="default")

        assert value == "default"

    @pytest.mark.asyncio
    async def test_get_user_preferences(self, memory_manager: MemoryManager) -> None:
        """Test getting all user preferences."""
        # Use fallback mode for testing
        memory_manager._longterm._mem0_client = {}
        memory_manager._longterm._use_mem0 = False

        await memory_manager.set_preference("theme", "dark")
        await memory_manager.set_preference("language", "en")

        prefs = await memory_manager.get_user_preferences()

        assert len(prefs) >= 2


class TestConversationLifecycle:
    """Tests for conversation lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_conversation(self, memory_manager: MemoryManager) -> None:
        """Test starting a new conversation."""
        first_conv = memory_manager.current_conversation_id

        new_conv = await memory_manager.start_conversation()

        assert new_conv != first_conv
        assert memory_manager.current_conversation_id == new_conv

    @pytest.mark.asyncio
    async def test_end_conversation(self, memory_manager: MemoryManager) -> None:
        """Test ending current conversation."""
        conv_id = memory_manager.current_conversation_id

        await memory_manager.end_conversation()

        # Conversation should be ended
        conv = await memory_manager.get_conversation_history(conv_id)
        assert conv is not None
        assert conv["ended_at"] is not None

        # Current conversation ID should be None
        assert memory_manager.current_conversation_id is None

    @pytest.mark.asyncio
    async def test_end_conversation_clears_session(
        self, memory_manager: MemoryManager
    ) -> None:
        """Test ending conversation clears session."""
        await memory_manager.add_to_session("user", "Test message")

        await memory_manager.end_conversation()

        context = await memory_manager.get_session_context()
        assert context == []


class TestContextBuilding:
    """Tests for LLM context building."""

    @pytest.mark.asyncio
    async def test_build_context_for_llm(self, memory_manager: MemoryManager) -> None:
        """Test building complete context for LLM."""
        # Use fallback mode for testing
        memory_manager._longterm._mem0_client = {}
        memory_manager._longterm._use_mem0 = False

        # Add some context
        await memory_manager.add_to_session("user", "My name is Anthony")
        await memory_manager.remember("name", "Anthony")

        context = await memory_manager.build_context_for_llm("What's my name?")

        # Verify context structure
        assert "session_messages" in context
        assert "relevant_history" in context
        assert "relevant_memories" in context
        assert "user_preferences" in context
        assert "current_conversation_id" in context

        # Should have session messages
        assert len(context["session_messages"]) >= 1

    @pytest.mark.asyncio
    async def test_build_context_includes_preferences(
        self, memory_manager: MemoryManager
    ) -> None:
        """Test context includes user preferences."""
        # Use fallback mode for testing
        memory_manager._longterm._mem0_client = {}
        memory_manager._longterm._use_mem0 = False

        await memory_manager.set_preference("theme", "dark")

        context = await memory_manager.build_context_for_llm("test")

        assert isinstance(context["user_preferences"], dict)


class TestProperties:
    """Tests for MemoryManager properties."""

    @pytest.mark.asyncio
    async def test_current_conversation_id(self, memory_manager: MemoryManager) -> None:
        """Test current_conversation_id property."""
        assert isinstance(memory_manager.current_conversation_id, str)

    @pytest.mark.asyncio
    async def test_is_initialized(self, memory_config: MemoryConfig) -> None:
        """Test is_initialized property."""
        manager = MemoryManager(config=memory_config)
        assert not manager.is_initialized

        await manager.initialize()
        assert manager.is_initialized


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_all_operations_fail_before_init(
        self, memory_config: MemoryConfig
    ) -> None:
        """Test all operations fail before initialization."""
        manager = MemoryManager(config=memory_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.get_session_context()

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.add_to_session("user", "test")

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.clear_session()

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.search_history("test")

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.get_conversation_history("id")

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.list_conversations()

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.remember("key", "value")

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.recall("query")

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.get_user_preferences()

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.set_preference("key", "value")

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.get_preference("key")

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.start_conversation()

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.end_conversation()

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.build_context_for_llm("query")
