"""Tests for ConversationHistory (Tier 2 of memory system)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sqlite3

from roxy.memory.history import ConversationHistory


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create temporary database path."""
    return tmp_path / "test_memory.db"


@pytest.fixture
async def mock_history(temp_db_path: Path) -> ConversationHistory:
    """Create ConversationHistory with mocked Ollama client."""
    history = ConversationHistory(
        db_path=str(temp_db_path),
        ollama_host="http://localhost:11434",
        embed_model="nomic-embed-text",
    )

    # Mock the HTTP client
    history._http_client = MagicMock()

    await history.initialize()
    yield history
    await history.close()


@pytest.fixture
def sample_embedding() -> list[float]:
    """Sample embedding vector for testing."""
    # 768-dimensional vector (nomic-embed-text size)
    return [0.1, 0.2, 0.3] * 256  # 768 values


class TestConversationHistoryInit:
    """Tests for ConversationHistory initialization."""

    @pytest.mark.asyncio
    async def test_init(self, temp_db_path: Path) -> None:
        """Test initialization creates database file."""
        history = ConversationHistory(db_path=str(temp_db_path))
        await history.initialize()
        await history.close()

        assert temp_db_path.exists()

    @pytest.mark.asyncio
    async def test_init_creates_tables(self, temp_db_path: Path) -> None:
        """Test initialization creates all required tables."""
        history = ConversationHistory(db_path=str(temp_db_path))
        await history.initialize()

        # Check tables exist
        cursor = history._conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        assert "conversations" in tables
        assert "messages" in tables

        await history.close()

    @pytest.mark.asyncio
    async def test_init_default_ollama_host(self, temp_db_path: Path) -> None:
        """Test default Ollama host is used when not specified."""
        with patch("roxy.memory.history.RoxyConfig") as mock_config:
            mock_roxy_config = MagicMock()
            mock_roxy_config.llm_local.host = "http://test-host:11434"
            mock_config.load.return_value = mock_roxy_config

            history = ConversationHistory(db_path=str(temp_db_path))
            assert history._ollama_host == "http://test-host:11434"


class TestConversationLifecycle:
    """Tests for conversation management."""

    @pytest.mark.asyncio
    async def test_start_conversation(self, mock_history: ConversationHistory) -> None:
        """Test starting a new conversation."""
        conv_id = await mock_history.start_conversation()

        assert conv_id is not None
        assert len(conv_id) > 0

        # Verify conversation exists
        conv = await mock_history.get_conversation(conv_id)
        assert conv is not None
        assert conv["id"] == conv_id
        assert conv["ended_at"] is None

    @pytest.mark.asyncio
    async def test_start_conversation_with_metadata(
        self, mock_history: ConversationHistory
    ) -> None:
        """Test starting conversation with metadata."""
        metadata = {"topic": "testing", "source": "unit_test"}
        conv_id = await mock_history.start_conversation(metadata=metadata)

        conv = await mock_history.get_conversation(conv_id)
        assert conv["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_end_conversation(self, mock_history: ConversationHistory) -> None:
        """Test ending a conversation."""
        conv_id = await mock_history.start_conversation()
        assert (await mock_history.get_conversation(conv_id))["ended_at"] is None

        await mock_history.end_conversation(conv_id)

        conv = await mock_history.get_conversation(conv_id)
        assert conv["ended_at"] is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent_conversation(
        self, mock_history: ConversationHistory
    ) -> None:
        """Test getting conversation that doesn't exist."""
        result = await mock_history.get_conversation("nonexistent-id")
        assert result is None


class TestMessageOperations:
    """Tests for message storage and retrieval."""

    @pytest.mark.asyncio
    async def test_add_user_message(
        self, mock_history: ConversationHistory, sample_embedding: list[float]
    ) -> None:
        """Test adding a user message."""
        # Mock embedding generation
        mock_history._generate_embedding = AsyncMock(return_value=sample_embedding)

        conv_id = await mock_history.start_conversation()
        msg_id = await mock_history.add_message(conv_id, "user", "Hello Roxy")

        assert msg_id is not None

        # Verify message was stored
        conv = await mock_history.get_conversation(conv_id)
        assert len(conv["messages"]) == 1
        assert conv["messages"][0]["role"] == "user"
        assert conv["messages"][0]["content"] == "Hello Roxy"

    @pytest.mark.asyncio
    async def test_add_assistant_message(
        self, mock_history: ConversationHistory, sample_embedding: list[float]
    ) -> None:
        """Test adding an assistant message."""
        mock_history._generate_embedding = AsyncMock(return_value=sample_embedding)

        conv_id = await mock_history.start_conversation()
        await mock_history.add_message(conv_id, "assistant", "Hi there!")

        conv = await mock_history.get_conversation(conv_id)
        assert len(conv["messages"]) == 1
        assert conv["messages"][0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_add_multiple_messages(
        self, mock_history: ConversationHistory, sample_embedding: list[float]
    ) -> None:
        """Test adding multiple messages to a conversation."""
        mock_history._generate_embedding = AsyncMock(return_value=sample_embedding)

        conv_id = await mock_history.start_conversation()
        await mock_history.add_message(conv_id, "user", "First")
        await mock_history.add_message(conv_id, "assistant", "Second")
        await mock_history.add_message(conv_id, "user", "Third")

        conv = await mock_history.get_conversation(conv_id)
        assert len(conv["messages"]) == 3
        assert conv["messages"][0]["content"] == "First"
        assert conv["messages"][1]["content"] == "Second"
        assert conv["messages"][2]["content"] == "Third"

    @pytest.mark.asyncio
    async def test_add_message_invalid_role(
        self, mock_history: ConversationHistory, sample_embedding: list[float]
    ) -> None:
        """Test adding message with invalid role raises error."""
        mock_history._generate_embedding = AsyncMock(return_value=sample_embedding)

        conv_id = await mock_history.start_conversation()

        with pytest.raises(ValueError, match="Invalid role"):
            await mock_history.add_message(conv_id, "invalid", "content")

    @pytest.mark.asyncio
    async def test_conversation_ordering(
        self, mock_history: ConversationHistory, sample_embedding: list[float]
    ) -> None:
        """Test messages are returned in chronological order."""
        mock_history._generate_embedding = AsyncMock(return_value=sample_embedding)

        conv_id = await mock_history.start_conversation()

        # Add messages with delays to ensure different timestamps
        import time
        for i in range(3):
            await mock_history.add_message(conv_id, "user", f"Message {i}")
            await asyncio.sleep(0.01)  # Small delay

        conv = await mock_history.get_conversation(conv_id)
        timestamps = [msg["timestamp"] for msg in conv["messages"]]
        assert timestamps == sorted(timestamps)


class TestSearch:
    """Tests for semantic and text search."""

    @pytest.mark.asyncio
    async def test_text_search_fallback(
        self, mock_history: ConversationHistory, sample_embedding: list[float]
    ) -> None:
        """Test text search when vector search is disabled."""
        mock_history._generate_embedding = AsyncMock(return_value=sample_embedding)
        mock_history._vec_enabled = False  # Force text search

        conv_id = await mock_history.start_conversation()
        await mock_history.add_message(conv_id, "user", "The weather is sunny today")
        await mock_history.add_message(conv_id, "assistant", "I enjoy sunny weather")
        await mock_history.add_message(conv_id, "user", "What about rain?")

        results = await mock_history.search("weather", limit=5)

        # Should find messages containing "weather"
        assert len(results) > 0
        assert any("weather" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_search_with_conversation_filter(
        self, mock_history: ConversationHistory, sample_embedding: list[float]
    ) -> None:
        """Test search scoped to a specific conversation."""
        mock_history._generate_embedding = AsyncMock(return_value=sample_embedding)
        mock_history._vec_enabled = False  # Use text search

        conv1 = await mock_history.start_conversation()
        await mock_history.add_message(conv1, "user", "Python programming")

        conv2 = await mock_history.start_conversation()
        await mock_history.add_message(conv2, "user", "Java programming")

        # Search only in conv1
        results = await mock_history.search("programming", conversation_id=conv1)

        assert len(results) == 1
        assert results[0]["conversation_id"] == conv1

    @pytest.mark.asyncio
    async def test_search_limit(
        self, mock_history: ConversationHistory, sample_embedding: list[float]
    ) -> None:
        """Test search respects limit parameter."""
        mock_history._generate_embedding = AsyncMock(return_value=sample_embedding)
        mock_history._vec_enabled = False

        conv_id = await mock_history.start_conversation()
        for i in range(10):
            await mock_history.add_message(conv_id, "user", f"test message {i}")

        results = await mock_history.search("test", limit=3)
        assert len(results) <= 3


class TestListConversations:
    """Tests for listing conversations."""

    @pytest.mark.asyncio
    async def test_list_empty(self, mock_history: ConversationHistory) -> None:
        """Test listing when no conversations exist."""
        conversations = await mock_history.list_conversations()
        assert conversations == []

    @pytest.mark.asyncio
    async def test_list_conversations(
        self, mock_history: ConversationHistory
    ) -> None:
        """Test listing conversations."""
        conv1 = await mock_history.start_conversation()
        await mock_history.end_conversation(conv1)

        conv2 = await mock_history.start_conversation()

        conversations = await mock_history.list_conversations()

        assert len(conversations) == 2
        # Most recent first
        assert conversations[0]["id"] == conv2
        assert conversations[1]["id"] == conv1

    @pytest.mark.asyncio
    async def test_list_with_message_count(
        self, mock_history: ConversationHistory, sample_embedding: list[float]
    ) -> None:
        """Test listing includes message count."""
        mock_history._generate_embedding = AsyncMock(return_value=sample_embedding)

        conv_id = await mock_history.start_conversation()
        await mock_history.add_message(conv_id, "user", "First")
        await mock_history.add_message(conv_id, "assistant", "Second")

        conversations = await mock_history.list_conversations()

        assert len(conversations) == 1
        assert conversations[0]["message_count"] == 2

    @pytest.mark.asyncio
    async def test_list_pagination(
        self, mock_history: ConversationHistory
    ) -> None:
        """Test listing with pagination."""
        # Create 5 conversations
        conv_ids = []
        for _ in range(5):
            conv_id = await mock_history.start_conversation()
            conv_ids.append(conv_id)

        # Get first page
        page1 = await mock_history.list_conversations(limit=2, offset=0)
        assert len(page1) == 2

        # Get second page
        page2 = await mock_history.list_conversations(limit=2, offset=2)
        assert len(page2) == 2

        # Pages should be different
        assert page1[0]["id"] != page2[0]["id"]


class TestEmbeddingGeneration:
    """Tests for embedding generation."""

    @pytest.mark.asyncio
    async def test_generate_embedding_success(
        self, mock_history: ConversationHistory, sample_embedding: list[float]
    ) -> None:
        """Test successful embedding generation."""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": sample_embedding}
        mock_response.raise_for_status = MagicMock()

        mock_history._http_client.post = AsyncMock(return_value=mock_response)

        embedding = await mock_history._generate_embedding("test text")

        assert embedding == sample_embedding
        mock_history._http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embedding_http_error(
        self, mock_history: ConversationHistory
    ) -> None:
        """Test embedding generation handles HTTP errors."""
        import httpx

        mock_history._http_client.post = AsyncMock(
            side_effect=httpx.HTTPError("Connection failed")
        )

        with pytest.raises(RuntimeError, match="Failed to generate embedding"):
            await mock_history._generate_embedding("test")

    @pytest.mark.asyncio
    async def test_generate_embedding_invalid_response(
        self, mock_history: ConversationHistory
    ) -> None:
        """Test embedding generation handles invalid responses."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "invalid"}  # No "embedding" key
        mock_response.raise_for_status = MagicMock()

        mock_history._http_client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Unexpected Ollama response"):
            await mock_history._generate_embedding("test")


class TestContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, temp_db_path: Path) -> None:
        """Test using ConversationHistory as async context manager."""
        async with ConversationHistory(db_path=str(temp_db_path)) as history:
            assert history._conn is not None
            assert history._http_client is not None

            # Can perform operations
            conv_id = await history.start_conversation()
            assert conv_id is not None

        # After exiting, connection should be closed
        assert history._conn is None


class TestPersistence:
    """Tests for data persistence."""

    @pytest.mark.asyncio
    async def test_data_persists_across_reinit(
        self, temp_db_path: Path, sample_embedding: list[float]
    ) -> None:
        """Test data persists when reinitializing with same database."""
        # Create and populate
        async with ConversationHistory(db_path=str(temp_db_path)) as history1:
            history1._generate_embedding = AsyncMock(return_value=sample_embedding)
            conv_id = await history1.start_conversation()
            await history1.add_message(conv_id, "user", "Persistent message")

        # Reinitialize and verify
        async with ConversationHistory(db_path=str(temp_db_path)) as history2:
            conv = await history2.get_conversation(conv_id)
            assert conv is not None
            assert len(conv["messages"]) == 1
            assert conv["messages"][0]["content"] == "Persistent message"


# Import for test
import asyncio
