"""Tests for SessionMemory (Tier 1 of memory system)."""

from __future__ import annotations

import asyncio

import pytest

from roxy.memory.session import Message, SessionMemory


class TestMessage:
    """Tests for the Message dataclass."""

    def test_to_dict(self) -> None:
        """Test Message converts to dictionary correctly."""
        msg = Message(role="user", content="Hello", timestamp=1234567890.0)
        result = msg.to_dict()

        assert result == {
            "role": "user",
            "content": "Hello",
            "timestamp": 1234567890.0,
        }

    def test_from_dict(self) -> None:
        """Test Message creates from dictionary correctly."""
        data = {"role": "assistant", "content": "Hi there!", "timestamp": 1234567891.0}
        msg = Message.from_dict(data)

        assert msg.role == "assistant"
        assert msg.content == "Hi there!"
        assert msg.timestamp == 1234567891.0

    def test_roundtrip(self) -> None:
        """Test Message to_dict and from_dict roundtrip."""
        original = Message(role="user", content="Test message", timestamp=1234567892.0)
        data = original.to_dict()
        restored = Message.from_dict(data)

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.timestamp == original.timestamp


class TestSessionMemoryInit:
    """Tests for SessionMemory initialization."""

    @pytest.mark.asyncio
    async def test_init_default(self) -> None:
        """Test initialization with default max_messages."""
        session = SessionMemory()
        assert session.max_messages == 50
        assert session.message_count == 0

    @pytest.mark.asyncio
    async def test_init_custom_max(self) -> None:
        """Test initialization with custom max_messages."""
        session = SessionMemory(max_messages=100)
        assert session.max_messages == 100
        assert session.message_count == 0


class TestSessionMemoryAddMessage:
    """Tests for adding messages to session."""

    @pytest.mark.asyncio
    async def test_add_user_message(self) -> None:
        """Test adding a user message."""
        session = SessionMemory()
        await session.add_message("user", "Hello Roxy")

        messages = await session.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello Roxy"
        assert "timestamp" in messages[0]

    @pytest.mark.asyncio
    async def test_add_assistant_message(self) -> None:
        """Test adding an assistant message."""
        session = SessionMemory()
        await session.add_message("assistant", "Hi there!")

        messages = await session.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_add_multiple_messages(self) -> None:
        """Test adding multiple messages in sequence."""
        session = SessionMemory()
        await session.add_message("user", "First")
        await session.add_message("assistant", "Second")
        await session.add_message("user", "Third")

        messages = await session.get_messages()
        assert len(messages) == 3
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"
        assert messages[2]["content"] == "Third"

    @pytest.mark.asyncio
    async def test_add_invalid_role(self) -> None:
        """Test that invalid role raises ValueError."""
        session = SessionMemory()

        with pytest.raises(ValueError, match="Invalid role"):
            await session.add_message("invalid", "content")


class TestSessionMemoryGetMessages:
    """Tests for retrieving messages from session."""

    @pytest.mark.asyncio
    async def test_get_messages_empty(self) -> None:
        """Test getting messages from empty session."""
        session = SessionMemory()
        messages = await session.get_messages()

        assert messages == []

    @pytest.mark.asyncio
    async def test_get_messages_all(self) -> None:
        """Test getting all messages."""
        session = SessionMemory(max_messages=10)
        for i in range(5):
            await session.add_message("user", f"Message {i}")

        messages = await session.get_messages()
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_get_messages_with_limit(self) -> None:
        """Test getting messages with a limit."""
        session = SessionMemory(max_messages=10)
        for i in range(5):
            await session.add_message("user", f"Message {i}")

        messages = await session.get_messages(limit=3)
        assert len(messages) == 3
        # Should get the last 3 messages
        assert messages[0]["content"] == "Message 2"
        assert messages[1]["content"] == "Message 3"
        assert messages[2]["content"] == "Message 4"

    @pytest.mark.asyncio
    async def test_get_messages_limit_exceeds_count(self) -> None:
        """Test getting messages when limit exceeds count."""
        session = SessionMemory()
        await session.add_message("user", "Only message")

        messages = await session.get_messages(limit=10)
        assert len(messages) == 1


class TestSessionMemoryTruncation:
    """Tests for automatic truncation at max_messages."""

    @pytest.mark.asyncio
    async def test_truncation_at_max(self) -> None:
        """Test that oldest messages are removed when exceeding max."""
        session = SessionMemory(max_messages=3)

        await session.add_message("user", "First")
        await session.add_message("assistant", "Second")
        await session.add_message("user", "Third")
        assert session.message_count == 3

        # This should remove "First"
        await session.add_message("assistant", "Fourth")

        messages = await session.get_messages()
        assert len(messages) == 3
        assert messages[0]["content"] == "Second"
        assert messages[1]["content"] == "Third"
        assert messages[2]["content"] == "Fourth"

    @pytest.mark.asyncio
    async def test_fifo_eviction(self) -> None:
        """Test that eviction follows FIFO (oldest first)."""
        session = SessionMemory(max_messages=5)

        # Add 10 messages
        for i in range(10):
            await session.add_message("user", f"Message {i}")

        messages = await session.get_messages()
        assert len(messages) == 5
        # Should have messages 5-9 (oldest 0-4 evicted)
        for i, msg in enumerate(messages):
            assert msg["content"] == f"Message {i + 5}"


class TestSessionMemoryClear:
    """Tests for clearing session."""

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        """Test clearing all messages."""
        session = SessionMemory()
        await session.add_message("user", "First")
        await session.add_message("assistant", "Second")

        assert session.message_count == 2

        await session.clear()

        messages = await session.get_messages()
        assert messages == []
        assert session.message_count == 0


class TestSessionMemoryContextWindow:
    """Tests for context window calculation."""

    @pytest.mark.asyncio
    async def test_context_window_empty(self) -> None:
        """Test context window with empty session."""
        session = SessionMemory()
        context = await session.get_context_window()

        assert context == []

    @pytest.mark.asyncio
    async def test_context_window_within_limit(self) -> None:
        """Test context window when all messages fit."""
        session = SessionMemory()
        await session.add_message("user", "Short message")

        context = await session.get_context_window(token_limit=1000)

        assert len(context) == 1
        assert context[0]["content"] == "Short message"

    @pytest.mark.asyncio
    async def test_context_window_exceeds_limit(self) -> None:
        """Test context window truncates at token limit."""
        session = SessionMemory()

        # Add messages with known lengths
        await session.add_message("user", "A" * 100)  # ~25 tokens
        await session.add_message("assistant", "B" * 100)  # ~25 tokens
        await session.add_message("user", "C" * 100)  # ~25 tokens

        # Limit to ~50 tokens (200 chars)
        context = await session.get_context_window(token_limit=50)

        # Should include last two messages (most recent)
        assert len(context) == 2
        assert context[0]["content"] == "B" * 100
        assert context[1]["content"] == "C" * 100

    @pytest.mark.asyncio
    async def test_context_window_keep_newest(self) -> None:
        """Test that context window keeps newest messages."""
        session = SessionMemory()

        for i in range(10):
            await session.add_message("user", f"Message {i} " + "X" * 50)

        # Limit should keep newest messages
        context = await session.get_context_window(token_limit=200)

        # Verify we got the most recent messages
        assert len(context) > 0
        # Last message should always be included
        assert context[-1]["content"] == "Message 9 " + "X" * 50


class TestSessionMemoryConcurrency:
    """Tests for concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_add(self) -> None:
        """Test thread-safe concurrent message additions."""
        session = SessionMemory(max_messages=100)

        async def add_messages(prefix: str) -> None:
            for i in range(10):
                await session.add_message("user", f"{prefix}-{i}")

        # Run concurrent additions
        await asyncio.gather(
            add_messages("A"),
            add_messages("B"),
            add_messages("C"),
        )

        messages = await session.get_messages()
        assert len(messages) == 30

    @pytest.mark.asyncio
    async def test_concurrent_add_and_read(self) -> None:
        """Test concurrent additions and reads."""
        session = SessionMemory(max_messages=100)

        async def add_messages() -> None:
            for i in range(50):
                await session.add_message("user", f"Message {i}")

        async def read_messages() -> list[list[dict]]:
            results = []
            for _ in range(10):
                messages = await session.get_messages()
                results.append(messages)
                await asyncio.sleep(0.001)  # Small delay
            return results

        # Run concurrently
        _, read_results = await asyncio.gather(
            add_messages(),
            read_messages(),
        )

        # All reads should have completed without error
        assert len(read_results) == 10
        # Final read should have all messages
        assert len(read_results[-1]) == 50

    @pytest.mark.asyncio
    async def test_concurrent_clear(self) -> None:
        """Test concurrent clear operations."""
        session = SessionMemory(max_messages=100)

        async def add_and_clear(task_id: int) -> None:
            for i in range(10):
                await session.add_message("user", f"Task{task_id}-Message{i}")
                if i % 3 == 0:
                    await session.clear()

        await asyncio.gather(
            add_and_clear(1),
            add_and_clear(2),
        )

        # Session should be in a consistent state
        messages = await session.get_messages()
        # We can't predict exact count due to concurrency,
        # but it should be consistent and <= max_messages
        assert len(messages) <= 100
