"""Session memory for the current conversation.

Provides fast in-memory storage for the current conversation with
automatic truncation when exceeding the configured maximum.

This is Tier 1 of Roxy's three-tier memory system.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Self


@dataclass
class Message:
    """A single message in the conversation.

    Attributes:
        role: Either "user" or "assistant"
        content: The message content
        timestamp: Unix timestamp when the message was added
    """

    role: str
    content: str
    timestamp: float

    def to_dict(self) -> dict[str, str | float]:
        """Convert to dictionary representation."""
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, data: dict[str, str | float]) -> Self:
        """Create Message from dictionary."""
        return cls(
            role=str(data["role"]), content=str(data["content"]), timestamp=float(data["timestamp"])
        )


class SessionMemory:
    """In-memory session context for the current conversation.

    Provides fast access to recent messages with automatic truncation
    when exceeding the configured maximum message limit.

    Thread-safe for concurrent access via asyncio locks.

    Example:
        >>> session = SessionMemory(max_messages=50)
        >>> await session.add_message("user", "Hello Roxy")
        >>> await session.add_message("assistant", "Hi there!")
        >>> messages = await session.get_messages()
        >>> len(messages)
        2
    """

    def __init__(self, max_messages: int = 50) -> None:
        """Initialize session memory with maximum message limit.

        Args:
            max_messages: Maximum number of messages to retain.
                          When exceeded, oldest messages are removed first.
        """
        self._max_messages = max_messages
        self._messages: deque[Message] = deque(maxlen=max_messages)
        self._lock = asyncio.Lock()

    async def add_message(self, role: str, content: str) -> None:
        """Add a message to the session.

        Args:
            role: Either "user" or "assistant"
            content: The message content

        Note:
            If adding this message would exceed max_messages, the oldest
            message is automatically removed (FIFO eviction).
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role!r}. Must be 'user' or 'assistant'.")

        message = Message(role=role, content=content, timestamp=time.time())

        async with self._lock:
            self._messages.append(message)

    async def get_messages(self, limit: int | None = None) -> list[dict]:
        """Get recent messages from the session.

        Args:
            limit: Maximum number of messages to return. None returns all messages.

        Returns:
            List of message dicts, each containing:
                - role: "user" or "assistant"
                - content: The message text
                - timestamp: Unix timestamp
        """
        async with self._lock:
            messages_list = list(self._messages)

        if limit is not None:
            messages_list = messages_list[-limit:]

        return [msg.to_dict() for msg in messages_list]

    async def clear(self) -> None:
        """Clear all messages from the session."""
        async with self._lock:
            self._messages.clear()

    async def get_context_window(self, token_limit: int = 4096) -> list[dict]:
        """Get messages that fit within a token limit.

        Returns the most recent messages that don't exceed the token limit.
        Estimates tokens at ~4 characters per token (rough approximation
        for English text).

        Args:
            token_limit: Maximum number of tokens to include

        Returns:
            List of message dicts that fit within the token limit,
            starting from the most recent and working backwards.
        """
        messages = await self.get_messages()

        # Estimate tokens: ~4 chars per token is a rough approximation
        # This is conservative; actual tokenization varies by model
        char_limit = token_limit * 4

        result: list[dict] = []
        total_chars = 0

        # Iterate from newest to oldest
        for msg in reversed(messages):
            msg_chars = len(msg["content"])
            if total_chars + msg_chars > char_limit and result:
                # Adding this would exceed the limit and we already have messages
                break
            result.insert(0, msg)
            total_chars += msg_chars

        return result

    @property
    def message_count(self) -> int:
        """Get the current number of messages in the session.

        Note:
            This property is not async and provides a snapshot.
            For accuracy in concurrent scenarios, use within a locked context.
        """
        return len(self._messages)

    @property
    def max_messages(self) -> int:
        """Get the configured maximum message limit."""
        return self._max_messages
