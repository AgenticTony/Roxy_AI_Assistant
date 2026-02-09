"""Roxy Memory System - Three-tier local-first memory architecture.

This package provides Roxy's memory system consisting of three tiers:

1. **SessionMemory (Tier 1):** In-memory current conversation context
2. **ConversationHistory (Tier 2):** SQLite + sqlite-vec persistent storage with semantic search
3. **LongTermMemory (Tier 3):** Mem0-based persistent facts and preferences using Ollama

All memory operations are local-only using Ollama for embeddings. No data
leaves the device.

Classes:
    MemoryManager: Unified interface to all three memory tiers
    SessionMemory: In-memory session context (Tier 1)
    ConversationHistory: Persistent conversation storage with semantic search (Tier 2)
    LongTermMemory: Persistent facts and preferences (Tier 3)

Example:
    >>> from roxy.config import RoxyConfig
    >>> from roxy.memory import MemoryManager
    >>>
    >>> config = RoxyConfig.load()
    >>> memory = MemoryManager(config.memory)
    >>> await memory.initialize()
    >>>
    >>> # Store in session
    >>> await memory.add_to_session("user", "Hello Roxy")
    >>>
    >>> # Search history
    >>> results = await memory.search_history("previous conversations")
    >>>
    >>> # Store long-term
    >>> await memory.remember("user_name", "Anthony")
"""

from __future__ import annotations

from roxy.memory.history import ConversationHistory
from roxy.memory.longterm import LongTermMemory
from roxy.memory.manager import MemoryManager
from roxy.memory.session import SessionMemory

__all__ = [
    "MemoryManager",
    "SessionMemory",
    "ConversationHistory",
    "LongTermMemory",
]
