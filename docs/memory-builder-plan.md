# Memory Builder Implementation Plan

**Author:** memory-builder agent
**Date:** 2025-02-08
**Status:** Planning Phase - Awaiting Approval

---

## Overview

This document details the implementation plan for Roxy's three-tier memory system. The memory system is a critical component that provides session context, conversation history with vector search, and long-term persistent memory - all operating locally on-device using Ollama.

---

## Architecture

### Three-Tier Memory Model

```
                    MemoryManager (Unified Interface)
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    Tier 1              Tier 2              Tier 3
  SessionMemory    ConversationHistory  LongTermMemory
  (In-Memory)      (SQLite+sqlite-vec)  (Mem0+Ollama)
        │                   │                   │
    Current         Vector Search        Persistent
    Conversation    Over History         Facts/Preferences
```

### Key Design Decisions

1. **Local-only operations:** All memory operations use Ollama for embeddings/storage. No cloud calls.
2. **Async-first:** All public methods are async for non-blocking operations.
3. **Type safety:** Full type hints using Python 3.12+ syntax.
4. **Privacy-first:** Conversation history and long-term memory never leave the device.
5. **Performance targets:** Session context <10ms, history search <100ms, long-term recall <500ms.

---

## File Structure

### Files to Create

```
src/roxy/memory/
├── __init__.py                    # Export MemoryManager
├── session.py                     # Tier 1: In-memory session context
├── history.py                     # Tier 2: SQLite + sqlite-vec conversation store
├── longterm.py                    # Tier 3: Mem0 persistent memory wrapper
└── manager.py                     # Unified MemoryManager interface

config/
└── privacy.yaml                   # Privacy configuration with PII patterns

tests/unit/
├── test_session_memory.py         # SessionMemory tests
├── test_history.py                # ConversationHistory tests
├── test_longterm.py               # LongTermMemory tests
└── test_manager.py                # MemoryManager integration tests

tests/
└── conftest.py                    # Shared pytest fixtures
```

---

## Component Specifications

### 1. session.py - SessionMemory (Tier 1)

**Purpose:** Maintain current conversation context in memory.

**Key Features:**
- Circular buffer with configurable max messages (default: 50)
- Thread-safe operations using asyncio locks
- Message format: `{role: "user|assistant", content: str, timestamp: float}`
- Automatic truncation when exceeding max messages
- Export/import for session persistence (optional)

**Interface:**
```python
class SessionMemory:
    """In-memory session context for the current conversation.

    Provides fast access to recent messages with automatic truncation
    when exceeding the configured maximum.
    """

    def __init__(self, max_messages: int = 50) -> None:
        """Initialize session memory with maximum message limit."""

    async def add_message(self, role: str, content: str) -> None:
        """Add a message to the session.

        Args:
            role: Either "user" or "assistant"
            content: The message content
        """

    async def get_messages(self, limit: int | None = None) -> list[dict]:
        """Get recent messages from the session.

        Args:
            limit: Maximum number of messages to return (None = all)

        Returns:
            List of message dicts with role, content, timestamp
        """

    async def clear(self) -> None:
        """Clear all messages from the session."""

    async def get_context_window(self, token_limit: int = 4096) -> list[dict]:
        """Get messages that fit within a token limit.

        Returns the most recent messages that don't exceed the token
        limit (estimated at ~4 chars per token).
        """
```

**Dependencies:**
- `asyncio` for locks
- `time` for timestamps
- `logging` for diagnostics

---

### 2. history.py - ConversationHistory (Tier 2)

**Purpose:** Persistent storage and semantic search over all conversations.

**Key Features:**
- SQLite database with sqlite-vec extension for vector search
- Stores messages with metadata (timestamp, conversation_id)
- Vector embeddings generated locally via Ollama (nomic-embed-text)
- Semantic search using cosine similarity
- Automatic schema initialization on first use
- Conversation grouping with session IDs

**Database Schema:**
```sql
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at REAL,
    metadata TEXT  -- JSON
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS message_vectors USING vec0(
    embedding FLOAT[768],  -- nomic-embed-text dimension
    message_id TEXT PRIMARY KEY,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);
```

**Interface:**
```python
class ConversationHistory:
    """Persistent conversation storage with semantic search.

    Stores all conversations in SQLite with vector embeddings
    for semantic search using sqlite-vec.
    """

    def __init__(
        self,
        db_path: str | Path = "data/memory.db",
        ollama_host: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text"
    ) -> None:
        """Initialize conversation history.

        Args:
            db_path: Path to SQLite database
            ollama_host: Ollama API endpoint for embeddings
            embed_model: Model to use for embeddings
        """

    async def initialize(self) -> None:
        """Initialize database schema and load sqlite-vec.

        Must be called before other methods. Creates tables if they
        don't exist and loads the sqlite-vec extension.
        """

    async def start_conversation(self, metadata: dict | None = None) -> str:
        """Start a new conversation session.

        Args:
            metadata: Optional metadata to attach to conversation

        Returns:
            Conversation ID (UUID)
        """

    async def end_conversation(self, conversation_id: str) -> None:
        """Mark a conversation as ended.

        Args:
            conversation_id: ID of conversation to end
        """

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str
    ) -> str:
        """Add a message to the conversation history.

        Generates and stores embedding for semantic search.

        Args:
            conversation_id: ID of conversation
            role: "user" or "assistant"
            content: Message content

        Returns:
            Message ID (UUID)
        """

    async def search(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None
    ) -> list[dict]:
        """Semantic search over conversation history.

        Args:
            query: Search query (will be embedded)
            limit: Maximum results to return
            conversation_id: Optional conversation ID to scope search

        Returns:
            List of matching message dicts with similarity scores
        """

    async def get_conversation(self, conversation_id: str) -> dict | None:
        """Get full conversation by ID.

        Args:
            conversation_id: ID of conversation

        Returns:
            Conversation dict with all messages or None if not found
        """

    async def list_conversations(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> list[dict]:
        """List conversations, most recent first.

        Args:
            limit: Maximum conversations to return
            offset: Pagination offset

        Returns:
            List of conversation summaries
        """

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using Ollama.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (768 dims for nomic-embed-text)
        """
```

**Dependencies:**
- `sqlite3` (built-in)
- `sqlite-vec` for vector operations
- `httpx.AsyncClient` for Ollama API
- `uuid` for ID generation
- `json` for metadata serialization

---

### 3. longterm.py - LongTermMemory (Tier 3)

**Purpose:** Persistent storage of facts, preferences, and long-term memories using Mem0.

**Key Features:**
- Mem0 integration with Ollama backend (local-only)
- Categorized memory storage (preferences, facts, conversations)
- Memory recall with relevance scoring
- User preference management
- Memory export/import for backup

**Interface:**
```python
class LongTermMemory:
    """Long-term persistent memory using Mem0 with Ollama backend.

    Stores facts, preferences, and important information across
    all sessions. All operations use local Ollama only.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/mem0",
        ollama_host: str = "http://localhost:11434",
        llm_model: str = "qwen3:8b",
        embed_model: str = "nomic-embed-text"
    ) -> None:
        """Initialize long-term memory.

        Args:
            data_dir: Directory for Mem0 storage
            ollama_host: Ollama API endpoint
            llm_model: Model for memory operations
            embed_model: Model for embeddings
        """

    async def initialize(self) -> None:
        """Initialize Mem0 client with Ollama configuration.

        Must be called before other methods. Sets up the Mem0
        client with local Ollama as the backend.
        """

    async def remember(self, key: str, value: str, category: str = "general") -> None:
        """Store a fact in long-term memory.

        Args:
            key: Identifier for the memory
            value: Content to remember
            category: Category (general, preference, fact, etc.)
        """

    async def recall(self, query: str, limit: int = 5) -> list[str]:
        """Retrieve relevant long-term memories.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of relevant memory contents
        """

    async def get_user_preferences(self) -> dict:
        """Get all stored user preferences.

        Returns:
            Dict of preference key-value pairs
        """

    async def set_preference(self, key: str, value: str) -> None:
        """Store a user preference.

        Args:
            key: Preference name
            value: Preference value
        """

    async def get_preference(self, key: str, default: str | None = None) -> str | None:
        """Get a specific user preference.

        Args:
            key: Preference name
            default: Default value if not found

        Returns:
            Preference value or default
        """

    async def search_memories(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10
    ) -> list[dict]:
        """Search memories with optional category filter.

        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of memory dicts with metadata
        """
```

**Dependencies:**
- `mem0ai` (Mem0 client)
- Configuration with Ollama as LLM and embedder provider

---

### 4. manager.py - MemoryManager (Unified Interface)

**Purpose:** Provide a single unified interface to all three memory tiers.

**Key Features:**
- Combines SessionMemory, ConversationHistory, and LongTermMemory
- Implements the exact interface specified in CLAUDE.md
- Delegates to appropriate tier based on operation
- Conversation lifecycle management
- Context building for LLM prompts

**Interface (Must match CLAUDE.md spec exactly):**
```python
class MemoryManager:
    """Unified interface to Roxy's three-tier memory system.

    Provides access to session context, conversation history,
    and long-term memory through a single interface.
    """

    def __init__(
        self,
        config: MemoryConfig,
        ollama_host: str = "http://localhost:11434"
    ) -> None:
        """Initialize MemoryManager with all three tiers.

        Args:
            config: Memory configuration from RoxyConfig
            ollama_host: Ollama API endpoint
        """

    async def initialize(self) -> None:
        """Initialize all memory tiers.

        Must be called before other methods. Initializes
        ConversationHistory and LongTermMemory (SessionMemory
        requires no initialization).
        """

    # ===== Tier 1: Session Context =====

    async def get_session_context(self) -> list[dict]:
        """Get current conversation messages from session.

        Returns:
            List of message dicts from the current session
        """

    async def add_to_session(self, role: str, content: str) -> None:
        """Add a message to the current session.

        Args:
            role: "user" or "assistant"
            content: Message content
        """

    async def clear_session(self) -> None:
        """Clear the current session context."""

    # ===== Tier 2: Conversation History =====

    async def search_history(self, query: str, limit: int = 5) -> list[dict]:
        """Semantic search over conversation history.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching message dicts with similarity scores
        """

    async def get_conversation_history(self, conversation_id: str) -> dict | None:
        """Get full conversation by ID.

        Args:
            conversation_id: ID of conversation

        Returns:
            Conversation dict or None
        """

    # ===== Tier 3: Long-Term Memory =====

    async def remember(self, key: str, value: str) -> None:
        """Store a fact in long-term memory.

        Args:
            key: Memory identifier
            value: Content to remember
        """

    async def recall(self, query: str) -> list[str]:
        """Retrieve relevant long-term memories.

        Args:
            query: Search query

        Returns:
            List of relevant memory contents
        """

    async def get_user_preferences(self) -> dict:
        """Get all stored user preferences.

        Returns:
            Dict of preference key-value pairs
        """

    async def set_preference(self, key: str, value: str) -> None:
        """Store a user preference.

        Args:
            key: Preference name
            value: Preference value
        """

    # ===== Conversation Lifecycle =====

    async def start_conversation(self) -> str:
        """Start a new conversation session.

        Creates a new conversation ID and initializes tracking.

        Returns:
            New conversation ID
        """

    async def end_conversation(self) -> None:
        """End the current conversation session.

        Saves the current session to history and clears
        the in-memory session context.
        """

    # ===== Context Building =====

    async def build_context_for_llm(self, query: str) -> dict:
        """Build complete context for LLM prompt.

        Combines session context, relevant history, and
        relevant long-term memories.

        Args:
            query: The current user query

        Returns:
            Dict with:
                - session_messages: Current conversation
                - relevant_history: Relevant past conversations
                - relevant_memories: Relevant long-term memories
                - user_preferences: User preferences
        """
```

---

### 5. config/privacy.yaml - Privacy Configuration

**Purpose:** Define PII patterns and privacy rules for memory operations.

**Content:**
```yaml
# Privacy Configuration for Roxy Memory System

# PII Detection Patterns
pii_patterns:
  email:
    regex: '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    redaction: '[EMAIL_REDACTED]'

  phone:
    regex: '(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    redaction: '[PHONE_REDACTED]'

  ssn:
    regex: '\d{3}-\d{2}-\d{4}'
    redaction: '[SSN_REDACTED]'

  credit_card:
    regex: '\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}'
    redaction: '[CARD_REDACTED]'

  address:
    regex: '\d+\s+[\w\s]+(?:Street|St|Ave|Avenue|Rd|Road|Lane|Ln|Drive|Dr|Boulevard|Blvd)[,\s]+[\w\s]+'
    redaction: '[ADDRESS_REDACTED]'

# Memory Storage Rules
memory_rules:
  # Never store these patterns in conversation history
  redact_before_storage: true

  # Sensitive topics to flag
  sensitive_topics:
    - health
    - financial
    - legal
    - password
    - api_key
    - token

# Long-term Memory Categories
categories:
  - general
  - preference
  - fact
  - contact
  - location
  - routine
```

---

### 6. tests/conftest.py - Shared Test Fixtures

**Purpose:** Provide reusable fixtures for all memory tests.

**Content:**
```python
"""Shared fixtures for memory system tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
from httpx import AsyncClient, Response

from roxy.config import MemoryConfig


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


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
async def mock_ollama_client() -> AsyncGenerator[AsyncClient, None]:
    """Mock Ollama client for testing.

    In real tests, this would use httpx_mock to provide
    mock responses for embedding generation.
    """
    # For integration tests, use real Ollama if available
    # For unit tests, mock the responses
    yield AsyncClient()


@pytest.fixture
def sample_messages() -> list[dict]:
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?", "timestamp": 1234567890.0},
        {"role": "assistant", "content": "I'm doing well, thank you!", "timestamp": 1234567891.0},
        {"role": "user", "content": "What's the weather like?", "timestamp": 1234567892.0},
        {"role": "assistant", "content": "I don't have access to weather data.", "timestamp": 1234567893.0},
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Sample embedding vectors for testing."""
    # 768-dimensional vectors (nomic-embed-text size)
    # Simplified example - real embeddings would be different
    return [[0.1] * 768, [0.2] * 768, [0.3] * 768]
```

---

### 7. tests/unit/test_session_memory.py - SessionMemory Tests

**Test Cases:**
1. `test_init_default` - Initialize with default values
2. `test_add_message` - Add user and assistant messages
3. `test_get_messages` - Retrieve all messages
4. `test_get_messages_with_limit` - Retrieve limited messages
5. `test_truncation` - Verify automatic truncation at max_messages
6. `test_clear` - Clear all messages
7. `test_get_context_window` - Verify token limit calculation
8. `test_concurrent_add` - Thread-safety test

---

### 8. tests/unit/test_history.py - ConversationHistory Tests

**Test Cases:**
1. `test_initialize` - Database initialization
2. `test_start_conversation` - Create new conversation
3. `test_end_conversation` - Mark conversation as ended
4. `test_add_message` - Store message with embedding
5. `test_search` - Semantic search over messages
6. `test_search_with_conversation_filter` - Scoped search
7. `test_get_conversation` - Retrieve full conversation
8. `test_list_conversations` - List with pagination
9. `test_generate_embedding` - Ollama embedding generation
10. `test_persistence` - Data persists across reinitialization

---

### 9. tests/unit/test_longterm.py - LongTermMemory Tests

**Test Cases:**
1. `test_initialize` - Mem0 client initialization
2. `test_remember` - Store a fact
3. `test_recall` - Retrieve memories
4. `test_set_preference` - Store preference
5. `test_get_user_preferences` - Retrieve all preferences
6. `test_search_memories` - Search with category filter
7. `test_categorized_storage` - Different memory categories
8. `test_relevance_scoring` - Verify recall ranking

---

### 10. tests/unit/test_manager.py - MemoryManager Tests

**Test Cases:**
1. `test_initialize` - Initialize all three tiers
2. `test_get_session_context` - Session context retrieval
3. `test_add_to_session` - Add to current session
4. `test_search_history` - Search conversation history
5. `test_remember` - Store long-term memory
6. `test_recall` - Retrieve long-term memories
7. `test_get_user_preferences` - Preference retrieval
8. `test_set_preference` - Store preference
9. `test_start_conversation` - Start new conversation
10. `test_end_conversation` - End and save conversation
11. `test_build_context_for_llm` - Complete context building
12. `test_integration_full_conversation` - Full conversation lifecycle

---

## Integration Points

### With brain-architect

The MemoryManager will be imported and used by:

1. **orchestrator.py:** Access memory for context building
   ```python
   from roxy.memory import MemoryManager

   # In orchestrator
   context = await self.memory.build_context_for_llm(user_input)
   ```

2. **router.py:** Use conversation history for routing decisions
   ```python
   history = await self.memory.search_history(query)
   # Use history to determine routing
   ```

3. **privacy.py:** Apply PII redaction to memory storage
   ```python
   # Privacy gateway processes content before storage
   redacted = await self.privacy.redact(content)
   await self.memory.remember(key, redacted)
   ```

### Configuration

MemoryManager receives configuration from RoxyConfig:
- `memory.session_max_messages` → SessionMemory max
- `memory.history_db` → ConversationHistory db_path
- `memory.mem0_*` → LongTermMemory Mem0 config
- `env.ollama_host` → Ollama endpoint for embeddings

---

## Local-Only Guarantee

All memory operations stay on-device:

1. **Embeddings:** Generated via Ollama's `nomic-embed-text` model locally
2. **Storage:** SQLite database stored in `~/roxy/data/memory.db`
3. **Mem0:** Configured with `ollama` as both LLM and embedder provider
4. **No API calls:** No external services contacted during memory operations

**Verification:**
- Unit tests mock Ollama but integration tests use real local Ollama
- No network calls in any memory module
- All paths resolve to local filesystem

---

## Implementation Order

### Phase 1: Foundation (First implementation session)
1. Create `session.py` - Simple in-memory storage
2. Create `tests/unit/test_session_memory.py`
3. Verify SessionMemory passes all tests

### Phase 2: History Storage (Second implementation session)
1. Create `history.py` - SQLite + sqlite-vec
2. Create `tests/unit/test_history.py`
3. Implement Ollama embedding generation
4. Verify ConversationHistory passes all tests

### Phase 3: Long-term Memory (Third implementation session)
1. Create `longterm.py` - Mem0 wrapper
2. Create `tests/unit/test_longterm.py`
3. Configure Mem0 with Ollama backend
4. Verify LongTermMemory passes all tests

### Phase 4: Unified Interface (Fourth implementation session)
1. Create `manager.py` - MemoryManager
2. Create `tests/unit/test_manager.py`
3. Create `config/privacy.yaml`
4. Update `src/roxy/memory/__init__.py` to export MemoryManager
5. Verify all integration tests pass

### Phase 5: Final Verification
1. Run full test suite
2. Verify performance targets (<100ms for search)
3. Document any edge cases or limitations

---

## Dependencies to Add

The following dependencies are already in `pyproject.toml` but need verification:
- `mem0ai>=0.1.0` - Mem0 client
- `chromadb>=0.5.0` - Vector database (if Mem0 needs it)
- `sqlite-vec>=0.1.0` - SQLite vector extension

**Note:** May need to add explicit Ollama client dependency if not covered by `openai` package.

---

## Risks and Mitigations

### Risk 1: Mem0 Ollama Integration
**Risk:** Mem0 may not fully support Ollama as a backend
**Mitigation:** Test Mem0 with Ollama early; implement fallback wrapper if needed

### Risk 2: sqlite-vec Installation
**Risk:** sqlite-vec may require compilation on macOS
**Mitigation:** Use pre-built wheels or document installation steps

### Risk 3: Embedding Performance
**Risk:** Local embeddings may be slower than targeted
**Mitigation:** Implement caching for frequently queried text

### Risk 4: Memory Growth
**Risk:** Unbounded memory database growth
**Mitigation:** Implement periodic cleanup/archival (future enhancement)

---

## Testing Strategy

### Unit Tests
- Mock Ollama API responses
- Use in-memory SQLite databases
- Fast execution (<1s per test file)

### Integration Tests
- Use real local Ollama for embeddings
- Use temporary file-based databases
- Slower but verifies real behavior

### Test Coverage Target
- 90%+ coverage on all memory modules
- All public methods tested
- Error paths covered

---

## Performance Targets

| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Session add/get | <10ms | pytest-benchmark |
| History search | <100ms | pytest-benchmark |
| Long-term recall | <500ms | pytest-benchmark |
| Manager context build | <600ms | pytest-benchmark |

---

## Open Questions

1. **Mem0 Configuration:** Need to verify exact Mem0 configuration syntax for Ollama backend
2. **sqlite-vec Loading:** Need to test sqlite-vec extension loading on macOS Apple Silicon
3. **Memory Export/Import:** Nice-to-have feature for backup - defer to Phase 4?

---

## Sign-Off

**Implementation to begin after:** Plan approval from team-lead

**Expected completion:** 4 implementation sessions

**Deliverables:**
- 4 source files (session.py, history.py, longterm.py, manager.py)
- 1 config file (privacy.yaml)
- 4 test files (test_session_memory.py, test_history.py, test_longterm.py, test_manager.py)
- Updated `src/roxy/memory/__init__.py`
- All tests passing with >90% coverage
