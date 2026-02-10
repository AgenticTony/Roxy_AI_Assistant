"""Conversation history with semantic search using SQLite + sqlite-vec.

Provides persistent storage for all conversations with vector embeddings
for semantic search capabilities.

This is Tier 2 of Roxy's three-tier memory system.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from roxy.config import RoxyConfig


class ConversationHistory:
    """Persistent conversation storage with semantic search.

    Stores all conversations in SQLite with vector embeddings
    for semantic search using sqlite-vec.

    All operations are local-only using Ollama for embeddings.

    Example:
        >>> history = ConversationHistory("data/memory.db")
        >>> await history.initialize()
        >>> conv_id = await history.start_conversation()
        >>> await history.add_message(conv_id, "user", "Hello Roxy")
        >>> results = await history.search("greetings")
    """

    def __init__(
        self,
        db_path: str | Path,
        ollama_host: str | None = None,
        embed_model: str = "nomic-embed-text",
    ) -> None:
        """Initialize conversation history.

        Args:
            db_path: Path to SQLite database (will be created if needed)
            ollama_host: Ollama API endpoint for embeddings. Uses config default if None.
            embed_model: Model to use for embeddings (must support embeddings in Ollama)
        """
        self._db_path = Path(db_path)
        # Use provided host or default from RoxyConfig
        if ollama_host is None:
            config = RoxyConfig.load()
            self._ollama_host = config.llm_local.host
        else:
            self._ollama_host = ollama_host
        self._embed_model = embed_model
        self._conn: sqlite3.Connection | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Initialize database schema and load sqlite-vec.

        Must be called before other methods. Creates tables if they
        don't exist and loads the sqlite-vec extension.

        Raises:
            sqlite3.Error: If database initialization fails
            RuntimeError: If sqlite-vec extension cannot be loaded
        """
        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create connection
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row

        # Try to load sqlite-vec extension
        try:
            # Try loading as a loadable extension
            self._conn.enable_load_extension(True)
            self._conn.load_extension("vec0")
        except (sqlite3.OperationalError, AttributeError):
            # Fallback: vec0 might be built-in or not available
            # We'll try creating the virtual table either way
            pass

        # Create schema
        self._create_schema()

        # Initialize HTTP client for Ollama
        self._http_client = httpx.AsyncClient(timeout=30.0)

    def _create_schema(self) -> None:
        """Create database tables if they don't exist.

        Creates three tables:
        - conversations: Conversation metadata
        - messages: Individual messages
        - message_vectors: Vector embeddings for semantic search
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized")

        cursor = self._conn.cursor()

        # Conversations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                ended_at REAL,
                metadata TEXT
            )
        """
        )

        # Messages table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """
        )

        # Create index on conversation_id for faster queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id)
        """
        )

        # Try to create vector table (requires sqlite-vec)
        try:
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS message_vectors
                USING vec0(
                    embedding FLOAT[768],
                    message_id TEXT PRIMARY KEY
                )
            """
            )
            self._vec_enabled = True
        except sqlite3.OperationalError:
            # Vector search not available, will store embeddings anyway
            self._vec_enabled = False

        self._conn.commit()

    async def start_conversation(self, metadata: dict[str, Any] | None = None) -> str:
        """Start a new conversation session.

        Args:
            metadata: Optional metadata to attach to conversation (will be JSON serialized)

        Returns:
            Conversation ID (UUID string)

        Raises:
            RuntimeError: If database not initialized
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        conv_id = str(uuid.uuid4())
        started_at = datetime.now().timestamp()
        metadata_json = json.dumps(metadata) if metadata else None

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversations (id, started_at, metadata)
            VALUES (?, ?, ?)
        """,
            (conv_id, started_at, metadata_json),
        )
        self._conn.commit()

        return conv_id

    async def end_conversation(self, conversation_id: str) -> None:
        """Mark a conversation as ended.

        Args:
            conversation_id: ID of conversation to end

        Raises:
            RuntimeError: If database not initialized
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        ended_at = datetime.now().timestamp()
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE conversations SET ended_at = ? WHERE id = ?",
            (ended_at, conversation_id),
        )
        self._conn.commit()

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> str:
        """Add a message to the conversation history.

        Generates and stores embedding for semantic search.

        Args:
            conversation_id: ID of conversation
            role: "user" or "assistant"
            content: Message content

        Returns:
            Message ID (UUID string)

        Raises:
            RuntimeError: If database not initialized
            ValueError: If role is invalid
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role!r}. Must be 'user' or 'assistant'.")

        msg_id = str(uuid.uuid4())
        timestamp = datetime.now().timestamp()

        # Store message
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """,
            (msg_id, conversation_id, role, content, timestamp),
        )

        # Generate and store embedding
        embedding = await self._generate_embedding(content)

        if self._vec_enabled:
            # Store in vec0 table for vector search
            cursor.execute(
                """
                INSERT INTO message_vectors (embedding, message_id)
                VALUES (?, ?)
            """,
                (sqlite3.Binary(embedding.tobytes()), msg_id),
            )
        else:
            # Store as JSON in message table (fallback)
            # We'd need to add an embedding column for this
            pass

        self._conn.commit()

        return msg_id

    async def search(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over conversation history.

        Generates embedding for query and finds similar messages.

        Args:
            query: Search query (will be embedded)
            limit: Maximum results to return
            conversation_id: Optional conversation ID to scope search

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
            RuntimeError: If database not initialized
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        if not self._vec_enabled:
            # Fallback to simple text search if vec0 not available
            return await self._text_search(query, limit, conversation_id)

        # Generate query embedding
        query_embedding = await self._generate_embedding(query)

        # Build SQL query
        if conversation_id:
            sql = """
                SELECT
                    m.id, m.conversation_id, m.role, m.content, m.timestamp,
                    distance
                FROM message_vectors v
                JOIN messages m ON v.message_id = m.id
                WHERE m.conversation_id = ?
                ORDER BY v.embedding MATCH ?
                LIMIT ?
            """
            params = (conversation_id, sqlite3.Binary(query_embedding.tobytes()), limit)
        else:
            sql = """
                SELECT
                    m.id, m.conversation_id, m.role, m.content, m.timestamp,
                    distance
                FROM message_vectors v
                JOIN messages m ON v.message_id = m.id
                ORDER BY v.embedding MATCH ?
                LIMIT ?
            """
            params = (sqlite3.Binary(query_embedding.tobytes()), limit)

        cursor = self._conn.cursor()
        cursor.execute(sql, params)

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "id": row["id"],
                    "conversation_id": row["conversation_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "similarity": 1.0 - row["distance"],  # Convert distance to similarity
                }
            )

        return results

    async def _text_search(
        self,
        query: str,
        limit: int = 5,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fallback text search using SQL LIKE.

        Used when sqlite-vec is not available.

        Args:
            query: Search query
            limit: Maximum results
            conversation_id: Optional conversation filter

        Returns:
            List of matching message dicts
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        cursor = self._conn.cursor()

        if conversation_id:
            sql = """
                SELECT id, conversation_id, role, content, timestamp
                FROM messages
                WHERE conversation_id = ? AND content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = (conversation_id, f"%{query}%", limit)
        else:
            sql = """
                SELECT id, conversation_id, role, content, timestamp
                FROM messages
                WHERE content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = (f"%{query}%", limit)

        cursor.execute(sql, params)

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "id": row["id"],
                    "conversation_id": row["conversation_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "similarity": 0.5,  # Default similarity for text search
                }
            )

        return results

    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Get full conversation by ID.

        Args:
            conversation_id: ID of conversation

        Returns:
            Conversation dict with all messages, or None if not found
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        cursor = self._conn.cursor()

        # Get conversation metadata
        cursor.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        conv_row = cursor.fetchone()

        if conv_row is None:
            return None

        # Get messages
        cursor.execute(
            """
            SELECT id, role, content, timestamp
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """,
            (conversation_id,),
        )

        messages = []
        for msg_row in cursor.fetchall():
            messages.append(
                {
                    "id": msg_row["id"],
                    "role": msg_row["role"],
                    "content": msg_row["content"],
                    "timestamp": msg_row["timestamp"],
                }
            )

        return {
            "id": conv_row["id"],
            "started_at": conv_row["started_at"],
            "ended_at": conv_row["ended_at"],
            "metadata": json.loads(conv_row["metadata"]) if conv_row["metadata"] else None,
            "messages": messages,
        }

    async def list_conversations(
        self,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List conversations, most recent first.

        Args:
            limit: Maximum conversations to return
            offset: Pagination offset

        Returns:
            List of conversation summaries (without full message lists)
        """
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT
                id, started_at, ended_at, metadata,
                (SELECT COUNT(*) FROM messages WHERE conversation_id = conversations.id) as message_count
            FROM conversations
            ORDER BY started_at DESC
            LIMIT ? OFFSET ?
        """,
            (limit, offset),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "id": row["id"],
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                    "message_count": row["message_count"],
                }
            )

        return results

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using Ollama.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (768 dims for nomic-embed-text)

        Raises:
            RuntimeError: If HTTP client not initialized or Ollama request fails
        """
        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized. Call initialize() first.")

        try:
            response = await self._http_client.post(
                f"{self._ollama_host}/api/embeddings",
                json={
                    "model": self._embed_model,
                    "prompt": text,
                },
            )
            response.raise_for_status()

            data = response.json()
            if "embedding" not in data:
                raise RuntimeError(f"Unexpected Ollama response: {data}")

            return data["embedding"]

        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

    async def close(self) -> None:
        """Close database connection and HTTP client.

        Should be called when shutting down the application.
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None

        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> ConversationHistory:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
