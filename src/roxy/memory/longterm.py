"""Long-term persistent memory using Mem0 with Ollama backend.

Provides storage for facts, preferences, and important information across
all sessions using Mem0 configured with local Ollama.

This is Tier 3 of Roxy's three-tier memory system.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from roxy.config import RoxyConfig

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Long-term persistent memory using Mem0 with Ollama backend.

    Stores facts, preferences, and important information across
    all sessions. All operations use local Ollama only.

    Example:
        >>> memory = LongTermMemory("data/mem0")
        >>> await memory.initialize()
        >>> await memory.remember("user_name", "Anthony")
        >>> facts = await memory.recall("what's my name")
    """

    def __init__(
        self,
        data_dir: str | Path,
        ollama_host: str | None = None,
        llm_model: str = "qwen3:8b",
        embed_model: str = "nomic-embed-text",
    ) -> None:
        """Initialize long-term memory.

        Args:
            data_dir: Directory for Mem0 storage
            ollama_host: Ollama API endpoint. Uses config default if None.
            llm_model: Model for memory operations (must be available in Ollama)
            embed_model: Model for embeddings (must support embeddings in Ollama)
        """
        self._data_dir = Path(data_dir)

        # Load config for defaults
        config = RoxyConfig.load()

        # Use provided host or default from config
        self._ollama_host = ollama_host or config.llm_local.host
        self._llm_model = llm_model
        self._embed_model = embed_model

        # Mem0 client (initialized in initialize())
        self._mem0_client: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Mem0 client with Ollama configuration.

        Configures Mem0 to use local Ollama for both LLM and embeddings.
        Must be called before other methods.

        Note:
            This implementation uses a simple in-memory fallback if Mem0
            is not properly installed. A full Mem0 integration will be
            added in a future update.
        """
        if self._initialized:
            return

        # Ensure data directory exists
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Try to import mem0ai
        try:
            from mem0 import Memory

            # Configure Mem0 with Ollama
            config_dict = {
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": self._llm_model,
                        "temperature": 0.7,
                    },
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": self._embed_model,
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "embedding_model_dims": 768,  # nomic-embed-text produces 768-dim vectors
                    },
                },
                "history_db_path": str(self._data_dir / "history.db"),
                "custom_prompt": """
                You are a memory storage assistant for Roxy, an AI assistant.
                Store important facts, preferences, and information that would
                be useful to remember in future conversations.
                """,
            }

            # Add Ollama host configuration
            if self._ollama_host:
                config_dict["llm"]["config"]["ollama_base_url"] = self._ollama_host
                config_dict["embedder"]["config"]["ollama_base_url"] = self._ollama_host

            self._mem0_client = Memory.from_config(config_dict)
            self._use_mem0 = True
            logger.info(f"Mem0 initialized with Ollama at {self._ollama_host}")

        except ImportError:
            # Fallback to simple in-memory storage
            logger.warning("mem0ai not installed, using in-memory fallback")
            self._mem0_client: dict[str, Any] = {}
            self._use_mem0 = False

        self._initialized = True

    async def remember(self, key: str, value: str, category: str = "general") -> None:
        """Store a fact in long-term memory.

        Args:
            key: Identifier for the memory
            value: Content to remember
            category: Category (general, preference, fact, contact, etc.)

        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("LongTermMemory not initialized. Call initialize() first.")

        if self._use_mem0:
            # Use Mem0 for storage
            memory_text = f"[{category}] {key}: {value}"
            self._mem0_client.add(memory_text, user_id="roxy", metadata={"category": category})
            logger.debug(f"Stored in Mem0: {key} = {value}")
        else:
            # Fallback: store in dictionary
            if category not in self._mem0_client:
                self._mem0_client[category] = {}
            self._mem0_client[category][key] = value
            logger.debug(f"Stored in fallback: {key} = {value}")

    async def recall(self, query: str, limit: int = 5) -> list[str]:
        """Retrieve relevant long-term memories.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of relevant memory contents

        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("LongTermMemory not initialized. Call initialize() first.")

        if self._use_mem0:
            # Use Mem0 for semantic search
            try:
                results = self._mem0_client.search(query, user_id="roxy", limit=limit)
                # Handle both dict and string result formats
                formatted_results = []
                for result in results:
                    if isinstance(result, dict):
                        formatted_results.append(result.get("memory", ""))
                    elif isinstance(result, str):
                        formatted_results.append(result)
                    else:
                        formatted_results.append(str(result))
                return formatted_results
            except Exception as e:
                logger.error(f"Mem0 search failed: {e}")
                return []
        else:
            # Fallback: simple key matching
            results = []
            for category, items in self._mem0_client.items():
                for key, value in items.items():
                    if query.lower() in key.lower() or query.lower() in value.lower():
                        results.append(f"{key}: {value}")
                        if len(results) >= limit:
                            return results
            return results

    async def get_user_preferences(self) -> dict[str, str]:
        """Get all stored user preferences.

        Returns:
            Dict of preference key-value pairs

        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("LongTermMemory not initialized. Call initialize() first.")

        if self._use_mem0:
            # Search for preference category memories
            try:
                results = self._mem0_client.search("category:preference", user_id="roxy", limit=100)
                preferences = {}
                for result in results:
                    # Handle both dict and string result formats
                    if isinstance(result, dict):
                        memory = result.get("memory", "")
                    elif isinstance(result, str):
                        memory = result
                    else:
                        memory = str(result)
                    # Parse format "[preference] key: value"
                    if "]: " in memory:
                        key_value = memory.split("]: ", 1)[1]
                        if ": " in key_value:
                            key, value = key_value.split(": ", 1)
                            preferences[key] = value
                return preferences
            except Exception as e:
                logger.error(f"Failed to get preferences: {e}")
                return {}
        else:
            # Fallback: return preferences from dictionary
            return self._mem0_client.get("preference", {})

    async def set_preference(self, key: str, value: str) -> None:
        """Store a user preference.

        Args:
            key: Preference name
            value: Preference value

        Raises:
            RuntimeError: If not initialized
        """
        await self.remember(key, value, category="preference")

    async def get_preference(self, key: str, default: str | None = None) -> str | None:
        """Get a specific user preference.

        Args:
            key: Preference name
            default: Default value if not found

        Returns:
            Preference value or default

        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("LongTermMemory not initialized. Call initialize() first.")

        preferences = await self.get_user_preferences()
        return preferences.get(key, default)

    async def search_memories(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memories with optional category filter.

        Args:
            query: Search query
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of memory dicts with metadata

        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("LongTermMemory not initialized. Call initialize() first.")

        # Modify query to include category filter
        search_query = f"{query} category:{category}" if category else query

        if self._use_mem0:
            try:
                results = self._mem0_client.search(search_query, user_id="roxy", limit=limit)
                formatted_results = []
                for result in results:
                    # Handle both dict and string result formats
                    if isinstance(result, dict):
                        formatted_results.append(
                            {
                                "content": result.get("memory", ""),
                                "metadata": result.get("metadata", {}),
                                "score": result.get("score", 0.0),
                            }
                        )
                    elif isinstance(result, str):
                        formatted_results.append(
                            {
                                "content": result,
                                "metadata": {},
                                "score": 0.0,
                            }
                        )
                    else:
                        formatted_results.append(
                            {
                                "content": str(result),
                                "metadata": {},
                                "score": 0.0,
                            }
                        )
                return formatted_results
            except Exception as e:
                logger.error(f"Memory search failed: {e}")
                return []
        else:
            # Fallback: search in dictionary
            results = []
            search_cats = [category] if category else self._mem0_client.keys()
            for cat in search_cats:
                if cat not in self._mem0_client:
                    continue
                for key, value in self._mem0_client[cat].items():
                    if query.lower() in key.lower() or query.lower() in value.lower():
                        results.append(
                            {
                                "content": f"{key}: {value}",
                                "metadata": {"category": cat},
                                "score": 0.5,
                            }
                        )
                        if len(results) >= limit:
                            return results
            return results

    async def get_all_categories(self) -> list[str]:
        """Get all available memory categories.

        Returns:
            List of category names

        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("LongTermMemory not initialized. Call initialize() first.")

        if self._use_mem0:
            # With Mem0, we'd need to track categories separately
            # For now, return standard categories
            return ["general", "preference", "fact", "contact", "location", "routine"]
        else:
            # Fallback: return keys from dictionary
            return list(self._mem0_client.keys())

    async def clear_category(self, category: str) -> None:
        """Clear all memories in a category.

        Args:
            category: Category to clear

        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("LongTermMemory not initialized. Call initialize() first.")

        if self._use_mem0:
            # With Mem0, we'd search and delete
            try:
                results = self._mem0_client.search(
                    f"category:{category}", user_id="roxy", limit=1000
                )
                for result in results:
                    memory_id = result.get("id")
                    if memory_id:
                        self._mem0_client.delete(memory_id)
                logger.info(f"Cleared {len(results)} memories from category: {category}")
            except Exception as e:
                logger.error(f"Failed to clear category: {e}")
        else:
            # Fallback: clear from dictionary
            if category in self._mem0_client:
                del self._mem0_client[category]
                logger.info(f"Cleared category: {category}")
