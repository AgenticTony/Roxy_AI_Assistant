"""Roxy System MCP Server.

Custom MCP server that exposes Roxy's internal capabilities as tools.
Allows external MCP clients to interact with Roxy's memory, system status,
and local file search.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Try to import FastMCP, fall back to basic implementation if not available
try:
    from fastmcp import FastMCP

    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    logger.warning("FastMCP not available, using placeholder implementation")

if TYPE_CHECKING:
    from collections.abc import Callable


# Create the FastMCP server instance
if HAS_FASTMCP:
    mcp = FastMCP(name="Roxy System")
else:
    # Placeholder for when FastMCP is not installed
    class PlaceholderMCP:
        """Placeholder for FastMCP when not installed."""

        def tool(self, func: Callable) -> Callable:
            """Decorator that does nothing."""
            return func

    mcp = PlaceholderMCP()  # type: ignore[assignment]


@mcp.tool()
async def roxy_search_memory(query: str, limit: int = 5) -> list[str]:
    """Search Roxy's conversation memory for relevant information.

    This tool searches through Roxy's conversation history using semantic
    search to find relevant past discussions and context.

    Args:
        query: The search query to look for in conversation history.
        limit: Maximum number of results to return (default: 5).

    Returns:
        List of relevant conversation excerpts or memory entries.
    """
    logger.info(f"Searching memory for: {query}")

    try:
        # Import here to avoid circular dependency
        from roxy.memory.manager import MemoryManager

        # This would normally be injected via dependency injection
        # For now, we'll return a placeholder response
        # In production, this would call: await memory.recall(query)

        return [
            f"Memory search result for '{query}':",
            "This is a placeholder response.",
            "The actual memory system will be connected by the memory-builder teammate.",
        ]

    except ImportError:
        logger.warning("MemoryManager not available, returning placeholder")
        return [
            f"Memory search for '{query}'",
            "Memory system not yet implemented.",
        ]


@mcp.tool()
async def roxy_system_status() -> dict:
    """Get Roxy's current system status and health information.

    Returns information about:
    - Active skills and their status
    - Memory system health
    - Connected MCP servers
    - System resource usage
    - Current configuration

    Returns:
        Dictionary containing system status information.
    """
    logger.debug("System status requested")

    status = {
        "name": "Roxy",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
        "status": "running",
        "components": {
            "memory": "pending",  # Will be updated by memory-builder
            "voice": "pending",  # Will be updated by voice-engineer
            "skills": "initializing",
            "mcp_servers": "starting",
        },
        "active_skills": [],
        "uptime_seconds": 0,  # Will be calculated from start time
    }

    return status


@mcp.tool()
async def roxy_search_files(
    query: str,
    limit: int = 10,
    path: str | None = None,
) -> list[str]:
    """Search local files using macOS Spotlight.

    Uses mdfind to search for files matching the query.
    Searches are fast and comprehensive, covering indexed files.

    Args:
        query: Search query for Spotlight (supports natural language).
        limit: Maximum number of results to return (default: 10).
        path: Optional path to restrict search (default: all mounted volumes).

    Returns:
        List of file paths matching the search query.
    """
    logger.info(f"Searching files for: {query}")

    try:
        # Build mdfind command
        cmd = ["mdfind", "-limit", str(limit), query]

        if path:
            cmd.extend(["-onlyin", path])

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.error(f"mdfind failed: {result.stderr}")
            return []

        # Parse results
        files = [line.strip() for line in result.stdout.split("\n") if line.strip()]

        logger.info(f"Found {len(files)} files for query: {query}")
        return files[:limit]

    except subprocess.TimeoutExpired:
        logger.error("File search timed out")
        return []
    except Exception as e:
        logger.error(f"File search error: {e}")
        return []


@mcp.tool()
async def roxy_get_user_preferences() -> dict:
    """Get Roxy's stored user preferences and settings.

    Returns user-specific preferences such as:
    - Default locations (home airport, home city)
    - Privacy settings
    - Voice preferences
    - Skill enable/disable settings

    Returns:
        Dictionary containing user preferences.
    """
    logger.debug("User preferences requested")

    # Placeholder - will be connected to actual memory system
    return {
        "home_airport": "CPH",  # Copenhagen
        "home_city": "Malmö, Sweden",
        "privacy": {
            "cloud_consent": "ask",
            "pii_redaction": True,
        },
        "voice": {
            "enabled": True,
            "wake_word": "hey_roxy",
            "tts_voice": "af_heart",
        },
        "skills": {
            "web_search": True,
            "flights": True,
            "calendar": True,
        },
    }


@mcp.tool()
async def roxy_set_preference(key: str, value: str) -> bool:
    """Set a user preference.

    Args:
        key: Preference key (e.g., 'home_airport', 'voice_enabled').
        value: Preference value to set.

    Returns:
        True if preference was set successfully.
    """
    logger.info(f"Setting preference: {key} = {value}")

    # Placeholder - will be connected to actual memory system
    # In production: await memory.set_preference(key, value)

    return True


# Export the server for use by MCPServerManager
__all__ = ["mcp", "roxy_search_memory", "roxy_system_status", "roxy_search_files"]


def main() -> None:
    """Run the Roxy System MCP server.

    This entry point allows the server to be run directly for testing.
    """
    if HAS_FASTMCP:
        mcp.run()
    else:
        logger.error("FastMCP is required to run this server")
        logger.info("Install with: uv add 'mcp>=1.0.0'")


if __name__ == "__main__":
    main()
