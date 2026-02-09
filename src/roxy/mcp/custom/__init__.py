"""Custom MCP servers built for Roxy.

This package contains custom MCP servers that expose Roxy's internal
capabilities through the Model Context Protocol.
"""

from roxy.mcp.custom.roxy_system import (
    mcp,
    main,
    roxy_get_user_preferences,
    roxy_search_files,
    roxy_search_memory,
    roxy_set_preference,
    roxy_system_status,
)

__all__ = [
    "mcp",
    "main",
    "roxy_search_memory",
    "roxy_system_status",
    "roxy_search_files",
    "roxy_get_user_preferences",
    "roxy_set_preference",
]
