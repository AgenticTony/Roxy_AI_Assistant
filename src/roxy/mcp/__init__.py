"""MCP (Model Context Protocol) integration for Roxy.

This package provides:
- MCPServerManager: Lifecycle management for MCP servers
- Custom MCP servers that expose Roxy's internal capabilities
- Configuration for external MCP services (Brave, Firecrawl, etc.)
"""

from roxy.mcp.servers import (
    MCPServerManager,
    ServerConfig,
    ServerConnection,
    ServerStatus,
    ServerTransport,
    ToolCallResult,
)

__all__ = [
    "MCPServerManager",
    "ServerConfig",
    "ServerConnection",
    "ServerStatus",
    "ServerTransport",
    "ToolCallResult",
]
