"""MCP Server Manager for Roxy.

This module now imports from split modules for better organization.
- server_config.py: Configuration data structures
- server_health.py: Health checking functionality
- server_manager.py: Main MCPServerManager class

The original contents have been split to improve maintainability.
This file provides backward compatibility by re-exporting the key classes.
"""

from __future__ import annotations

# Re-export configuration classes from server_config
from .server_config import (
    ServerConfig,
    ServerConnection,
    ServerStatus,
    ServerTransport,
    ToolCallResult,
)

# Re-export the main manager class
from .server_manager import MCPServerManager

# Re-export health checker for direct use
from .server_health import ServerHealthChecker

__all__ = [
    "MCPServerManager",
    "ServerConfig",
    "ServerConnection",
    "ServerStatus",
    "ServerTransport",
    "ToolCallResult",
    "ServerHealthChecker",
]
