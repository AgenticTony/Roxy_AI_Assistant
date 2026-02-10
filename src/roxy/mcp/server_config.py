"""MCP Server configuration data structures.

Defines enums and dataclasses for MCP server configuration and connections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ServerTransport(str, Enum):
    """Transport types for MCP server connections."""

    STDIO = "stdio"  # Standard input/output (local processes)
    SSE = "sse"      # Server-Sent Events (HTTP-based)
    HTTP = "http"    # HTTP-based (for APIs)


class ServerStatus(str, Enum):
    """Status of MCP server connections."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class ServerConfig:
    """Configuration for a single MCP server."""

    name: str
    description: str
    enabled: bool = True
    transport: ServerTransport = ServerTransport.SSE
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    module: str | None = None  # For custom servers (Python module path)
    health_check: str | None = None
    tools: list[dict] = field(default_factory=list)

    # Connection settings
    timeout_connection: int = 10
    timeout_request: int = 30

    def __post_init__(self) -> None:
        """Validate and convert transport string to enum."""
        if isinstance(self.transport, str):
            self.transport = ServerTransport(self.transport)


@dataclass
class ServerConnection:
    """Active connection to an MCP server."""

    name: str
    config: ServerConfig
    status: ServerStatus = ServerStatus.STOPPED
    process: Any = None  # subprocess.Process for stdio servers
    client: Any = None   # MCP client instance
    tools: list[dict] = field(default_factory=list)
    error: str | None = None
    started_at: datetime | None = None


@dataclass
class ToolCallResult:
    """Result from calling an MCP tool."""

    success: bool
    data: Any = None
    error: str | None = None
    server: str = ""
    tool: str = ""
