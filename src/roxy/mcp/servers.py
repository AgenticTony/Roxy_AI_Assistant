"""MCP Server Manager for Roxy.

Manages lifecycle and communication with MCP (Model Context Protocol) servers.
Provides unified interface for skills to call MCP tools.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

import yaml

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


class MCPServerManager:
    """
    Manager for MCP server lifecycle and tool calls.

    Handles starting/stopping servers, listing tools, and executing
    tool calls through the MCP protocol.
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        """Initialize MCP server manager.

        Args:
            config_path: Path to mcp_servers.yaml. Defaults to config/mcp_servers.yaml.
        """
        if config_path is None:
            # Default to project config directory
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "mcp_servers.yaml"

        self.config_path = Path(config_path)
        self.servers: dict[str, ServerConnection] = {}
        self.configs: dict[str, ServerConfig] = {}

        # Load configuration
        self._load_config()

        logger.info(f"MCPServerManager initialized with {len(self.configs)} server configs")

    def _load_config(self) -> None:
        """Load server configurations from YAML file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"MCP config not found at {self.config_path}, using defaults")
                return

            with self.config_path.open("r") as f:
                data = yaml.safe_load(f) or {}

            # Load regular servers
            for name, server_data in data.get("servers", {}).items():
                self.configs[name] = ServerConfig(
                    name=name,
                    description=server_data.get("description", ""),
                    enabled=server_data.get("enabled", True),
                    transport=server_data.get("transport", "sse"),
                    url=server_data.get("url"),
                    command=server_data.get("command"),
                    args=server_data.get("args", []),
                    module=server_data.get("module"),
                    health_check=server_data.get("health_check"),
                    tools=server_data.get("tools", []),
                )

            # Load custom servers
            for name, server_data in data.get("custom_servers", {}).items():
                self.configs[name] = ServerConfig(
                    name=name,
                    description=server_data.get("description", ""),
                    enabled=server_data.get("enabled", True),
                    transport=server_data.get("transport", "stdio"),
                    module=server_data.get("module"),
                    tools=server_data.get("tools", []),
                )

            # Load fallback servers
            for name, server_data in data.get("fallback_servers", {}).items():
                self.configs[name] = ServerConfig(
                    name=name,
                    description=server_data.get("description", ""),
                    enabled=server_data.get("enabled", False),
                    transport=server_data.get("transport", "http"),
                    url=server_data.get("url"),
                    health_check=server_data.get("health_check"),
                    tools=server_data.get("tools", []),
                )

            logger.debug(f"Loaded {len(self.configs)} server configurations")

        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")

    def register_server(self, name: str, config: ServerConfig) -> None:
        """Register a server configuration dynamically.

        Args:
            name: Unique name for the server.
            config: ServerConfig instance.
        """
        self.configs[name] = config
        logger.debug(f"Registered MCP server: {name}")

    async def start_server(self, name: str) -> bool:
        """Start an MCP server.

        Args:
            name: Server name from config.

        Returns:
            True if server started successfully.
        """
        if name not in self.configs:
            logger.error(f"Unknown server: {name}")
            return False

        config = self.configs[name]

        if not config.enabled:
            logger.warning(f"Server {name} is disabled")
            return False

        if name in self.servers and self.servers[name].status == ServerStatus.RUNNING:
            logger.debug(f"Server {name} already running")
            return True

        # Create connection object
        connection = ServerConnection(
            name=name,
            config=config,
            status=ServerStatus.STARTING,
        )
        self.servers[name] = connection

        try:
            # Start server based on transport type
            if config.transport == ServerTransport.STDIO:
                await self._start_stdio_server(connection)
            elif config.transport in (ServerTransport.SSE, ServerTransport.HTTP):
                await self._start_http_server(connection)
            else:
                raise ValueError(f"Unknown transport: {config.transport}")

            connection.status = ServerStatus.RUNNING
            connection.started_at = datetime.now()
            logger.info(f"Started MCP server: {name}")

            return True

        except Exception as e:
            connection.status = ServerStatus.ERROR
            connection.error = str(e)
            logger.error(f"Failed to start server {name}: {e}")
            return False

    async def _start_stdio_server(self, connection: ServerConnection) -> None:
        """Start a stdio-based MCP server (local process).

        For custom Python servers, imports and runs the module directly.
        For external commands, spawns a subprocess.
        """
        config = connection.config

        if config.module:
            # Import custom Python server module
            # This will be started separately as a FastMCP server
            logger.debug(f"Custom server module: {config.module}")
            connection.tools = config.tools  # Use predefined tools
        elif config.command:
            # Spawn subprocess for external command
            logger.debug(f"Starting stdio server: {config.command}")
            # TODO: Implement subprocess management
            connection.tools = config.tools

    async def _start_http_server(self, connection: ServerConnection) -> None:
        """Connect to an HTTP/SSE-based MCP server.

        Establishes client connection and retrieves available tools.
        """
        config = connection.config

        if not config.url:
            raise ValueError(f"HTTP server {connection.name} has no URL configured")

        logger.debug(f"Connecting to HTTP server: {config.url}")

        # TODO: Implement MCP HTTP client connection
        # For now, use predefined tools from config
        connection.tools = config.tools

    async def stop_server(self, name: str) -> bool:
        """Stop an MCP server.

        Args:
            name: Server name.

        Returns:
            True if server stopped successfully.
        """
        if name not in self.servers:
            logger.warning(f"Server {name} not running")
            return True

        connection = self.servers[name]

        try:
            # Clean up based on transport type
            if connection.config.transport == ServerTransport.STDIO:
                if connection.process:
                    connection.process.terminate()
                    await connection.process.wait()

            connection.status = ServerStatus.STOPPED
            connection.started_at = None
            logger.info(f"Stopped MCP server: {name}")

            return True

        except Exception as e:
            logger.error(f"Failed to stop server {name}: {e}")
            return False

    async def start_all(self) -> dict[str, bool]:
        """Start all enabled servers marked for auto-start.

        Returns:
            Dict mapping server names to start success status.
        """
        results = {}

        for name, config in self.configs.items():
            if config.enabled:
                results[name] = await self.start_server(name)

        return results

    async def stop_all(self) -> dict[str, bool]:
        """Stop all running servers.

        Returns:
            Dict mapping server names to stop success status.
        """
        results = {}

        for name in list(self.servers.keys()):
            results[name] = await self.stop_server(name)

        return results

    def get_tools(self, server: str) -> list[dict]:
        """Get available tools from a server.

        Args:
            server: Server name.

        Returns:
            List of tool definitions with 'name' and 'description'.
        """
        if server not in self.servers:
            logger.warning(f"Server {server} not running")
            return []

        connection = self.servers[server]

        if connection.status != ServerStatus.RUNNING:
            logger.warning(f"Server {server} not running (status: {connection.status})")
            return []

        return connection.tools

    async def call_tool(
        self,
        server: str,
        tool: str,
        args: dict[str, Any],
    ) -> ToolCallResult:
        """Call a tool on an MCP server.

        Args:
            server: Server name.
            tool: Tool name to call.
            args: Tool arguments.

        Returns:
            ToolCallResult with success status and data.
        """
        if server not in self.servers:
            return ToolCallResult(
                success=False,
                error=f"Server {server} not running",
                server=server,
                tool=tool,
            )

        connection = self.servers[server]

        if connection.status != ServerStatus.RUNNING:
            return ToolCallResult(
                success=False,
                error=f"Server {server} not running (status: {connection.status})",
                server=server,
                tool=tool,
            )

        try:
            # Check if tool exists
            available_tools = {t["name"]: t for t in connection.tools}
            if tool not in available_tools:
                return ToolCallResult(
                    success=False,
                    error=f"Tool {tool} not found on server {server}",
                    server=server,
                    tool=tool,
                )

            logger.info(f"Calling {server}.{tool} with args: {list(args.keys())}")

            # TODO: Implement actual MCP tool call
            # For now, return a placeholder response
            result_data = {
                "server": server,
                "tool": tool,
                "args": args,
                "timestamp": datetime.now().isoformat(),
            }

            return ToolCallResult(
                success=True,
                data=result_data,
                server=server,
                tool=tool,
            )

        except Exception as e:
            logger.error(f"Error calling {server}.{tool}: {e}")
            return ToolCallResult(
                success=False,
                error=str(e),
                server=server,
                tool=tool,
            )

    async def health_check(self, server: str) -> bool:
        """Check if a server is healthy and responding.

        Args:
            server: Server name.

        Returns:
            True if server is healthy.
        """
        if server not in self.servers:
            return False

        connection = self.servers[server]

        if connection.status != ServerStatus.RUNNING:
            return False

        # If server has a health check URL, verify it
        if connection.config.health_check:
            try:
                # TODO: Implement HTTP health check
                pass
            except Exception as e:
                logger.error(f"Health check failed for {server}: {e}")
                return False

        return True

    def get_server_status(self, name: str | None = None) -> dict[str, ServerStatus]:
        """Get status of servers.

        Args:
            name: Specific server name, or None for all servers.

        Returns:
            Dict mapping server names to their status.
        """
        if name:
            if name in self.servers:
                return {name: self.servers[name].status}
            return {}

        return {name: conn.status for name, conn in self.servers.items()}
