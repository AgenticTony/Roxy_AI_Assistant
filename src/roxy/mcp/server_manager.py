"""MCP Server Manager for Roxy.

Manages lifecycle and communication with MCP (Model Context Protocol) servers.
This module now imports from split modules for better organization.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from mcp import ClientSession
from mcp.client.sse import sse_client

# Import from split modules
from .server_config import (
    ServerConfig,
    ServerConnection,
    ServerStatus,
    ServerTransport,
    ToolCallResult,
)
from .server_health import ServerHealthChecker

logger = logging.getLogger(__name__)


class MCPServerManager:
    """
    Manager for MCP server lifecycle and tool calls.

    Handles starting/stopping servers, listing tools, and executing
    tool calls through the MCP protocol.
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        """Initialize MCP server manager."""
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "mcp_servers.yaml"

        self.config_path = Path(config_path)
        self.servers: dict[str, ServerConnection] = {}
        self.configs: dict[str, ServerConfig] = {}
        self._load_config()
        logger.info(f"MCPServerManager initialized with {len(self.configs)} server configs")

    def _load_config(self) -> None:
        """Load server configurations from YAML file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"MCP config not found at {self.config_path}")
                return

            with self.config_path.open("r") as f:
                data = yaml.safe_load(f) or {}

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

            logger.debug(f"Loaded {len(self.configs)} server configurations")

        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")

    async def start_server(self, name: str) -> bool:
        """Start an MCP server."""
        if name not in self.configs:
            logger.error(f"Unknown server: {name}")
            return False

        config = self.configs[name]
        if not config.enabled:
            return False

        if name in self.servers and self.servers[name].status == ServerStatus.RUNNING:
            return True

        connection = ServerConnection(
            name=name,
            config=config,
            status=ServerStatus.STARTING,
        )
        self.servers[name] = connection

        try:
            if config.module:
                import importlib
                module = importlib.import_module(config.module)
                if not hasattr(module, "mcp"):
                    raise ValueError(f"Module {config.module} has no 'mcp' attribute")
                connection.client = module.mcp
                connection.tools = config.tools

            connection.status = ServerStatus.RUNNING
            connection.started_at = datetime.now()
            logger.info(f"Started MCP server: {name}")
            return True

        except Exception as e:
            connection.status = ServerStatus.ERROR
            connection.error = str(e)
            logger.error(f"Failed to start server {name}: {e}")
            return False

    async def stop_server(self, name: str) -> bool:
        """Stop an MCP server."""
        if name not in self.servers:
            return True

        connection = self.servers[name]
        connection.status = ServerStatus.STOPPED
        del self.servers[name]
        return True

    async def start_all(self) -> dict[str, bool]:
        """Start all enabled servers."""
        results = {}
        for name, config in self.configs.items():
            if config.enabled:
                results[name] = await self.start_server(name)
        return results

    async def stop_all(self) -> dict[str, bool]:
        """Stop all running servers."""
        results = {}
        for name in list(self.servers.keys()):
            results[name] = await self.stop_server(name)
        return results

    def get_tools(self, server: str) -> list[dict]:
        """Get available tools from a server."""
        if server not in self.servers:
            return []
        return self.servers[server].tools

    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict | None = None,
    ) -> ToolCallResult:
        """Call a tool on an MCP server."""
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
                error=f"Server {server} not in RUNNING state",
                server=server,
                tool=tool,
            )

        try:
            if hasattr(connection.client, "call_tool"):
                result = connection.client.call_tool(tool, arguments or {})
                return ToolCallResult(
                    success=True,
                    data=result,
                    server=server,
                    tool=tool,
                )

            return ToolCallResult(
                success=False,
                error=f"Cannot call tool on this server type",
                server=server,
                tool=tool,
            )

        except Exception as e:
            logger.error(f"Error calling tool {tool} on {server}: {e}")
            return ToolCallResult(
                success=False,
                error=str(e),
                server=server,
                tool=tool,
            )

    async def health_check(self, server: str) -> bool:
        """Check if a server is healthy."""
        if server not in self.servers:
            return False
        connection = self.servers[server]
        return await ServerHealthChecker.check_server_health(connection)

    def get_server_status(self, name: str | None = None) -> dict[str, ServerStatus]:
        """Get status of servers."""
        if name:
            if name in self.servers:
                return {name: self.servers[name].status}
            return {}
        return {name: conn.status for name, conn in self.servers.items()}
