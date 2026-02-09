"""MCP Server Manager for Roxy.

Manages lifecycle and communication with MCP (Model Context Protocol) servers.
Provides unified interface for skills to call MCP tools.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

import yaml
from mcp import ClientSession, types
from mcp.client.sse import sse_client

# Try to import aiohttp for HTTP health checks
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

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

        Args:
            connection: ServerConnection to initialize.

        Raises:
            ValueError: If module import fails or subprocess doesn't start.
            RuntimeError: If server doesn't respond to initialization.
        """
        config = connection.config

        if config.module:
            # Import custom Python server module (FastMCP)
            logger.debug(f"Loading custom server module: {config.module}")

            import importlib

            try:
                module = importlib.import_module(config.module)

                # Get FastMCP instance - module should export 'mcp'
                if not hasattr(module, "mcp"):
                    raise ValueError(f"Module {config.module} has no 'mcp' attribute")

                mcp_instance = module.mcp
                connection.client = mcp_instance

                # Extract tools from FastMCP server
                # FastMCP has ToolManager accessible via _tool_manager attribute
                if hasattr(mcp_instance, "_tool_manager"):
                    try:
                        # get_tools() is async and returns dict of FunctionTool objects
                        tools_dict = await mcp_instance._tool_manager.get_tools()
                        connection.tools = [
                            {
                                "name": tool.name,
                                "description": tool.description or "",
                            }
                            for tool in tools_dict.values()
                        ]
                    except Exception as e:
                        logger.warning(f"Failed to extract tools from FastMCP: {e}")
                        connection.tools = config.tools
                elif hasattr(mcp_instance, "_tools"):
                    # Fallback for older FastMCP versions
                    connection.tools = [
                        {
                            "name": name,
                            "description": getattr(tool, "func", tool).__doc__ or "",
                        }
                        for name, tool in mcp_instance._tools.items()
                    ]
                else:
                    # Fallback to config tools if no tool introspection available
                    connection.tools = config.tools

                logger.info(
                    f"Loaded custom server {connection.name} with {len(connection.tools)} tools"
                )

            except ImportError as e:
                raise ValueError(f"Failed to import module {config.module}: {e}") from e

        elif config.command:
            # Spawn subprocess for external command
            logger.debug(f"Starting stdio server: {config.command} {' '.join(config.args)}")

            try:
                process = await asyncio.create_subprocess_exec(
                    config.command,
                    *config.args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                connection.process = process

                # Verify process started successfully
                await asyncio.sleep(0.1)  # Give process time to start
                if process.returncode is not None:
                    stderr_output = ""
                    if process.stderr:
                        stderr_output = (await process.stderr.read()).decode()
                    raise RuntimeError(
                        f"Server process exited with code {process.returncode}: {stderr_output}"
                    )

                # Send initialize message and retrieve tools dynamically
                connection.tools = await self._discover_stdio_tools(connection)

                logger.info(f"Started stdio server {connection.name} (PID: {process.pid})")

            except FileNotFoundError as e:
                raise ValueError(f"Command not found: {config.command}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to start subprocess: {e}") from e
        else:
            raise ValueError(
                f"STDIO server {connection.name} has neither module nor command"
            )

    async def _discover_stdio_tools(self, connection: ServerConnection) -> list[dict]:
        """Discover available tools from a stdio MCP server via JSON-RPC.

        Sends initialize and tools/list requests to the server process.

        Args:
            connection: ServerConnection with running process.

        Returns:
            List of tool dictionaries with name and description.
        """
        import json

        if not connection.process or not connection.process.stdin:
            logger.warning(f"Cannot discover tools for {connection.name}: no stdin")
            return connection.config.tools

        tools = []
        request_id = 0

        async def _send_request(method: str, params: dict | None = None) -> None:
            """Send a JSON-RPC request to the server."""
            nonlocal request_id
            request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
            message = json.dumps(request) + "\n"
            await connection.process.stdin.write(message.encode())
            await connection.process.stdin.drain()

        async def _read_response(timeout: float = 5.0) -> dict | None:
            """Read a JSON-RPC response from the server."""
            try:
                line = await asyncio.wait_for(
                    connection.process.stdout.readline(),
                    timeout=timeout,
                )
                if not line:
                    return None
                return json.loads(line.decode())
            except asyncio.TimeoutError:
                logger.warning(f"Timeout reading response from {connection.name}")
                return None
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON-RPC response: {e}")
                return None

        try:
            # Send initialize request
            logger.debug(f"Initializing MCP server {connection.name}")
            await _send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "roxy",
                    "version": "0.1.0",
                },
            })

            # Read initialize response
            init_response = await _read_response()
            if not init_response:
                logger.warning(f"No initialize response from {connection.name}")
                return connection.config.tools

            if "error" in init_response:
                logger.warning(f"Initialize error from {connection.name}: {init_response['error']}")
                return connection.config.tools

            # Send initialized notification
            await _send_request("notifications/initialized")

            # Request tools list
            logger.debug(f"Requesting tools from {connection.name}")
            await _send_request("tools/list")

            # Read tools response
            tools_response = await _read_response(timeout=10.0)
            if not tools_response:
                logger.warning(f"No tools response from {connection.name}")
                return connection.config.tools

            if "error" in tools_response:
                logger.warning(f"Tools list error from {connection.name}: {tools_response['error']}")
                return connection.config.tools

            # Parse tools from response
            result = tools_response.get("result", {})
            tools_list = result.get("tools", [])

            for tool in tools_list:
                tools.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema", {}),
                })

            logger.info(f"Discovered {len(tools)} tools from {connection.name}")
            return tools

        except Exception as e:
            logger.warning(f"Failed to discover tools from {connection.name}: {e}")
            return connection.config.tools

    async def _start_http_server(self, connection: ServerConnection) -> None:
        """Connect to an HTTP/SSE-based MCP server.

        Establishes a temporary connection to initialize session and retrieve available tools.
        The actual connection is created on-demand for each tool call.

        Args:
            connection: ServerConnection to initialize.

        Raises:
            ValueError: If URL is not configured.
            RuntimeError: If connection or initialization fails.
        """
        config = connection.config

        if not config.url:
            raise ValueError(f"HTTP server {connection.name} has no URL configured")

        logger.debug(f"Connecting to HTTP/SSE server: {config.url}")

        # SSE client timeout settings
        sse_read_timeout = 300  # 5 minutes for SSE connection
        connection_timeout = config.timeout_connection

        try:
            # Create a temporary SSE client connection to retrieve server info and tools
            async with sse_client(
                url=config.url,
                timeout=connection_timeout,
                sse_read_timeout=sse_read_timeout,
            ) as (read_stream, write_stream):
                # Create ClientSession with the streams
                session = ClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    read_timeout_seconds=timedelta(seconds=config.timeout_request),
                )

                # Initialize the session (handshake with server)
                logger.debug(f"Initializing session with {connection.name}")
                init_result = await session.initialize()

                logger.info(
                    f"Initialized {connection.name}: "
                    f"protocol={init_result.protocolVersion}, "
                    f"server={init_result.serverInfo.name} {init_result.serverInfo.version}"
                )

                # Retrieve available tools from server
                tools_response = await session.list_tools()

                # Convert MCP Tool objects to dict format for internal use
                connection.tools = [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": tool.inputSchema,
                    }
                    for tool in tools_response.tools
                ]

                logger.info(f"Retrieved {len(connection.tools)} tools from {connection.name}")

                # Store config for creating new connections during tool calls
                # We don't store the session/streams since they're closed when exiting the context
                connection.client = {
                    "url": config.url,
                    "timeout": connection_timeout,
                    "sse_read_timeout": sse_read_timeout,
                    "request_timeout": timedelta(seconds=config.timeout_request),
                }

        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to MCP server {connection.name} at {config.url}: {e}"
            ) from e

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
                    # Gracefully terminate the subprocess
                    connection.process.terminate()

                    try:
                        # Wait up to 5 seconds for graceful shutdown
                        await asyncio.wait_for(connection.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # Force kill if graceful shutdown fails
                        logger.warning(f"Server {name} did not terminate gracefully, killing")
                        connection.process.kill()
                        await connection.process.wait()

                    logger.debug(f"Terminated subprocess for {name} (PID: {connection.process.pid})")

                # Clear client reference for custom servers
                if connection.client:
                    connection.client = None

            connection.status = ServerStatus.STOPPED
            connection.started_at = None
            connection.process = None
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

            # Route to appropriate transport handler
            if connection.config.transport == ServerTransport.STDIO:
                return await self._call_stdio_tool(connection, tool, args)
            elif connection.config.transport in (ServerTransport.SSE, ServerTransport.HTTP):
                return await self._call_http_tool(connection, tool, args)
            else:
                return ToolCallResult(
                    success=False,
                    error=f"Unsupported transport: {connection.config.transport}",
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

    async def _call_http_tool(
        self,
        connection: ServerConnection,
        tool: str,
        args: dict[str, Any],
    ) -> ToolCallResult:
        """Call a tool on an HTTP/SSE MCP server.

        Creates a new SSE connection for this tool call.

        Args:
            connection: Active server connection.
            tool: Tool name to call.
            args: Tool arguments.

        Returns:
            ToolCallResult with success status and data.
        """
        client_config = connection.client
        if not client_config or "url" not in client_config:
            return ToolCallResult(
                success=False,
                error="HTTP client not initialized",
                server=connection.name,
                tool=tool,
            )

        try:
            # Create a new SSE connection for this tool call
            async with sse_client(
                url=client_config["url"],
                timeout=client_config["timeout"],
                sse_read_timeout=client_config["sse_read_timeout"],
            ) as (read_stream, write_stream):
                session = ClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    read_timeout_seconds=client_config["request_timeout"],
                )

                # Initialize session
                await session.initialize()

                # Call the tool
                result = await session.call_tool(name=tool, arguments=args)

                # Convert MCP result to ToolCallResult
                if result.isError:
                    error_content = []
                    for content in result.content:
                        if hasattr(content, "text"):
                            error_content.append(content.text)
                        else:
                            error_content.append(str(content))

                    return ToolCallResult(
                        success=False,
                        error=" | ".join(error_content),
                        server=connection.name,
                        tool=tool,
                    )

                # Extract result data
                result_data = []
                for content in result.content:
                    if hasattr(content, "text"):
                        result_data.append({"type": "text", "text": content.text})
                    elif hasattr(content, "data"):
                        result_data.append({"type": "data", "data": content.data})
                    else:
                        result_data.append({"type": "unknown", "value": str(content)})

                return ToolCallResult(
                    success=True,
                    data=result_data,
                    server=connection.name,
                    tool=tool,
                )

        except Exception as e:
            logger.error(f"Error calling HTTP tool {connection.name}.{tool}: {e}")
            return ToolCallResult(
                success=False,
                error=str(e),
                server=connection.name,
                tool=tool,
            )

    async def _call_stdio_tool(
        self,
        connection: ServerConnection,
        tool: str,
        args: dict[str, Any],
    ) -> ToolCallResult:
        """Call a tool on a stdio-based MCP server.

        For custom FastMCP servers, calls the tool directly via the MCP instance.
        For external process servers, uses JSON-RPC over stdin/stdout.

        Args:
            connection: Active server connection.
            tool: Tool name to call.
            args: Tool arguments.

        Returns:
            ToolCallResult with success status and data.
        """
        # For custom FastMCP servers
        if connection.config.module and connection.client:
            try:
                mcp_instance = connection.client

                # FastMCP servers have a run method for calling tools
                # We need to invoke the tool function directly
                if hasattr(mcp_instance, "_tools") and tool in mcp_instance._tools:
                    tool_obj = mcp_instance._tools[tool]
                    tool_func = getattr(tool_obj, "func", tool_obj)

                    # Call the tool function
                    result = await tool_func(**args) if asyncio.iscoroutinefunction(tool_func) else tool_func(**args)

                    return ToolCallResult(
                        success=True,
                        data={"result": result},
                        server=connection.name,
                        tool=tool,
                    )
                else:
                    return ToolCallResult(
                        success=False,
                        error=f"Tool {tool} not found in FastMCP server",
                        server=connection.name,
                        tool=tool,
                    )

            except Exception as e:
                logger.error(f"Error calling stdio tool {connection.name}.{tool}: {e}")
                return ToolCallResult(
                    success=False,
                    error=str(e),
                    server=connection.name,
                    tool=tool,
                )

        # For external process servers - use JSON-RPC over stdin/stdout
        return await self._call_stdio_process_tool(connection, tool, args)

    async def _call_stdio_process_tool(
        self,
        connection: ServerConnection,
        tool: str,
        args: dict[str, Any],
    ) -> ToolCallResult:
        """Call a tool on an external stdio MCP server process via JSON-RPC.

        Sends a JSON-RPC request to the server process and reads the response.

        Args:
            connection: Active server connection with running process.
            tool: Tool name to call.
            args: Tool arguments.

        Returns:
            ToolCallResult with success status and data.
        """
        import json

        if not connection.process or not connection.process.stdin:
            return ToolCallResult(
                success=False,
                error="Server process not available or stdin not accessible",
                server=connection.name,
                tool=tool,
            )

        request_id = 0

        async def _send_request(method: str, params: dict | None = None) -> None:
            """Send a JSON-RPC request to the server."""
            nonlocal request_id
            request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
            message = json.dumps(request) + "\n"
            await connection.process.stdin.write(message.encode())
            await connection.process.stdin.drain()

        async def _read_response(timeout: float = 30.0) -> dict | None:
            """Read a JSON-RPC response from the server."""
            try:
                line = await asyncio.wait_for(
                    connection.process.stdout.readline(),
                    timeout=timeout,
                )
                if not line:
                    return None
                return json.loads(line.decode())
            except asyncio.TimeoutError:
                logger.warning(f"Timeout reading response from {connection.name}")
                return None
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON-RPC response: {e}")
                return None

        try:
            # Send tools/call request
            logger.debug(f"Calling tool {tool} on {connection.name}")
            await _send_request("tools/call", {
                "name": tool,
                "arguments": args,
            })

            # Read response
            response = await _read_response(timeout=60.0)
            if not response:
                return ToolCallResult(
                    success=False,
                    error="No response from server",
                    server=connection.name,
                    tool=tool,
                )

            # Check for error
            if "error" in response:
                error_data = response["error"]
                error_message = error_data.get("message", str(error_data))
                return ToolCallResult(
                    success=False,
                    error=error_message,
                    server=connection.name,
                    tool=tool,
                )

            # Extract result
            result = response.get("result", {})
            content = result.get("content", [])

            # Parse content items
            result_data = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        result_data.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "resource":
                        result_data.append({"type": "resource", "data": item})
                    elif item.get("type") == "image":
                        result_data.append({"type": "image", "data": item})
                    else:
                        result_data.append({"type": "unknown", "value": item})
                else:
                    result_data.append({"type": "unknown", "value": str(item)})

            return ToolCallResult(
                success=True,
                data=result_data,
                server=connection.name,
                tool=tool,
            )

        except asyncio.TimeoutError:
            return ToolCallResult(
                success=False,
                error="Tool call timed out",
                server=connection.name,
                tool=tool,
            )
        except Exception as e:
            logger.error(f"Error calling process tool {connection.name}.{tool}: {e}")
            return ToolCallResult(
                success=False,
                error=str(e),
                server=connection.name,
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

        # For STDIO servers, check if process is still alive
        if connection.config.transport == ServerTransport.STDIO:
            if connection.process:
                # Process is dead if returncode is not None
                if connection.process.returncode is not None:
                    logger.warning(
                        f"Server {server} process exited (code: {connection.process.returncode})"
                    )
                    connection.status = ServerStatus.ERROR
                    connection.error = f"Process exited with code {connection.process.returncode}"
                    return False
            return True

        # If server has a health check URL, verify it
        if connection.config.health_check:
            try:
                if HAS_AIOHTTP:
                    # Use aiohttp for async HTTP request
                    timeout = aiohttp.ClientTimeout(total=5)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(connection.config.health_check) as response:
                            if 200 <= response.status < 300:
                                logger.debug(f"Health check passed for {server}: {response.status}")
                                return True
                            else:
                                logger.warning(
                                    f"Health check failed for {server}: {response.status}"
                                )
                                return False
                else:
                    # Fallback: use asyncio with basic socket connection
                    # Parse URL to get host and port
                    from urllib.parse import urlparse

                    parsed = urlparse(connection.config.health_check)
                    host = parsed.hostname or parsed.path
                    port = parsed.port or (443 if parsed.scheme == "https" else 80)

                    # Try to open a connection
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(host, port),
                        timeout=5.0,
                    )
                    writer.close()
                    await writer.wait_closed()
                    logger.debug(f"Health check passed for {server} (socket)")
                    return True
            except asyncio.TimeoutError:
                logger.warning(f"Health check timed out for {server}")
                return False
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
