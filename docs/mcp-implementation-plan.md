# MCP Implementation Plan for Roxy

## Overview

**MCP (Model Context Protocol)** is a standardized protocol for connecting AI assistants to external tools and data sources. It provides a unified interface for:

1. **Tool Discovery** - Dynamically listing available tools from a server
2. **Tool Execution** - Calling tools with arguments and receiving results
3. **Transport Flexibility** - Supporting stdio (local processes), HTTP, and SSE connections
4. **Bi-directional Communication** - Servers can also request information from the client

For Roxy, MCP enables:
- **Web search** (Brave Search API)
- **Web scraping** (Firecrawl)
- **GitHub operations** (repository management)
- **Task management** (Dart MCP)
- **Custom system tools** (Roxy System server for memory, file search, etc.)

---

## Current State Analysis

### What's Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| Core dataclasses | **Complete** | ServerConfig, ServerConnection, ToolCallResult, enums |
| Config loading | **Complete** | YAML parsing from config/mcp_servers.yaml |
| Server registration | **Complete** | Dynamic registration via `register_server()` |
| Lifecycle skeleton | **Partial** | Start/stop methods exist but need implementation |
| Custom server stub | **Complete** | roxy_system.py with FastMCP placeholder |
| Unit tests | **Partial** | Basic tests pass, but not testing actual MCP protocol |

### What's Stub/TODO

| Location | TODO | Description |
|----------|------|-------------|
| `servers.py:248` | Subprocess management | STDIO servers need actual subprocess spawning |
| `servers.py:263` | HTTP/SSE client | HTTP transport needs MCP client connection |
| `servers.py:394` | Tool call protocol | Actual MCP tool calling via JSON-RPC 2.0 |
| `servers.py:439` | Health check | HTTP health checks for server monitoring |

---

## Detailed Implementation Tasks

### Task 1: Implement STDIO Subprocess Management

**File:** `src/roxy/mcp/servers.py`
**Line:** 248
**Method:** `_start_stdio_server()`

**Description:**
Spawn and manage subprocesses for STDIO-based MCP servers. The server communicates via stdin/stdout using JSON-RPC 2.0 messages.

**Expected Implementation:**
```python
async def _start_stdio_server(self, connection: ServerConnection) -> None:
    """Start a stdio-based MCP server (local process)."""
    config = connection.config

    if config.module:
        # Import and run custom Python server directly
        # Use importlib to load and run FastMCP server
        pass
    elif config.command:
        # Spawn subprocess with stdin/stdout pipes
        process = await asyncio.create_subprocess_exec(
            config.command,
            *config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        connection.process = process

        # Send initialize message and wait for response
        # Populate connection.tools from server response
```

**Dependencies:**
- None (foundational task)

**Complexity:** Medium

**Testing Approach:**
- Mock subprocess in unit tests
- Integration test with a real echo server
- Verify process cleanup on server stop

---

### Task 2: Implement HTTP/SSE Client Connection

**File:** `src/roxy/mcp/servers.py`
**Line:** 263
**Method:** `_start_http_server()`

**Description:**
Connect to HTTP/SSE-based MCP servers, establish session, and retrieve available tools.

**Expected Implementation:**
```python
async def _start_http_server(self, connection: ServerConnection) -> None:
    """Connect to an HTTP/SSE-based MCP server."""
    config = connection.config

    if not config.url:
        raise ValueError(f"HTTP server {connection.name} has no URL")

    # Use httpx or aiohttp for async HTTP
    # For SSE, establish EventSource connection
    # Send initialize message
    # Parse tools from server response

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{config.url}/initialize",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "roxy", "version": "0.1.0"}
                }
            },
            timeout=config.timeout_connection
        )
        # Parse response and populate connection.tools
```

**Dependencies:**
- None (can parallelize with Task 1)

**Complexity:** Medium

**Testing Approach:**
- Mock httpx client
- Test with local MCP test server
- Verify timeout handling

---

### Task 3: Implement MCP Tool Call Protocol

**File:** `src/roxy/mcp/servers.py`
**Line:** 394
**Method:** `call_tool()`

**Description:**
Execute tool calls using JSON-RPC 2.0 protocol. Handle both STDIO and HTTP transports.

**Expected Implementation:**
```python
async def call_tool(
    self,
    server: str,
    tool: str,
    args: dict[str, Any],
) -> ToolCallResult:
    """Call a tool on an MCP server."""
    # ... existing validation ...

    connection = self.servers[server]

    try:
        if connection.config.transport == ServerTransport.STDIO:
            # Write JSON-RPC request to stdin
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool,
                    "arguments": args
                }
            }
            # Write to process.stdin
            # Read response from stdout
        else:
            # HTTP POST to /tools/call endpoint
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{connection.config.url}/tools/call",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": tool, "arguments": args}
                    },
                    timeout=connection.config.timeout_request
                )
                # Parse response

        return ToolCallResult(success=True, data=response_data)

    except Exception as e:
        return ToolCallResult(success=False, error=str(e))
```

**Dependencies:**
- Task 1 (for STDIO)
- Task 2 (for HTTP)

**Complexity:** Hard

**Testing Approach:**
- Mock transport layer
- Test error handling (timeout, invalid tool, etc.)
- Integration test with real MCP server

---

### Task 4: Implement Health Check

**File:** `src/roxy/mcp/servers.py`
**Line:** 439
**Method:** `health_check()`

**Description:**
Verify server health via configured health check endpoint or ping.

**Expected Implementation:**
```python
async def health_check(self, server: str) -> bool:
    """Check if a server is healthy and responding."""
    if server not in self.servers:
        return False

    connection = self.servers[server]

    if connection.status != ServerStatus.RUNNING:
        return False

    # For STDIO, check process is alive
    if connection.config.transport == ServerTransport.STDIO:
        if connection.process and connection.process.returncode is not None:
            return False
        return True

    # For HTTP/SSE, use health check URL
    if connection.config.health_check:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    connection.config.health_check,
                    timeout=5
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed for {server}: {e}")
            return False

    return True
```

**Dependencies:**
- Task 2 (uses HTTP client)

**Complexity:** Easy

**Testing Approach:**
- Mock httpx client
- Test process status check for STDIO
- Test timeout handling

---

### Task 5: Add Tool List Retrieval

**File:** `src/roxy/mcp/servers.py`
**Method:** New method `list_tools()`

**Description:**
Dynamically retrieve available tools from running MCP server via protocol.

**Expected Implementation:**
```python
async def list_tools(self, server: str) -> list[dict]:
    """Dynamically list tools from a running MCP server.

    Queries the server for its available tools rather than using
    the static config. Useful for servers with dynamic tools.

    Args:
        server: Server name.

    Returns:
        List of tool definitions.
    """
    if server not in self.servers:
        return []

    connection = self.servers[server]

    if connection.config.transport == ServerTransport.STDIO:
        # Send tools/list request via stdin
        pass
    else:
        # HTTP GET /tools/list
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{connection.config.url}/tools/list",
                timeout=5
            )
            return response.json().get("tools", [])
```

**Dependencies:**
- Task 1 (for STDIO)
- Task 2 (for HTTP)

**Complexity:** Medium

**Testing Approach:**
- Mock transport layer
- Test with server that returns tool list

---

### Task 6: Implement Server Auto-Start from Config

**File:** `src/roxy/mcp/servers.py`
**Method:** Update `_load_config()` and add `auto_start_enabled`

**Description:**
Read the `startup.auto_start` list from config and automatically start those servers.

**Expected Implementation:**
```python
def _load_config(self) -> None:
    # ... existing code ...

    # Load startup configuration
    self.auto_start_servers = data.get("startup", {}).get("auto_start", [])
    self.on_demand_servers = data.get("startup", {}).get("on_demand", [])

async def start_auto_start_servers(self) -> dict[str, bool]:
    """Start all servers marked for auto-start in config."""
    results = {}
    for name in self.auto_start_servers:
        if name in self.configs:
            results[name] = await self.start_server(name)
    return results
```

**Dependencies:**
- None

**Complexity:** Easy

**Testing Approach:**
- Unit test with config containing auto_start list
- Verify only auto-start servers are started

---

### Task 7: Add Retry Logic with Backoff

**File:** `src/roxy/mcp/servers.py`
**New module:** `src/roxy/mcp/retry.py` (optional)

**Description:**
Implement configurable retry logic for tool calls and connections.

**Expected Implementation:**
```python
async def call_tool_with_retry(
    self,
    server: str,
    tool: str,
    args: dict[str, Any],
    max_retries: int = 3,
    backoff: float = 1.0,
) -> ToolCallResult:
    """Call a tool with retry logic on transient failures."""
    last_error = None

    for attempt in range(max_retries):
        result = await self.call_tool(server, tool, args)

        if result.success:
            return result

        # Don't retry on certain errors
        if "not found" in result.error.lower():
            return result

        last_error = result.error
        if attempt < max_retries - 1:
            await asyncio.sleep(backoff * (2 ** attempt))

    return ToolCallResult(
        success=False,
        error=f"Failed after {max_retries} attempts: {last_error}",
        server=server,
        tool=tool,
    )
```

**Dependencies:**
- Task 3 (base call_tool)

**Complexity:** Easy

**Testing Approach:**
- Mock call_tool to fail then succeed
- Verify exponential backoff timing

---

### Task 8: Complete Custom Server Integration

**File:** `src/roxy/mcp/servers.py`
**Method:** `_start_stdio_server()` extension

**Description:**
Properly integrate FastMCP custom servers by importing and running them.

**Expected Implementation:**
```python
async def _start_stdio_server(self, connection: ServerConnection) -> None:
    """Start a stdio-based MCP server (local process)."""
    config = connection.config

    if config.module:
        # Import custom Python server module
        import importlib

        try:
            module = importlib.import_module(config.module)

            # Get FastMCP instance
            if hasattr(module, "mcp"):
                # Store the mcp instance for direct tool calls
                connection.client = module.mcp

                # Get tools from the FastMCP server
                if hasattr(module.mcp, "_tools"):
                    connection.tools = [
                        {"name": name, "description": tool.func.__doc__}
                        for name, tool in module.mcp._tools.items()
                    ]
                else:
                    connection.tools = config.tools
            else:
                raise ValueError(f"Module {config.module} has no 'mcp' attribute")

        except ImportError as e:
            raise ValueError(f"Failed to import module {config.module}: {e}")
```

**Dependencies:**
- None

**Complexity:** Medium

**Testing Approach:**
- Test with actual roxy_system server
- Verify tool extraction from FastMCP

---

### Task 9: Add Logging and Metrics

**File:** `src/roxy/mcp/servers.py`
**New module:** `src/roxy/mcp/metrics.py` (optional)

**Description:**
Add structured logging and metrics for MCP operations.

**Expected Implementation:**
```python
@dataclass
class MCPMetrics:
    """Metrics for MCP operations."""
    tools_called: int = 0
    tools_succeeded: int = 0
    tools_failed: int = 0
    total_latency_ms: float = 0
    server_uptime_seconds: dict[str, float] = field(default_factory=dict)

class MCPServerManager:
    def __init__(self, ...):
        # ... existing code ...
        self.metrics = MCPMetrics()

    async def call_tool(self, ...) -> ToolCallResult:
        start = time.monotonic()
        self.metrics.tools_called += 1

        try:
            result = await self._call_tool_impl(server, tool, args)
            if result.success:
                self.metrics.tools_succeeded += 1
            else:
                self.metrics.tools_failed += 1
            return result
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            self.metrics.total_latency_ms += latency_ms
            logger.info(
                "MCP tool call",
                extra={
                    "server": server,
                    "tool": tool,
                    "success": result.success,
                    "latency_ms": latency_ms,
                }
            )
```

**Dependencies:**
- Task 3 (needs working tool calls)

**Complexity:** Easy

**Testing Approach:**
- Verify metrics increment correctly
- Test log output format

---

### Task 10: Add MCP Integration Tests

**File:** `tests/integration/test_mcp_integration.py` (new)

**Description:**
End-to-end tests with real MCP servers (local or mocked).

**Expected Implementation:**
```python
@pytest.mark.asyncio
async def test_full_mcp_lifecycle():
    """Test starting server, listing tools, calling tool, stopping."""
    manager = MCPServerManager()

    # Start a test server
    started = await manager.start_server("test_stdio")
    assert started is True

    # List tools
    tools = manager.get_tools("test_stdio")
    assert len(tools) > 0

    # Call a tool
    result = await manager.call_tool(
        "test_stdio",
        "echo",
        {"message": "hello"}
    )
    assert result.success is True

    # Stop server
    stopped = await manager.stop_server("test_stdio")
    assert stopped is True
```

**Dependencies:**
- All previous tasks

**Complexity:** Medium

**Testing Approach:**
- Use pytest-asyncio
- Mock external servers or use test fixtures

---

## Implementation Phases

### Phase 1: Core Subprocess Management
**Duration:** ~2-3 hours

| Task | Priority | Dependencies |
|------|----------|--------------|
| Task 1: STDIO subprocess | High | None |
| Task 8: Custom server integration | High | Task 1 |
| Task 4: Health check | Medium | None |

**Deliverables:**
- Functional STDIO server startup
- Custom FastMCP servers load correctly
- Process lifecycle management

### Phase 2: HTTP/SSE Client
**Duration:** ~2-3 hours

| Task | Priority | Dependencies |
|------|----------|--------------|
| Task 2: HTTP/SSE client | High | None |
| Task 4: Health check (HTTP part) | Medium | Task 2 |
| Task 6: Auto-start from config | Low | None |

**Deliverables:**
- HTTP connection to MCP servers
- SSE streaming support
- Dynamic tool discovery

### Phase 3: Tool Calling Protocol
**Duration:** ~3-4 hours

| Task | Priority | Dependencies |
|------|----------|--------------|
| Task 3: Tool call protocol | High | Task 1, Task 2 |
| Task 5: Tool list retrieval | Medium | Task 1, Task 2 |
| Task 7: Retry logic | Medium | Task 3 |

**Deliverables:**
- Working JSON-RPC 2.0 tool calls
- Error handling and retries
- Tool discovery

### Phase 4: Testing and Validation
**Duration:** ~2-3 hours

| Task | Priority | Dependencies |
|------|----------|--------------|
| Task 9: Logging and metrics | Low | Task 3 |
| Task 10: Integration tests | High | All previous |

**Deliverables:**
- Comprehensive test coverage
- Performance metrics
- Documentation

---

## Integration Points

### With Skill System

The MCP manager will be injected into skills via `SkillContext`:

```python
@dataclass
class SkillContext:
    user_input: str
    intent: str
    parameters: dict[str, Any]
    memory: "MemoryManager"
    config: "RoxyConfig"
    conversation_history: list[dict]
    mcp: MCPServerManager  # <-- New field
```

Skills can then call MCP tools:

```python
class WebSearchSkill(RoxySkill):
    async def execute(self, context: SkillContext) -> SkillResult:
        result = await context.mcp.call_tool(
            server="brave_search",
            tool="brave_web_search",
            args={"query": context.user_input}
        )

        if result.success:
            return SkillResult(
                success=True,
                response_text=f"Found: {result.data}"
            )
```

### With Orchestrator

The orchestrator will initialize the MCP manager on startup:

```python
class Orchestrator:
    async def start(self):
        # Initialize MCP manager
        self.mcp_manager = MCPServerManager()
        await self.mcp_manager.start_auto_start_servers()

        # Pass to skill context factory
        self.context_factory.mcp = self.mcp_manager
```

### With Privacy Gateway

All MCP calls through HTTP/SSE should go through the privacy gateway:

```python
async def call_tool(self, server: str, tool: str, args: dict) -> ToolCallResult:
    # Check if server is external (not custom)
    if self.configs[server].transport != ServerTransport.STDIO:
        # Apply PII redaction to args
        args = self.privacy_gateway.redact(args)

        # Log the call
        self._log_cloud_request(server, tool, args)
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `httpx` | latest | Async HTTP client for MCP |
| `mcp` | latest | MCP protocol types |
| `fastmcp` | latest | Custom MCP server creation |
| `pyyaml` | latest | Config parsing (already installed) |

**Installation:**
```bash
uv add httpx mcp fastmcp
```

---

## Success Criteria

- [ ] All TODO items in `servers.py` implemented
- [ ] At least 80% test coverage for MCP module
- [ ] Integration tests pass with a real MCP server
- [ ] Documentation updated with MCP usage examples
- [ ] Performance targets met:
  - Tool call latency < 500ms (local STDIO)
  - Tool call latency < 2s (HTTP servers)
  - Server startup < 5s

---

## Notes

1. **Backward Compatibility:** Changes should not break existing tests in `test_mcp_manager.py`

2. **Error Handling:** All MCP errors should be caught and returned as `ToolCallResult(success=False)`, not raised as exceptions

3. **Logging:** Use structured logging with `extra` for observability

4. **Thread Safety:** The manager may be accessed from multiple async tasks - ensure proper use of async/await throughout

5. **Resource Cleanup:** All processes and HTTP clients must be properly closed in `stop_server()` and `stop_all()`
