"""Unit tests for MCPServerManager.

Tests MCP server lifecycle management and tool calling.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from roxy.mcp.servers import (
    MCPServerManager,
    ServerConfig,
    ServerConnection,
    ServerStatus,
    ServerTransport,
    ToolCallResult,
)


@pytest.fixture
def server_config():
    """Fixture providing a test ServerConfig."""
    return ServerConfig(
        name="test_server",
        description="Test MCP server",
        enabled=True,
        transport=ServerTransport.SSE,
        url="http://localhost:8000",
    )


@pytest.fixture
def temp_config_file(tmp_path):
    """Fixture providing a temporary config file."""
    config_file = tmp_path / "mcp_servers.yaml"

    config_content = """
servers:
  test_server:
    name: "Test Server"
    description: "Test description"
    enabled: true
    transport: "sse"
    url: "http://localhost:8000"
    tools:
      - name: "test_tool"
        description: "A test tool"

custom_servers:
  roxy_system:
    name: "Roxy System"
    description: "Internal Roxy server"
    enabled: true
    transport: "stdio"
    module: "roxy.mcp.custom.roxy_system"
    tools:
      - name: "roxy_search_memory"
        description: "Search memory"
"""

    config_file.write_text(config_content)
    return config_file


class TestServerConfig:
    """Tests for ServerConfig."""

    def test_init(self):
        """Test ServerConfig initialization."""
        config = ServerConfig(
            name="test",
            description="Test server",
            transport="sse",
            url="http://localhost:8000",
        )

        assert config.name == "test"
        assert config.transport == ServerTransport.SSE
        assert config.enabled is True

    def test_transport_string_conversion(self):
        """Test that transport strings are converted to enums."""
        config = ServerConfig(
            name="test",
            description="Test",
            transport="stdio",  # String
        )

        assert isinstance(config.transport, ServerTransport)
        assert config.transport == ServerTransport.STDIO


class TestMCPServerManager:
    """Tests for MCPServerManager."""

    def test_init_with_config(self, temp_config_file):
        """Test manager initialization with config file."""
        manager = MCPServerManager(config_path=temp_config_file)

        assert len(manager.configs) > 0
        assert "test_server" in manager.configs
        assert "roxy_system" in manager.configs

    def test_init_without_config(self):
        """Test manager initialization without config file."""
        with patch("roxy.mcp.servers.Path.exists", return_value=False):
            manager = MCPServerManager(config_path="/nonexistent/config.yaml")

            # Should not crash, just have no configs
            assert len(manager.configs) == 0

    def test_register_server(self):
        """Test registering a server dynamically."""
        manager = MCPServerManager()

        config = ServerConfig(
            name="dynamic_server",
            description="Dynamically registered",
            enabled=True,
        )

        manager.register_server("dynamic_server", config)

        assert "dynamic_server" in manager.configs
        assert manager.configs["dynamic_server"].name == "dynamic_server"

    @pytest.mark.asyncio
    async def test_start_unknown_server(self):
        """Test starting an unknown server."""
        manager = MCPServerManager()

        result = await manager.start_server("unknown")

        assert result is False

    @pytest.mark.asyncio
    async def test_start_disabled_server(self, temp_config_file):
        """Test starting a disabled server."""
        # Modify config to disable server
        import yaml

        with open(temp_config_file, "r") as f:
            config_data = yaml.safe_load(f)

        config_data["servers"]["test_server"]["enabled"] = False

        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        manager = MCPServerManager(config_path=temp_config_file)

        result = await manager.start_server("test_server")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_tools_from_running_server(self, temp_config_file):
        """Test getting tools from a running server."""
        manager = MCPServerManager(config_path=temp_config_file)

        # Start server (will be mocked)
        await manager.start_server("test_server")

        tools = manager.get_tools("test_server")

        # Should return tools from config
        assert len(tools) > 0
        assert tools[0]["name"] == "test_tool"

    def test_get_tools_from_nonexistent_server(self):
        """Test getting tools from a server that doesn't exist."""
        manager = MCPServerManager()

        tools = manager.get_tools("nonexistent")

        assert tools == []

    @pytest.mark.asyncio
    async def test_call_tool_on_nonexistent_server(self):
        """Test calling a tool on a nonexistent server."""
        manager = MCPServerManager()

        result = await manager.call_tool(
            server="nonexistent",
            tool="some_tool",
            args={"param": "value"},
        )

        assert result.success is False
        assert "not running" in result.error.lower()

    @pytest.mark.asyncio
    async def test_stop_server(self, temp_config_file):
        """Test stopping a server."""
        manager = MCPServerManager(config_path=temp_config_file)

        # Start server first
        await manager.start_server("test_server")

        # Then stop it
        result = await manager.stop_server("test_server")

        assert result is True
        assert manager.servers["test_server"].status == ServerStatus.STOPPED

    def test_get_server_status(self, temp_config_file):
        """Test getting server status."""
        manager = MCPServerManager(config_path=temp_config_file)

        # Get all statuses
        all_statuses = manager.get_server_status()

        assert isinstance(all_statuses, dict)

        # Get specific server status (not running yet)
        specific_status = manager.get_server_status("test_server")

        # Should return empty for non-running server
        assert specific_status == {}


class TestToolCallResult:
    """Tests for ToolCallResult."""

    def test_success_result(self):
        """Test successful tool call result."""
        result = ToolCallResult(
            success=True,
            data={"key": "value"},
            server="test_server",
            tool="test_tool",
        )

        assert result.success is True
        assert result.data["key"] == "value"
        assert result.server == "test_server"

    def test_error_result(self):
        """Test error tool call result."""
        result = ToolCallResult(
            success=False,
            error="Tool not found",
            server="test_server",
            tool="test_tool",
        )

        assert result.success is False
        assert result.error == "Tool not found"
