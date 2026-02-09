"""Unit tests for Talon Voice bridge."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.voice.talon_bridge import TalonBridge, DEFAULT_SOCKET_PATH


class TestTalonBridge:
    """Tests for TalonBridge class."""

    def test_initialization(self) -> None:
        """Test bridge initialization with default values."""
        bridge = TalonBridge()
        assert bridge.socket_path == Path(DEFAULT_SOCKET_PATH)
        assert not bridge.is_running
        assert bridge.connection_count == 0

    def test_initialization_with_custom_path(self) -> None:
        """Test bridge initialization with custom socket path."""
        bridge = TalonBridge(socket_path="/tmp/test_roxy.sock")
        assert bridge.socket_path == Path("/tmp/test_roxy.sock")

    @pytest.mark.asyncio
    async def test_start_server_creates_socket(self) -> None:
        """Test that start_server creates the socket file."""
        bridge = TalonBridge()

        mock_orchestrator = MagicMock()

        with patch("roxy.voice.talon_bridge.asyncio.start_unix_server") as mock_start_server:
            mock_server = MagicMock()
            mock_server.sockets = []
            mock_start_server.return_value = mock_server

            await bridge.start_server(mock_orchestrator)

            assert bridge.is_running
            assert bridge._orchestrator is mock_orchestrator

    @pytest.mark.asyncio
    async def test_start_server_already_running(self) -> None:
        """Test that starting already-running server raises error."""
        bridge = TalonBridge()
        bridge._running = True

        mock_orchestrator = MagicMock()

        with pytest.raises(RuntimeError, match="already running"):
            await bridge.start_server(mock_orchestrator)

    @pytest.mark.asyncio
    async def test_stop_server(self) -> None:
        """Test stopping the server."""
        bridge = TalonBridge()
        bridge._running = True
        bridge._server = MagicMock()
        bridge._connections = {MagicMock(), MagicMock()}

        with patch.object(bridge.socket_path, "unlink"):
            await bridge.stop_server()

            assert not bridge.is_running
            assert bridge.connection_count == 0

    @pytest.mark.asyncio
    async def test_stop_server_not_running(self) -> None:
        """Test stopping server when not running does nothing."""
        bridge = TalonBridge()
        bridge._running = False

        # Should not raise
        await bridge.stop_server()

    @pytest.mark.asyncio
    async def test_process_open_app_command(self) -> None:
        """Test processing open_app command."""
        bridge = TalonBridge()

        with patch("roxy.voice.talon_bridge.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = await bridge._open_app({"name": "Safari"})

            assert result["success"] is True
            assert "Opened Safari" in result["response"]
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_open_app_missing_name(self) -> None:
        """Test open_app command with missing name."""
        bridge = TalonBridge()

        result = await bridge._open_app({})

        assert result["success"] is False
        assert "Missing 'name'" in result["error"]

    @pytest.mark.asyncio
    async def test_process_close_app_command(self) -> None:
        """Test processing close_app command."""
        bridge = TalonBridge()

        with patch("roxy.voice.talon_bridge.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = await bridge._close_app({"name": "Safari"})

            assert result["success"] is True
            assert "Closed Safari" in result["response"]
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_close_app_missing_name(self) -> None:
        """Test close_app command with missing name."""
        bridge = TalonBridge()

        result = await bridge._close_app({})

        assert result["success"] is False
        assert "Missing 'name'" in result["error"]

    @pytest.mark.asyncio
    async def test_process_text_command(self) -> None:
        """Test processing process_text command."""
        bridge = TalonBridge()
        mock_orchestrator = MagicMock()
        mock_orchestrator.process = AsyncMock(return_value="Response text")
        bridge._orchestrator = mock_orchestrator

        result = await bridge._process_text({"text": "Hello Roxy"})

        assert result["success"] is True
        assert result["response"] == "Response text"
        mock_orchestrator.process.assert_called_once_with("Hello Roxy")

    @pytest.mark.asyncio
    async def test_process_text_command_missing_text(self) -> None:
        """Test process_text command with missing text."""
        bridge = TalonBridge()

        result = await bridge._process_text({})

        assert result["success"] is False
        assert "Missing 'text'" in result["error"]

    @pytest.mark.asyncio
    async def test_process_text_command_no_orchestrator(self) -> None:
        """Test process_text command with no orchestrator configured."""
        bridge = TalonBridge()
        bridge._orchestrator = None

        result = await bridge._process_text({"text": "Hello"})

        assert result["success"] is False
        assert "Orchestrator not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_get_status_command(self) -> None:
        """Test processing get_status command."""
        bridge = TalonBridge()
        bridge._running = True
        bridge._connections = {MagicMock(), MagicMock()}

        result = await bridge._get_status()

        assert result["success"] is True
        assert result["status"]["running"] is True
        assert result["status"]["connections"] == 2

    @pytest.mark.asyncio
    async def test_process_unknown_command(self) -> None:
        """Test processing unknown command."""
        bridge = TalonBridge()

        result = await bridge._process_command({"command": "unknown_command"})

        assert result["success"] is False
        assert "Unknown command" in result["error"]
        assert "available_commands" in result

    @pytest.mark.asyncio
    async def test_process_command_missing_command_field(self) -> None:
        """Test processing command with missing 'command' field."""
        bridge = TalonBridge()

        result = await bridge._process_command({})

        assert result["success"] is False
        assert "Missing 'command'" in result["error"]

    @pytest.mark.asyncio
    async def test_send_response(self) -> None:
        """Test sending response to client."""
        bridge = TalonBridge()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()

        response = {"success": True, "data": "test"}

        await bridge._send_response(mock_writer, response)

        # Verify write was called
        mock_writer.write.assert_called_once()
        mock_writer.drain.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_error(self) -> None:
        """Test sending error response."""
        bridge = TalonBridge()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()

        await bridge._send_error(mock_writer, "Test error")

        # Verify write was called
        mock_writer.write.assert_called_once()
        mock_writer.drain.assert_called_once()


class TestCreateTalonBridge:
    """Tests for create_talon_bridge convenience function."""

    @pytest.mark.asyncio
    async def test_create_talon_bridge_returns_instance(self) -> None:
        """Test that create_talon_bridge returns TalonBridge instance."""
        from roxy.voice.talon_bridge import create_talon_bridge

        bridge = await create_talon_bridge()

        assert isinstance(bridge, TalonBridge)
        assert bridge.socket_path == Path(DEFAULT_SOCKET_PATH)

    @pytest.mark.asyncio
    async def test_create_talon_bridge_with_custom_path(self) -> None:
        """Test create_talon_bridge with custom socket path."""
        from roxy.voice.talon_bridge import create_talon_bridge

        bridge = await create_talon_bridge(socket_path="/tmp/custom.sock")

        assert bridge.socket_path == Path("/tmp/custom.sock")
