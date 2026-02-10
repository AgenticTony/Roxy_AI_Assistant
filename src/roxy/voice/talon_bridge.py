"""Talon Voice bridge via Unix socket.

Provides a socket server that Talon Voice can connect to
for bidirectional command communication.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roxy.brain.orchestrator import RoxyOrchestrator

logger = logging.getLogger(__name__)

# Default socket path
DEFAULT_SOCKET_PATH = "/tmp/roxy_talon.sock"

# Protocol constants
MAX_MESSAGE_SIZE = 4096  # Maximum message size in bytes


class TalonBridge:
    """
    Unix socket server for Talon Voice integration.

    Listens for JSON commands from Talon and routes them to
    the orchestrator for processing.
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
    ) -> None:
        """Initialize the Talon bridge.

        Args:
            socket_path: Path to the Unix socket file.
        """
        self.socket_path = Path(socket_path)

        # State management
        self._server: asyncio.Server | None = None
        self._running: bool = False
        self._orchestrator: RoxyOrchestrator | None = None

        # Active connections
        self._connections: set[asyncio.StreamWriter] = set()

        logger.debug(f"TalonBridge initialized with socket_path='{socket_path}'")

    async def start_server(self, orchestrator: RoxyOrchestrator) -> None:
        """Start the socket server.

        Args:
            orchestrator: Roxy orchestrator for processing commands.

        Raises:
            RuntimeError: If server is already running.
        """
        if self._running:
            raise RuntimeError("Talon bridge server is already running")

        self._orchestrator = orchestrator

        # Clean up existing socket if present
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
                logger.debug(f"Removed existing socket: {self.socket_path}")
            except Exception as e:
                logger.warning(f"Could not remove existing socket: {e}")

        # Create server
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path),
        )

        # Set socket permissions
        try:
            import os

            os.chmod(str(self.socket_path), 0o777)
        except Exception as e:
            logger.warning(f"Could not set socket permissions: {e}")

        self._running = True
        logger.info(f"Talon bridge server started on {self.socket_path}")

    async def stop_server(self) -> None:
        """Stop the socket server."""
        if not self._running:
            return

        logger.info("Stopping Talon bridge server...")

        self._running = False

        # Close all connections
        for writer in list(self._connections):
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                logger.debug(f"Error closing connection: {e}")

        self._connections.clear()

        # Close server
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Remove socket file
        try:
            self.socket_path.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Error removing socket: {e}")

        logger.info("Talon bridge server stopped")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a client connection.

        Args:
            reader: Stream reader for incoming data.
            writer: Stream writer for outgoing data.
        """
        # Track connection
        self._connections.add(writer)

        # Get client address (for logging)
        peer_name = writer.get_extra_info("peername")
        logger.debug(f"Client connected: {peer_name}")

        try:
            while self._running:
                # Read message length (4 bytes)
                length_bytes = await reader.readexactly(4)
                message_length = int.from_bytes(length_bytes, byteorder="big")

                if message_length > MAX_MESSAGE_SIZE:
                    logger.warning(f"Message too large: {message_length} bytes")
                    await self._send_error(writer, "Message too large")
                    break

                # Read message content
                message_bytes = await reader.readexactly(message_length)

                try:
                    # Parse JSON message
                    message = json.loads(message_bytes.decode("utf-8"))
                    logger.debug(f"Received message: {message}")

                    # Process command
                    response = await self._process_command(message)

                    # Send response
                    await self._send_response(writer, response)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    await self._send_error(writer, "Invalid JSON")
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    await self._send_error(writer, str(e))

        except asyncio.IncompleteReadError:
            # Client disconnected
            logger.debug(f"Client disconnected: {peer_name}")
        except Exception as e:
            logger.error(f"Error handling client: {e}", exc_info=True)
        finally:
            # Clean up connection
            self._connections.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _process_command(self, message: dict) -> dict:
        """Process a command message.

        Args:
            message: Parsed JSON command message.

        Returns:
            Response dictionary.
        """
        command = message.get("command")
        args = message.get("args", {})

        if not command:
            return {
                "success": False,
                "error": "Missing 'command' field",
            }

        logger.info(f"Processing command: {command} with args: {args}")

        try:
            # Route command based on type
            if command == "open_app":
                return await self._open_app(args)
            elif command == "close_app":
                return await self._close_app(args)
            elif command == "process_text":
                return await self._process_text(args)
            elif command == "get_status":
                return await self._get_status()
            else:
                return {
                    "success": False,
                    "error": f"Unknown command: {command}",
                    "available_commands": [
                        "open_app",
                        "close_app",
                        "process_text",
                        "get_status",
                    ],
                }

        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def _open_app(self, args: dict) -> dict:
        """Open an application.

        Args:
            args: Must contain 'name' key with app name.

        Returns:
            Response with success status.
        """
        import subprocess

        name = args.get("name")
        if not name:
            return {"success": False, "error": "Missing 'name' argument"}

        try:
            result = subprocess.run(
                ["open", "-a", name],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "response": f"Opened {name}",
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr or f"Failed to open {name}",
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout opening application"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _close_app(self, args: dict) -> dict:
        """Close an application.

        Args:
            args: Must contain 'name' key with app name.

        Returns:
            Response with success status.
        """
        import subprocess

        name = args.get("name")
        if not name:
            return {"success": False, "error": "Missing 'name' argument"}

        try:
            result = subprocess.run(
                ["osascript", "-e", f'quit app "{name}"'],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "response": f"Closed {name}",
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr or f"Failed to close {name}",
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout closing application"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _process_text(self, args: dict) -> dict:
        """Process text through the orchestrator.

        Args:
            args: Must contain 'text' key with user input.

        Returns:
            Response with orchestrator output.
        """
        text = args.get("text")
        if not text:
            return {"success": False, "error": "Missing 'text' argument"}

        if self._orchestrator is None:
            return {"success": False, "error": "Orchestrator not configured"}

        try:
            response = await self._orchestrator.process(text)
            return {
                "success": True,
                "response": response,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_status(self) -> dict:
        """Get the current status.

        Returns:
            Status information.
        """
        return {
            "success": True,
            "status": {
                "running": self._running,
                "connections": len(self._connections),
                "orchestrator_configured": self._orchestrator is not None,
            },
        }

    async def _send_response(self, writer: asyncio.StreamWriter, response: dict) -> None:
        """Send a response to the client.

        Args:
            writer: Stream writer.
            response: Response dictionary.
        """
        response_json = json.dumps(response)
        response_bytes = response_json.encode("utf-8")
        length_bytes = len(response_bytes).to_bytes(4, byteorder="big")

        writer.write(length_bytes + response_bytes)
        await writer.drain()

    async def _send_error(self, writer: asyncio.StreamWriter, error: str) -> None:
        """Send an error response.

        Args:
            writer: Stream writer.
            error: Error message.
        """
        await self._send_response(
            writer,
            {
                "success": False,
                "error": error,
            },
        )

    @property
    def is_running(self) -> bool:
        """Check if the server is running.

        Returns:
            True if running, False otherwise.
        """
        return self._running

    @property
    def connection_count(self) -> int:
        """Get the number of active connections.

        Returns:
            Number of connected clients.
        """
        return len(self._connections)


async def create_talon_bridge(
    socket_path: str = DEFAULT_SOCKET_PATH,
) -> TalonBridge:
    """
    Create a Talon bridge server.

    Convenience function that creates a TalonBridge instance.

    Args:
        socket_path: Path to the Unix socket file.

    Returns:
        TalonBridge instance.
    """
    bridge = TalonBridge(socket_path=socket_path)
    return bridge
