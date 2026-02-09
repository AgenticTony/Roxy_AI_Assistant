"""Talon Voice Python integration for Roxy.

This module provides the Python functions that Talon Voice calls
to communicate with the Roxy voice pipeline.

Install in your Talon user directory (typically ~/.talon/user/roxy/).
"""

import asyncio
import json
import logging
import socket
from pathlib import Path
from typing import Any

# Socket path for Roxy Talon bridge
ROXY_SOCKET_PATH = "/tmp/roxy_talon.sock"

logger = logging.getLogger(__name__)


class RoxyTalonClient:
    """Client for communicating with Roxy via Unix socket."""

    def __init__(self, socket_path: str = ROXY_SOCKET_PATH) -> None:
        """Initialize the Roxy Talon client.

        Args:
            socket_path: Path to the Roxy Unix socket.
        """
        self.socket_path = socket_path
        self._socket: socket.socket | None = None

    def _send_command(self, command: str, args: dict[str, Any]) -> dict[str, Any]:
        """Send a command to Roxy and get the response.

        Args:
            command: Command name.
            args: Command arguments.

        Returns:
            Response dictionary.
        """
        message = {"command": command, "args": args}
        message_json = json.dumps(message)
        message_bytes = message_json.encode("utf-8")
        length_bytes = len(message_bytes).to_bytes(4, byteorder="big")

        try:
            # Create socket connection
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(self.socket_path)

            # Send message
            self._socket.sendall(length_bytes + message_bytes)

            # Receive response length
            response_length_bytes = self._socket.recv(4)
            response_length = int.from_bytes(response_length_bytes, byteorder="big")

            # Receive response
            response_bytes = b""
            while len(response_bytes) < response_length:
                chunk = self._socket.recv(response_length - len(response_bytes))
                if not chunk:
                    break
                response_bytes += chunk

            response = json.loads(response_bytes.decode("utf-8"))

            return response

        except FileNotFoundError:
            logger.error(f"Roxy socket not found at {self.socket_path}. Is Roxy running?")
            return {"success": False, "error": "Roxy not running"}
        except Exception as e:
            logger.error(f"Error communicating with Roxy: {e}")
            return {"success": False, "error": str(e)}
        finally:
            if self._socket:
                self._socket.close()
                self._socket = None


# Global client instance
_client = RoxyTalonClient()


# Wake and activation commands
def roxy_wake() -> str:
    """Wake up Roxy and start listening."""
    response = _client._send_command("wake", {})
    if response.get("success"):
        return "Roxy is listening"
    return f"Error: {response.get('error', 'Unknown error')}"


def roxy_stop_listening() -> str:
    """Stop Roxy from listening."""
    response = _client._send_command("stop_listening", {})
    if response.get("success"):
        return "Roxy stopped listening"
    return f"Error: {response.get('error', 'Unknown error')}"


# Text processing
def roxy_process_text(text: str) -> str:
    """Process text through Roxy.

    Args:
        text: Text to process.

    Returns:
        Response text.
    """
    response = _client._send_command("process_text", {"text": text})
    if response.get("success"):
        return response.get("response", "No response")
    return f"Error: {response.get('error', 'Unknown error')}"


# Application control
def roxy_open_app(app: str) -> str:
    """Open an application.

    Args:
        app: Application name.

    Returns:
        Status message.
    """
    response = _client._send_command("open_app", {"name": app})
    if response.get("success"):
        return response.get("response", f"Opened {app}")
    return f"Error: {response.get('error', 'Unknown error')}"


def roxy_close_app(app: str) -> str:
    """Close an application.

    Args:
        app: Application name.

    Returns:
        Status message.
    """
    response = _client._send_command("close_app", {"name": app})
    if response.get("success"):
        return response.get("response", f"Closed {app}")
    return f"Error: {response.get('error', 'Unknown error')}"


# System commands
def roxy_volume_change(amount: int) -> str:
    """Change volume.

    Args:
        amount: Amount to change (positive or negative).

    Returns:
        Status message.
    """
    response = _client._send_command("volume_change", {"amount": amount})
    if response.get("success"):
        return response.get("response", f"Volume changed by {amount}")
    return f"Error: {response.get('error', 'Unknown error')}"


def roxy_volume_mute() -> str:
    """Mute volume."""
    response = _client._send_command("volume_mute", {})
    if response.get("success"):
        return "Volume muted"
    return f"Error: {response.get('error', 'Unknown error')}"


def roxy_volume_set(amount: int) -> str:
    """Set volume to specific level.

    Args:
        amount: Volume percentage (0-100).

    Returns:
        Status message.
    """
    response = _client._send_command("volume_set", {"amount": amount})
    if response.get("success"):
        return response.get("response", f"Volume set to {amount}")
    return f"Error: {response.get('error', 'Unknown error')}"


def roxy_brightness_change(amount: int) -> str:
    """Change screen brightness.

    Args:
        amount: Amount to change (positive or negative).

    Returns:
        Status message.
    """
    response = _client._send_command("brightness_change", {"amount": amount})
    if response.get("success"):
        return response.get("response", f"Brightness changed by {amount}")
    return f"Error: {response.get('error', 'Unknown error')}"


def roxy_system_sleep() -> str:
    """Put system to sleep."""
    response = _client._send_command("system_sleep", {})
    if response.get("success"):
        return "System going to sleep"
    return f"Error: {response.get('error', 'Unknown error')}"


def roxy_system_lock() -> str:
    """Lock the screen."""
    response = _client._send_command("system_lock", {})
    if response.get("success"):
        return "Screen locked"
    return f"Error: {response.get('error', 'Unknown error')}"


def roxy_system_shutdown() -> str:
    """Shutdown the system."""
    response = _client._send_command("system_shutdown", {})
    if response.get("success"):
        return "System shutting down"
    return f"Error: {response.get('error', 'Unknown error')}"


def roxy_web_search(query: str) -> str:
    """Perform a web search.

    Args:
        query: Search query.

    Returns:
        Status message.
    """
    response = _client._send_command("web_search", {"query": query})
    if response.get("success"):
        return response.get("response", f"Searched for {query}")
    return f"Error: {response.get('error', 'Unknown error')}"


def roxy_find_files(pattern: str) -> str:
    """Find files matching pattern.

    Args:
        pattern: File name pattern.

    Returns:
        Status message.
    """
    response = _client._send_command("find_files", {"pattern": pattern})
    if response.get("success"):
        return response.get("response", f"Found files matching {pattern}")
    return f"Error: {response.get('error', 'Unknown error')}"


# Git commands
def roxy_git_status() -> str:
    """Get git status."""
    return roxy_process_text("git status")


def roxy_git_commit() -> str:
    """Commit changes."""
    return roxy_process_text("git commit")


def roxy_git_push() -> str:
    """Push changes."""
    return roxy_process_text("git push")


def roxy_git_pull() -> str:
    """Pull changes."""
    return roxy_process_text("git pull")


# Development commands
def roxy_start_coding() -> str:
    """Start coding session."""
    return roxy_process_text("start coding session")


def roxy_code_review() -> str:
    """Run code review."""
    return roxy_process_text("review my code")


def roxy_new_project(name: str) -> str:
    """Create new project.

    Args:
        name: Project name.

    Returns:
        Status message.
    """
    return roxy_process_text(f"create new project named {name}")


def roxy_open_project(name: str) -> str:
    """Open project.

    Args:
        name: Project name.

    Returns:
        Status message.
    """
    return roxy_process_text(f"open project {name}")


def roxy_run_tests() -> str:
    """Run tests."""
    return roxy_process_text("run tests")


def roxy_test_coverage() -> str:
    """Get test coverage."""
    return roxy_process_text("test coverage")


def roxy_get_status() -> dict[str, Any]:
    """Get Roxy status.

    Returns:
        Status dictionary.
    """
    return _client._send_command("get_status", {})


# Export all functions for Talon
__all__ = [
    "roxy_wake",
    "roxy_stop_listening",
    "roxy_process_text",
    "roxy_open_app",
    "roxy_close_app",
    "roxy_volume_change",
    "roxy_volume_mute",
    "roxy_volume_set",
    "roxy_brightness_change",
    "roxy_system_sleep",
    "roxy_system_lock",
    "roxy_system_shutdown",
    "roxy_web_search",
    "roxy_find_files",
    "roxy_git_status",
    "roxy_git_commit",
    "roxy_git_push",
    "roxy_git_pull",
    "roxy_start_coding",
    "roxy_code_review",
    "roxy_new_project",
    "roxy_open_project",
    "roxy_run_tests",
    "roxy_test_coverage",
    "roxy_get_status",
]
