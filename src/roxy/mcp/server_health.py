"""MCP Server health checking functionality.

Provides health check methods for MCP server connections.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

# Try to import aiohttp for HTTP health checks
try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from .server_config import ServerConnection, ServerStatus, ServerTransport

logger = logging.getLogger(__name__)


class ServerHealthChecker:
    """
    Health checker for MCP servers.

    Provides methods to check if servers are running and responsive.
    """

    @staticmethod
    async def check_server_health(connection: ServerConnection) -> bool:
        """Check if a server connection is healthy and responding.

        Args:
            connection: ServerConnection to check.

        Returns:
            True if server is healthy.
        """
        if connection.status != ServerStatus.RUNNING:
            return False

        # For STDIO servers, check if process is still alive
        if connection.config.transport == ServerTransport.STDIO:
            if connection.process:
                # Process is dead if returncode is not None
                if connection.process.returncode is not None:
                    logger.warning(
                        f"Server {connection.name} process exited (code: {connection.process.returncode})"
                    )
                    connection.status = ServerStatus.ERROR
                    connection.error = f"Process exited with code {connection.process.returncode}"
                    return False
            return True

        # If server has a health check URL, verify it
        if connection.config.health_check:
            return await ServerHealthChecker._check_http_health(connection)

        return True

    @staticmethod
    async def _check_http_health(connection: ServerConnection) -> bool:
        """Check server health via HTTP endpoint.

        Args:
            connection: ServerConnection with health_check URL.

        Returns:
            True if health check passes.
        """
        try:
            if HAS_AIOHTTP:
                return await ServerHealthChecker._check_with_aiohttp(connection)
            else:
                return await ServerHealthChecker._check_with_socket(connection)
        except TimeoutError:
            logger.warning(f"Health check timed out for {connection.name}")
            return False
        except Exception as e:
            logger.error(f"Health check failed for {connection.name}: {e}")
            return False

    @staticmethod
    async def _check_with_aiohttp(connection: ServerConnection) -> bool:
        """Check health using aiohttp.

        Args:
            connection: ServerConnection with health_check URL.

        Returns:
            True if health check passes.
        """
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(connection.config.health_check) as response:
                if 200 <= response.status < 300:
                    logger.debug(f"Health check passed for {connection.name}: {response.status}")
                    return True
                else:
                    logger.warning(f"Health check failed for {connection.name}: {response.status}")
                    return False

    @staticmethod
    async def _check_with_socket(connection: ServerConnection) -> bool:
        """Check health using socket connection (fallback).

        Args:
            connection: ServerConnection with health_check URL.

        Returns:
            True if connection succeeds.
        """
        # Parse URL to get host and port
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
        logger.debug(f"Health check passed for {connection.name} (socket)")
        return True

    @staticmethod
    def get_status_summary(connections: dict[str, ServerConnection]) -> dict[str, dict]:
        """Get a summary of all server statuses.

        Args:
            connections: Dict of server name to ServerConnection.

        Returns:
            Dict with status information for each server.
        """
        summary = {}
        for name, conn in connections.items():
            summary[name] = {
                "status": conn.status.value,
                "error": conn.error,
                "started_at": conn.started_at.isoformat() if conn.started_at else None,
            }
        return summary
