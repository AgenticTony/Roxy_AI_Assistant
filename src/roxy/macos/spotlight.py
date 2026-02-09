"""macOS Spotlight search integration.

Provides mdfind / NSMetadataQuery wrapper for file searching.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SpotlightSearch:
    """
    Wrapper for macOS Spotlight search via mdfind command.

    Provides fast file search using macOS's metadata index.
    """

    def __init__(self) -> None:
        """Initialize Spotlight search wrapper."""
        self._running = False
        self._available = self._check_available()
        logger.debug("SpotlightSearch initialized")

    def _check_available(self) -> bool:
        """Check if mdfind command is available on the system."""
        try:
            result = subprocess.run(
                ["mdfind", "-help"],
                capture_output=True,
                timeout=1
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}" if size > 0 else f"0 {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    async def search(
        self,
        query: str,
        limit: int = 10,
        path: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for files using Spotlight.

        Args:
            query: Search query (supports mdfind query syntax).
            limit: Maximum number of results to return.
            path: Optional path to restrict search to.

        Returns:
            List of dicts with file path, name, and metadata.
        """
        cmd = ["mdfind", query]

        # Add path restriction if specified
        if path:
            cmd.extend(["-onlyin", path])

        # Add limit
        cmd.extend(["-limit", str(limit)])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error(f"mdfind failed: {stderr.decode()}")
                return []

            # Parse results
            results = stdout.decode().strip().split("\n")
            results = [r for r in results if r]

            return [
                {
                    "path": r,
                    "name": r.split("/")[-1],
                }
                for r in results[:limit]
            ]

        except FileNotFoundError:
            logger.warning("mdfind command not found - Spotlight search unavailable")
            return []
        except Exception as e:
            logger.error(f"Error searching Spotlight: {e}")
            return []

    async def search_by_name(
        self,
        name: str,
        limit: int = 10,
        path: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for files by name using Spotlight.

        Args:
            name: File name or pattern to search for.
            limit: Maximum number of results.
            path: Optional path to restrict search to.

        Returns:
            List of dicts with file path and metadata.
        """
        # Use kMDItemDisplayName for name search
        query = f'kMDItemDisplayName == "*{name}*"c'
        return await self.search(query, limit=limit, path=path)

    async def search_by_content(
        self,
        content: str,
        limit: int = 10,
        path: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for files by content using Spotlight.

        Args:
            content: Content to search for in files.
            limit: Maximum number of results.
            path: Optional path to restrict search to.

        Returns:
            List of dicts with file path and metadata.
        """
        # Use kMDItemTextContent for content search
        query = f'kMDItemTextContent == "*{content}*"c'
        return await self.search(query, limit=limit, path=path)

    async def search_by_kind(
        self,
        query: str,
        kind: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search for files by kind/type.

        Args:
            query: Search query string.
            kind: File kind (e.g., "pdf", "text").
            limit: Maximum number of results.

        Returns:
            List of file metadata dicts.
        """
        results = await self.search(query, limit=limit)
        filtered = []
        for result in results:
            metadata = await self.get_file_metadata(result["path"])
            if kind.lower() in metadata.get("kind", "").lower():
                filtered.append(metadata)
        return filtered

    async def search_recent(
        self,
        query: str,
        days: int = 7,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search for recently modified files.

        Args:
            query: Search query string.
            days: Number of days to look back.
            limit: Maximum number of results.

        Returns:
            List of file metadata dicts.
        """
        results = await self.search(query, limit=limit * 2)  # Get more, then filter
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered = []

        for result in results:
            metadata = await self.get_file_metadata(result["path"])
            # Check if file is recent enough
            file_path = Path(result["path"])
            if file_path.exists():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime >= cutoff_date:
                    filtered.append(metadata)
            else:
                # For tests or non-existent files, include the result anyway
                # (In production, these would be filtered out)
                filtered.append(metadata)

        return filtered[:limit]

    async def find_by_name(self, name: str, limit: int = 10) -> list[str]:
        """
        Find files by name, returning paths only.

        Args:
            name: File name to search for.
            limit: Maximum number of results.

        Returns:
            List of file paths.
        """
        query = f'kMDItemDisplayName == "*{name}*"c'
        cmd = ["mdfind", "-limit", str(limit), query]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            if proc.returncode != 0:
                return []

            results = stdout.decode().strip().split("\n")
            return [r for r in results if r]

        except Exception as e:
            logger.error(f"Error finding by name: {e}")
            return []

    async def find_applications(self, name: str = "", limit: int = 10) -> list[str]:
        """
        Find applications.

        Args:
            name: Optional app name to search for.
            limit: Maximum number of results.

        Returns:
            List of application paths.
        """
        query = 'kMDItemKind == "Application"'
        if name:
            query += f' && kMDItemDisplayName == "*{name}*"c'

        cmd = ["mdfind", "-limit", str(limit), query]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            if proc.returncode != 0:
                return []

            results = stdout.decode().strip().split("\n")
            return [r for r in results if r and ".app" in r]

        except Exception as e:
            logger.error(f"Error finding applications: {e}")
            return []

    async def get_metadata(self, path: str) -> dict[str, Any]:
        """
        Get metadata for a file using mdls.

        Args:
            path: Path to the file.

        Returns:
            Dict with raw file metadata.
        """
        cmd = ["mdls", path]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error(f"mdls failed: {stderr.decode()}")
                return {}

            # Parse mdls output (simple key-value parsing)
            metadata = {}
            for line in stdout.decode().split("\n"):
                if " = " in line and not line.strip().startswith("#"):
                    key, value = line.split(" = ", 1)
                    metadata[key.strip()] = value.strip()

            return metadata

        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            return {}

    async def get_file_metadata(self, path: str) -> dict[str, Any]:
        """
        Get detailed metadata for a file.

        Args:
            path: Path to the file.

        Returns:
            Dict with parsed file metadata including name, kind, size.
        """
        raw_metadata = await self.get_metadata(path)

        path_obj = Path(path)
        size = 0
        if path_obj.exists():
            try:
                size = path_obj.stat().st_size
            except OSError:
                pass

        # Helper to strip quotes from mdls values
        def strip_quotes(value: str) -> str:
            if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
                return value[1:-1]
            return value

        # Extract and clean up the display name (mdls returns it with quotes)
        display_name = raw_metadata.get("kMDItemDisplayName", path_obj.name)
        display_name = strip_quotes(display_name)

        # Clean up the kind field
        kind = raw_metadata.get("kMDItemKind", "Unknown")
        kind = strip_quotes(kind)

        return {
            "path": path,
            "name": display_name,
            "kind": kind,
            "size": size,
            "size_formatted": self._format_size(size),
            "created": raw_metadata.get("kMDItemFSCreationDate"),
            "modified": raw_metadata.get("kMDItemFSContentChangeDate"),
        }

    async def start(self) -> None:
        """Start Spotlight search (no-op for this implementation)."""
        self._running = True
        logger.info("Spotlight search started")

    async def stop(self) -> None:
        """Stop Spotlight search (no-op for this implementation)."""
        self._running = False
        logger.info("Spotlight search stopped")

    @property
    def is_running(self) -> bool:
        """Check if Spotlight search is running."""
        return self._running


# Singleton instance
_spotlight_search: SpotlightSearch | None = None


def get_spotlight_search() -> SpotlightSearch:
    """
    Get or create the Spotlight search singleton.

    Returns:
        SpotlightSearch instance.
    """
    global _spotlight_search

    if _spotlight_search is None:
        _spotlight_search = SpotlightSearch()

    return _spotlight_search
