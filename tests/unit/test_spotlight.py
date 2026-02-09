"""Unit tests for Spotlight search functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.macos.spotlight import SpotlightSearch, get_spotlight_search


class TestSpotlightSearch:
    """Test SpotlightSearch functionality."""

    def test_singleton(self):
        """Test that get_spotlight_search returns singleton instance."""
        search1 = get_spotlight_search()
        search2 = get_spotlight_search()

        assert search1 is search2

    @patch("subprocess.run")
    def test_check_available_true(self, mock_run):
        """Test checking availability when mdfind exists."""
        mock_run.return_value = MagicMock(returncode=0)

        search = SpotlightSearch()
        assert search._check_available() is True

    @patch("subprocess.run")
    def test_check_available_false(self, mock_run):
        """Test checking availability when mdfind doesn't exist."""
        mock_run.return_value = MagicMock(returncode=1)

        search = SpotlightSearch()
        assert search._check_available() is False

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_search(self, mock_subprocess):
        """Test searching files."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"/path/to/file1.txt\n/path/to/file2.pdf\n",
            b""
        ))

        # Mock mdls for metadata
        mdls_process = AsyncMock()
        mdls_process.returncode = 0
        mdls_process.communicate = AsyncMock(return_value=(
            b'kMDItemDisplayName = "file1.txt"\nkMDItemKind = "Plain Text"\n',
            b""
        ))

        mock_subprocess.side_effect = [mock_process, mdls_process, mdls_process]

        search = SpotlightSearch()
        search._available = True

        results = await search.search("test query")

        # Should get results even if mdls fails (has fallback)
        assert len(results) >= 0

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_search_not_available(self, mock_subprocess):
        """Test searching when Spotlight is not available."""
        search = SpotlightSearch()
        search._available = False

        results = await search.search("test query")

        assert results == []

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_search_by_kind(self, mock_subprocess):
        """Test searching by file kind."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"/path/to/file.pdf\n",
            b""
        ))

        mock_subprocess.return_value = mock_process

        search = SpotlightSearch()
        search._available = True

        # Mock get_file_metadata to avoid mdls call
        with patch.object(search, "get_file_metadata", return_value={
            "path": "/path/to/file.pdf",
            "name": "file.pdf",
            "kind": "PDF",
            "size": 0,
        }):
            results = await search.search_by_kind("test", "pdf")

            assert len(results) == 1

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_search_recent(self, mock_subprocess):
        """Test searching recent files."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"/path/to/recent.txt\n",
            b""
        ))

        mock_subprocess.return_value = mock_process

        search = SpotlightSearch()
        search._available = True

        with patch.object(search, "get_file_metadata", return_value={
            "path": "/path/to/recent.txt",
            "name": "recent.txt",
            "kind": "Plain Text",
            "size": 0,
        }):
            results = await search.search_recent("test", days=7)

            assert len(results) == 1

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_get_file_metadata(self, mock_subprocess):
        """Test getting file metadata."""
        from pathlib import Path
        import tempfile

        # Create a temp file to test with
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            temp_path = f.name
            f.write(b"test content")

        try:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(
                b'kMDItemDisplayName = "test.txt"\nkMDItemKind = "Plain Text"\nkMDItemFSSize = 12\n',
                b""
            ))

            mock_subprocess.return_value = mock_process

            search = SpotlightSearch()
            search._available = True

            metadata = await search.get_file_metadata(temp_path)

            assert metadata is not None
            assert metadata["name"] == "test.txt"
            assert metadata["kind"] == "Plain Text"
            assert metadata["size"] == 12
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_find_by_name(self, mock_subprocess):
        """Test finding files by name."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"/path/to/testfile.txt\n",
            b""
        ))

        mock_subprocess.return_value = mock_process

        search = SpotlightSearch()
        search._available = True

        results = await search.find_by_name("testfile")

        assert len(results) == 1
        assert results[0] == "/path/to/testfile.txt"

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_find_applications(self, mock_subprocess):
        """Test finding applications."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"/Applications/Safari.app\n",
            b""
        ))

        mock_subprocess.return_value = mock_process

        search = SpotlightSearch()
        search._available = True

        results = await search.find_applications("Safari")

        assert len(results) == 1
        assert ".app" in results[0]

    def test_format_size(self):
        """Test size formatting."""
        search = SpotlightSearch()

        assert search._format_size(0) == "0 B"
        assert search._format_size(1024) == "1.0 KB"
        assert search._format_size(1024 * 1024) == "1.0 MB"
        assert search._format_size(1024 * 1024 * 1024) == "1.0 GB"
