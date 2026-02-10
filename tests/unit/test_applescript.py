"""Unit tests for AppleScript runner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.macos.applescript import AppleScriptRunner, get_applescript_runner


class TestAppleScriptRunner:
    """Test AppleScriptRunner functionality."""

    def test_singleton(self):
        """Test that get_applescript_runner returns singleton instance."""
        runner1 = get_applescript_runner()
        runner2 = get_applescript_runner()

        assert runner1 is runner2

    @patch("subprocess.run")
    def test_check_available_true(self, mock_run):
        """Test checking availability when osascript exists."""
        mock_run.return_value = MagicMock(returncode=0)

        runner = AppleScriptRunner()
        assert runner._check_available() is True
        assert runner._available is True

    @patch("subprocess.run")
    def test_check_available_false(self, mock_run):
        """Test checking availability when osascript doesn't exist."""
        mock_run.return_value = MagicMock(returncode=1)

        runner = AppleScriptRunner()
        assert runner._check_available() is False
        assert runner._available is False

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_run_success(self, mock_subprocess):
        """Test running AppleScript successfully."""
        # Mock the process
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"test output\n", b""))

        # Mock the create_subprocess_exec to return our mock
        mock_subprocess.return_value = mock_process

        runner = AppleScriptRunner()
        result = await runner.run('tell application "System Events" to return "test"')

        assert result == "test output"

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_run_error(self, mock_subprocess):
        """Test running AppleScript with error."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"AppleScript error"))

        mock_subprocess.return_value = mock_process

        runner = AppleScriptRunner()
        runner._available = True  # Skip availability check

        with pytest.raises(Exception):  # subprocess.CalledProcessError
            await runner.run("invalid script")

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_run_jxa(self, mock_subprocess):
        """Test running JXA code."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'{"test": "value"}', b""))

        mock_subprocess.return_value = mock_process

        runner = AppleScriptRunner()
        runner._available = True

        result = await runner.run_jxa("JSON.stringify({test: 'value'})")

        assert result == '{"test": "value"}'

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_get_running_apps(self, mock_subprocess):
        """Test getting running applications."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(
                b'[{"name":"Finder","bundleID":"com.apple.finder","frontmost":true}]',
                b"",
            )
        )

        mock_subprocess.return_value = mock_process
        runner = AppleScriptRunner()
        runner._available = True

        apps = await runner.get_running_apps()

        assert len(apps) == 1
        assert apps[0]["name"] == "Finder"
        assert apps[0]["bundle_id"] == "com.apple.finder"
        assert apps[0]["frontmost"] is True

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_get_frontmost_app(self, mock_subprocess):
        """Test getting frontmost application."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Safari||com.apple.Safari", b""))

        mock_subprocess.return_value = mock_process
        runner = AppleScriptRunner()
        runner._available = True

        app = await runner.get_frontmost_app()

        assert app["name"] == "Safari"
        assert app["bundle_id"] == "com.apple.Safari"

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_send_notification(self, mock_subprocess):
        """Test sending notification."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        mock_subprocess.return_value = mock_process
        runner = AppleScriptRunner()
        runner._available = True

        result = await runner.send_notification("Test Title", "Test Message")

        assert result is True

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_get_clipboard(self, mock_subprocess):
        """Test getting clipboard content."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"test clipboard content", b""))

        mock_subprocess.return_value = mock_process
        runner = AppleScriptRunner()
        runner._available = True

        content = await runner.get_clipboard()

        assert content == "test clipboard content"

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_set_clipboard(self, mock_subprocess):
        """Test setting clipboard content."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"success", b""))

        mock_subprocess.return_value = mock_process
        runner = AppleScriptRunner()
        runner._available = True

        result = await runner.set_clipboard("new content")

        assert result is True

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_get_mail_unread_count(self, mock_subprocess):
        """Test getting mail unread count."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"5", b""))

        mock_subprocess.return_value = mock_process
        runner = AppleScriptRunner()
        runner._available = True

        count = await runner.get_mail_unread_count()

        assert count == 5

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_get_calendar_events_today(self, mock_subprocess):
        """Test getting today's calendar events."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b'[{"summary":"Test Event","startDate":"2024-01-01T10:00:00"}]', b"")
        )

        mock_subprocess.return_value = mock_process
        runner = AppleScriptRunner()
        runner._available = True

        events = await runner.get_calendar_events_today()

        assert len(events) == 1
        assert events[0]["summary"] == "Test Event"

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_create_note(self, mock_subprocess):
        """Test creating a note."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        mock_subprocess.return_value = mock_process
        runner = AppleScriptRunner()
        runner._available = True

        result = await runner.create_note("Test Note", "Note content here")

        assert result is True
