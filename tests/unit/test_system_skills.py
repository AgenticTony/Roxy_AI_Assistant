"""Unit tests for system skills."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.skills.base import Permission, SkillContext, StubMemoryManager
from roxy.skills.system.app_launcher import AppLauncherSkill
from roxy.skills.system.file_search import FileSearchSkill
from roxy.skills.system.window_manager import WindowManagerSkill
from roxy.skills.system.system_info import SystemInfoSkill
from roxy.skills.system.clipboard import ClipboardSkill
from roxy.skills.system.shortcuts import ShortcutsSkill


# Fixtures
@pytest.fixture
def skill_context():
    """Create a skill context for testing."""
    from roxy.config import LocalLLMConfig, CloudLLMConfig, PrivacyConfig, RoxyConfig

    config = RoxyConfig(
        name="TestRoxy",
        data_dir="/tmp/test_roxy",
        llm_local=LocalLLMConfig(),
        llm_cloud=CloudLLMConfig(),
        privacy=PrivacyConfig(),
    )

    return SkillContext(
        user_input="test input",
        intent="test",
        parameters={},
        memory=StubMemoryManager(),
        config=config,
        conversation_history=[],
    )


class TestAppLauncherSkill:
    """Test AppLauncherSkill."""

    def test_skill_metadata(self):
        """Test skill has correct metadata."""
        skill = AppLauncherSkill()

        assert skill.name == "app_launcher"
        assert "launch" in skill.triggers
        assert Permission.APPLESCRIPT in skill.permissions

    @pytest.mark.asyncio
    @patch("roxy.skills.system.app_launcher.AppLauncherSkill._launch_app")
    async def test_execute_success(self, mock_launch, skill_context):
        """Test successful app launch."""
        mock_launch.return_value = True

        skill_context.user_input = "open Safari"
        skill_context.intent = "open safari"

        skill = AppLauncherSkill()
        result = await skill.execute(skill_context)

        assert result.success is True
        assert "Opening" in result.response_text
        assert result.speak is True

    @pytest.mark.asyncio
    @patch("roxy.skills.system.app_launcher.AppLauncherSkill._launch_app")
    async def test_execute_failure(self, mock_launch, skill_context):
        """Test failed app launch."""
        mock_launch.return_value = False

        skill_context.user_input = "open NonExistentApp"

        skill = AppLauncherSkill()
        result = await skill.execute(skill_context)

        assert result.success is False

    @pytest.mark.asyncio
    @patch("roxy.skills.system.app_launcher.AppLauncherSkill._launch_app")
    async def test_execute_no_app_name(self, mock_launch, skill_context):
        """Test execution with no app name detected."""
        skill_context.user_input = "open something please"
        skill_context.intent = "open"

        skill = AppLauncherSkill()
        result = await skill.execute(skill_context)

        assert result.success is False
        assert result.speak is False

    def test_can_handle(self):
        """Test intent matching."""
        skill = AppLauncherSkill()

        confidence = skill.can_handle("open Safari", {})
        assert confidence > 0.5

        confidence = skill.can_handle("unrelated input", {})
        assert confidence == 0.0


class TestFileSearchSkill:
    """Test FileSearchSkill."""

    def test_skill_metadata(self):
        """Test skill has correct metadata."""
        skill = FileSearchSkill()

        assert skill.name == "file_search"
        assert "find file" in skill.triggers
        assert Permission.FILESYSTEM in skill.permissions

    @pytest.mark.asyncio
    @patch("roxy.macos.spotlight.SpotlightSearch.search")
    async def test_execute_success(self, mock_search, skill_context):
        """Test successful file search."""
        mock_search.return_value = [
            {"path": "/path/to/file.txt", "name": "file.txt", "kind": "Plain Text"}
        ]

        skill_context.user_input = "find file about test"
        skill_context.intent = "find file about test"

        skill = FileSearchSkill()
        result = await skill.execute(skill_context)

        assert result.success is True
        assert "Found" in result.response_text

    @pytest.mark.asyncio
    @patch("roxy.macos.spotlight.SpotlightSearch.search")
    async def test_execute_no_results(self, mock_search, skill_context):
        """Test file search with no results."""
        mock_search.return_value = []

        skill_context.user_input = "find file nonexistent"

        skill = FileSearchSkill()
        result = await skill.execute(skill_context)

        assert result.success is True
        assert "No files found" in result.response_text

    @pytest.mark.asyncio
    @patch("roxy.macos.spotlight.SpotlightSearch.search")
    async def test_extract_query(self, mock_search, skill_context):
        """Test query extraction from user input."""
        mock_search.return_value = []

        skill_context.user_input = "find file about my document"
        skill_context.intent = "find file about my document"

        skill = FileSearchSkill()
        await skill.execute(skill_context)

        # Should have called search with extracted query
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert "document" in call_args[0][0] or "my" in call_args[0][0]


class TestWindowManagerSkill:
    """Test WindowManagerSkill."""

    def test_skill_metadata(self):
        """Test skill has correct metadata."""
        skill = WindowManagerSkill()

        assert skill.name == "window_manager"
        assert "coding layout" in skill.triggers
        assert Permission.SHELL in skill.permissions

    @pytest.mark.asyncio
    @patch("roxy.macos.hammerspoon.HammerspoonClient._check_available")
    async def test_execute_with_hammerspoon(self, mock_available, skill_context):
        """Test window layout with Hammerspoon available."""
        mock_available.return_value = True

        skill_context.user_input = "set up coding layout"

        skill = WindowManagerSkill()

        # Mock the layout application
        with patch.object(skill, "_apply_layout_hammerspoon", return_value=True):
            result = await skill.execute(skill_context)
            assert result.success is True

    @pytest.mark.asyncio
    @patch("roxy.macos.hammerspoon.HammerspoonClient._check_available")
    async def test_execute_without_hammerspoon(self, mock_available, skill_context):
        """Test window layout without Hammerspoon."""
        mock_available.return_value = False

        skill_context.user_input = "coding layout"

        skill = WindowManagerSkill()
        result = await skill.execute(skill_context)

        # Should return a result about Hammerspoon
        assert result is not None


class TestSystemInfoSkill:
    """Test SystemInfoSkill."""

    def test_skill_metadata(self):
        """Test skill has correct metadata."""
        skill = SystemInfoSkill()

        assert skill.name == "system_info"
        assert "running" in skill.triggers
        assert Permission.APPLESCRIPT in skill.permissions

    @pytest.mark.asyncio
    @patch("roxy.macos.applescript.AppleScriptRunner.get_running_apps")
    async def test_get_running_apps(self, mock_apps, skill_context):
        """Test getting running applications."""
        mock_apps.return_value = [
            {"name": "Safari", "bundle_id": "com.apple.Safari", "frontmost": True},
            {"name": "Finder", "bundle_id": "com.apple.Finder", "frontmost": False},
        ]

        skill_context.user_input = "what's running"
        skill_context.intent = "what's running"

        skill = SystemInfoSkill()
        result = await skill.execute(skill_context)

        assert result.success is True
        assert "2 application" in result.response_text or "applications" in result.response_text

    @pytest.mark.asyncio
    @patch("roxy.macos.pyobjc_bridge.MacOSBridge.get_system_info")
    async def test_get_system_info(self, mock_info, skill_context):
        """Test getting system information."""
        mock_info.return_value = {
            "os_version": "macOS 14.0",
            "computer_name": "Test Mac",
            "cpu": "Apple M1",
            "ram_gb": 16.0,
        }

        skill_context.user_input = "system info"
        skill_context.intent = "system info"

        skill = SystemInfoSkill()
        result = await skill.execute(skill_context)

        assert result.success is True
        assert "macOS" in result.response_text or "System" in result.response_text


class TestClipboardSkill:
    """Test ClipboardSkill."""

    def test_skill_metadata(self):
        """Test skill has correct metadata."""
        skill = ClipboardSkill()

        assert skill.name == "clipboard"
        assert "clipboard" in skill.triggers
        assert Permission.APPLESCRIPT in skill.permissions

    @pytest.mark.asyncio
    @patch("roxy.macos.applescript.AppleScriptRunner.get_clipboard")
    async def test_read_clipboard(self, mock_get, skill_context):
        """Test reading clipboard."""
        mock_get.return_value = "test clipboard content"

        skill_context.user_input = "what's in my clipboard"
        skill_context.intent = "what's in my clipboard"

        skill = ClipboardSkill()
        result = await skill.execute(skill_context)

        assert result.success is True
        assert "clipboard" in result.response_text.lower()

    @pytest.mark.asyncio
    @patch("roxy.macos.applescript.AppleScriptRunner.set_clipboard")
    async def test_copy_to_clipboard(self, mock_set, skill_context):
        """Test copying to clipboard."""
        mock_set.return_value = True

        skill_context.user_input = "copy that"
        skill_context.intent = "copy"
        skill_context.parameters = {"content": "test content"}
        skill_context.conversation_history = [
            {"role": "user", "content": "previous message"},
            {"role": "assistant", "content": "test content"},
        ]

        skill = ClipboardSkill()
        result = await skill.execute(skill_context)

        assert result.success is True


class TestShortcutsSkill:
    """Test ShortcutsSkill."""

    def test_skill_metadata(self):
        """Test skill has correct metadata."""
        skill = ShortcutsSkill()

        assert skill.name == "shortcuts"
        assert "run shortcut" in skill.triggers
        assert Permission.SHELL in skill.permissions

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_list_shortcuts(self, mock_subprocess, skill_context):
        """Test listing shortcuts."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"Test Shortcut 1\nTest Shortcut 2\n",
            b""
        ))
        mock_subprocess.return_value = mock_process

        skill_context.user_input = "list shortcuts"
        skill_context.intent = "list shortcuts"

        skill = ShortcutsSkill()
        result = await skill.execute(skill_context)

        assert result.success is True
        assert "shortcut" in result.response_text.lower()

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_run_shortcut(self, mock_subprocess, skill_context):
        """Test running a shortcut."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_subprocess.return_value = mock_process

        skill_context.user_input = "run shortcut Test Shortcut"
        skill_context.intent = "run shortcut"
        skill_context.parameters = {"shortcut_name": "Test Shortcut"}

        skill = ShortcutsSkill()
        result = await skill.execute(skill_context)

        assert result.success is True
        assert "Running" in result.response_text
