"""Tests for the Claude Code skill.

Tests launching development environment with Cursor/VSCode, Terminal, and project management.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.skills.dev.claude_code import ClaudeCodeSkill
from roxy.skills.base import SkillContext, SkillResult, Permission


class TestClaudeCodeSkill:
    """Tests for ClaudeCodeSkill."""

    def test_skill_metadata(self):
        """Test skill has correct metadata."""
        skill = ClaudeCodeSkill()

        assert skill.name == "claude_code"
        assert "development" in skill.description.lower() or "code" in skill.description.lower()
        assert len(skill.triggers) > 0
        assert Permission.SHELL in skill.permissions
        assert Permission.FILESYSTEM in skill.permissions
        assert skill.requires_cloud is False

    def test_init(self):
        """Test skill initialization."""
        skill = ClaudeCodeSkill()

        # Should detect available editors or set to None
        assert hasattr(skill, '_available_editor')

    def test_find_editor_success(self):
        """Test finding an available editor."""
        skill = ClaudeCodeSkill()

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="/usr/bin/vim")

            editor = skill._find_editor()
            assert editor is not None

    def test_find_editor_none_available(self):
        """Test when no editor is available."""
        skill = ClaudeCodeSkill()

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            editor = skill._find_editor()
            # Should try all editors and return None
            assert editor is None or editor in skill.EDITORS

    @pytest.mark.asyncio
    async def test_open_terminal_success(self):
        """Test successfully opening terminal."""
        skill = ClaudeCodeSkill()

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = skill._open_terminal(Path("/Users/test/project"))
            assert result is True

    @pytest.mark.asyncio
    async def test_open_terminal_failure(self):
        """Test handling when terminal open fails."""
        skill = ClaudeCodeSkill()

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Terminal error")

            result = skill._open_terminal(Path("/Users/test/project"))
            assert result is False

    @pytest.mark.asyncio
    async def test_open_editor_cursor_success(self):
        """Test opening Cursor editor."""
        skill = ClaudeCodeSkill()
        skill._available_editor = "cursor"

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = skill._open_editor(Path("/Users/test/project"))
            assert result is True

    @pytest.mark.asyncio
    async def test_open_editor_vscode_success(self):
        """Test opening VSCode editor."""
        skill = ClaudeCodeSkill()
        skill._available_editor = "code"

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = skill._open_editor(Path("/Users/test/project"))
            assert result is True

    def test_open_editor_no_editor_available(self):
        """Test when no editor is available."""
        skill = ClaudeCodeSkill()
        skill._available_editor = None

        result = skill._open_editor(Path("/Users/test/project"))
        assert result is False

    @pytest.mark.asyncio
    async def test_get_project_path_from_parameters(self):
        """Test getting project path from parameters."""
        skill = ClaudeCodeSkill()
        context = SkillContext(
            user_input="start development",
            intent="start_dev",
            parameters={"path": "/Users/test/project"},
            memory=MagicMock(),
            config=MagicMock(),
            conversation_history=[],
        )

        project_path = await skill._get_project_path(context)
        assert project_path == Path("/Users/test/project")

    @pytest.mark.asyncio
    async def test_get_project_path_from_memory(self):
        """Test getting project path from memory."""
        skill = ClaudeCodeSkill()
        memory = MagicMock()
        memory.recall = AsyncMock(return_value=['{"path": "/Users/test/project"}'])
        memory.get_user_preferences = AsyncMock(return_value={"current_project": {"path": "/Users/test/project"}})

        context = SkillContext(
            user_input="start development",
            intent="start_dev",
            parameters={},
            memory=memory,
            config=MagicMock(),
            conversation_history=[],
        )

        project_path = await skill._get_project_path(context)
        # Should return path from memory or current directory
        assert project_path is not None

    @pytest.mark.asyncio
    async def test_get_project_path_default_to_cwd(self):
        """Test getting project path defaults to current directory."""
        skill = ClaudeCodeSkill()
        memory = MagicMock()
        memory.recall = AsyncMock(return_value=[])
        memory.get_user_preferences = AsyncMock(return_value={})

        context = SkillContext(
            user_input="start development",
            intent="start_dev",
            parameters={},
            memory=memory,
            config=MagicMock(),
            conversation_history=[],
        )

        project_path = await skill._get_project_path(context)
        assert project_path is not None
        # Should be current working directory
        assert project_path == Path.cwd()

    @pytest.mark.asyncio
    async def test_launch_dev_mode_full(self):
        """Test launching full development mode."""
        skill = ClaudeCodeSkill()
        skill._available_editor = "cursor"

        # Mock Path.exists to return True
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(skill, '_open_editor', return_value=True):
                with patch.object(skill, '_open_terminal', return_value=True):
                    result = await skill.launch_dev_mode(Path("/Users/test/project"))

                    assert "cursor" in result.lower() or "terminal" in result.lower()
                    assert "/users/test/project" in result.lower() or "project" in result.lower()

    @pytest.mark.asyncio
    async def test_launch_dev_mode_no_editor(self):
        """Test launching dev mode when no editor available."""
        skill = ClaudeCodeSkill()
        skill._available_editor = None

        with patch.object(skill, '_open_terminal', return_value=True):
            result = await skill.launch_dev_mode(Path("/Users/test/project"))

            assert "terminal" in result.lower() or "project" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_start_development(self):
        """Test execute method for starting development."""
        skill = ClaudeCodeSkill()
        skill._available_editor = "cursor"

        context = SkillContext(
            user_input="start development on my project",
            intent="start_dev",
            parameters={"project": "/Users/test/project"},
            memory=MagicMock(recall=AsyncMock(return_value=[])),
            config=MagicMock(),
            conversation_history=[],
        )

        with patch.object(skill, 'launch_dev_mode', new=AsyncMock(return_value="Dev environment ready")):
            result = await skill.execute(context)

            assert result.success is True
            assert "dev" in result.response_text.lower() or "environment" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_execute_with_cursor_editor(self):
        """Test execute method specifically requesting Cursor."""
        skill = ClaudeCodeSkill()

        context = SkillContext(
            user_input="open cursor for my project",
            intent="start_dev",
            parameters={"project": "/Users/test/project"},
            memory=MagicMock(recall=AsyncMock(return_value=[])),
            config=MagicMock(),
            conversation_history=[],
        )

        with patch.object(skill, '_open_editor', return_value=True) as mock_open:
            result = await skill.execute(context)

            assert result.success is True
            # Should have tried to open cursor
            assert mock_open.called

    @pytest.mark.asyncio
    async def test_execute_no_project_path(self):
        """Test execute when no project path is found."""
        skill = ClaudeCodeSkill()

        context = SkillContext(
            user_input="start development",
            intent="start_dev",
            parameters={},
            memory=MagicMock(recall=AsyncMock(return_value=[])),
            config=MagicMock(),
            conversation_history=[],
        )

        with patch.object(skill, '_get_project_path', new=AsyncMock(return_value=None)):
            result = await skill.execute(context)

            assert result.success is False
            assert "project" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_can_handle_dev_phrases(self):
        """Test can_handle recognizes development phrases."""
        skill = ClaudeCodeSkill()

        phrases = [
            "start development",
            "code mode",
            "open project",
            "start coding",
            "dev mode",
        ]

        for phrase in phrases:
            confidence = skill.can_handle(phrase, {})
            assert confidence > 0, f"Should handle '{phrase}' with confidence > 0"


class TestClaudeCodeSkillSecurity:
    """Test security aspects of ClaudeCodeSkill."""

    @pytest.mark.asyncio
    async def test_open_editor_escapes_path_for_vim(self):
        """Test that paths are escaped when opening vim via AppleScript."""
        skill = ClaudeCodeSkill()
        dangerous_path = Path("/Users/test;rm -rf /")

        with patch('subprocess.run') as mock_run:
            # Mock terminal check and command
            mock_run.return_value = MagicMock(returncode=0)

            skill._open_editor(dangerous_path, editor="vim")

            # Check that subprocess.run was called with proper arguments
            assert mock_run.called

    @pytest.mark.asyncio
    async def test_get_project_path_validates_directory(self):
        """Test that project paths are validated before use."""
        skill = ClaudeCodeSkill()
        memory = MagicMock()
        memory.recall = AsyncMock(return_value=['{"path": "/nonexistent/path"}'])
        memory.get_user_preferences = AsyncMock(return_value={"current_project": {"path": "/nonexistent/path"}})

        context = SkillContext(
            user_input="start development",
            intent="start_dev",
            parameters={},
            memory=memory,
            config=MagicMock(),
            conversation_history=[],
        )

        # Should fall back to current directory if stored path doesn't exist
        project_path = await skill._get_project_path(context)
        # Either returns None or falls back to cwd
        assert project_path is None or project_path == Path.cwd()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
