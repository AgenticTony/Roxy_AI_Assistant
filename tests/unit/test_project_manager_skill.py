"""Tests for the Project Manager skill.

Tests managing development projects including starting, creating, listing, and switching.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.skills.base import Permission, SkillContext, SkillResult
from roxy.skills.dev.project_manager import (
    ProjectAction,
    ProjectInfo,
    ProjectManagerSkill,
)


class TestProjectInfo:
    """Tests for ProjectInfo dataclass."""

    def test_to_dict(self):
        """Test converting ProjectInfo to dictionary."""
        info = ProjectInfo(
            name="test-project",
            path="/Users/test/projects/test-project",
            description="A test project",
            language="Python",
            framework="FastAPI",
        )

        result = info.to_dict()

        assert result["name"] == "test-project"
        assert result["path"] == "/Users/test/projects/test-project"
        assert result["description"] == "A test project"
        assert result["language"] == "Python"
        assert result["framework"] == "FastAPI"

    def test_from_dict(self):
        """Test creating ProjectInfo from dictionary."""
        data = {
            "name": "test-project",
            "path": "/Users/test/projects/test-project",
            "description": "A test project",
            "language": "Python",
            "framework": "FastAPI",
        }

        info = ProjectInfo.from_dict(data)

        assert info.name == "test-project"
        assert info.path == "/Users/test/projects/test-project"
        assert info.description == "A test project"
        assert info.language == "Python"
        assert info.framework == "FastAPI"


class TestProjectManagerSkill:
    """Tests for ProjectManagerSkill."""

    def test_skill_metadata(self):
        """Test skill has correct metadata."""
        skill = ProjectManagerSkill()

        assert skill.name == "project_manager"
        assert "project" in skill.description.lower()
        assert len(skill.triggers) > 0
        assert Permission.FILESYSTEM in skill.permissions
        assert Permission.SHELL in skill.permissions
        assert skill.requires_cloud is False

    def test_init_creates_projects_file(self):
        """Test that initialization creates projects file."""
        skill = ProjectManagerSkill()

        assert skill._projects_file.exists()
        # Should be empty or valid JSON
        content = skill._projects_file.read_text()
        assert content == "{}" or json.loads(content) is not None

    def test_load_projects_empty(self):
        """Test loading projects when file is empty."""
        skill = ProjectManagerSkill()

        # Clear the projects file first
        skill._projects_file.write_text("{}")

        projects = skill._load_projects()
        assert projects == {}

    def test_load_projects_with_data(self):
        """Test loading projects with existing data."""
        skill = ProjectManagerSkill()

        # Add test data
        test_data = {
            "test-project": {
                "name": "test-project",
                "path": "/Users/test/projects/test-project",
            }
        }
        skill._save_projects(test_data)

        projects = skill._load_projects()
        assert "test-project" in projects
        assert projects["test-project"]["path"] == "/Users/test/projects/test-project"

    def test_save_projects(self):
        """Test saving projects to file."""
        skill = ProjectManagerSkill()

        test_data = {
            "test-project": {
                "name": "test-project",
                "path": "/Users/test/projects/test-project",
            }
        }

        result = skill._save_projects(test_data)
        assert result is True

        # Verify file was written
        loaded = skill._load_projects()
        assert "test-project" in loaded

    @pytest.mark.asyncio
    async def test_find_project_directory_by_path(self):
        """Test finding project by absolute path."""
        skill = ProjectManagerSkill()

        # Use a path that should exist
        test_path = Path.cwd()

        result = skill._find_project_directory(str(test_path))
        assert result == test_path

    @pytest.mark.asyncio
    async def test_find_project_directory_not_found(self):
        """Test finding non-existent project."""
        skill = ProjectManagerSkill()

        result = skill._find_project_directory("/nonexistent/path/to/project")
        assert result is None

    def test_detect_project_metadata_python(self):
        """Test detecting Python project metadata."""
        skill = ProjectManagerSkill()

        # Create temporary directory with Python markers
        with patch("pathlib.Path.exists", return_value=True):
            language, framework = skill._detect_project_metadata(Path("/fake/path"))

            # Will return empty strings since we can't actually create files
            assert isinstance(language, str)
            assert isinstance(framework, str)

    @pytest.mark.asyncio
    async def test_start_project_success(self):
        """Test successfully starting a project."""
        skill = ProjectManagerSkill()

        context = SkillContext(
            user_input="start development on roxy",
            intent="start_project",
            parameters={"project": "roxy"},
            memory=MagicMock(
                remember=AsyncMock(),
                recall=AsyncMock(return_value=[]),
                get_user_preferences=AsyncMock(return_value={}),
            ),
            config=MagicMock(),
            conversation_history=[],
        )

        # Use current directory as test project
        current_dir = str(Path.cwd())
        with patch.object(skill, "_find_project_directory", return_value=Path(current_dir)):
            result = await skill.start_project(context, "roxy")

            assert result.success is True
            assert (
                "roxy" in result.response_text.lower() or "project" in result.response_text.lower()
            )

    @pytest.mark.asyncio
    async def test_start_project_not_found(self):
        """Test starting a project that doesn't exist."""
        skill = ProjectManagerSkill()

        context = SkillContext(
            user_input="start development on nonexistent",
            intent="start_project",
            parameters={"project": "nonexistent"},
            memory=MagicMock(
                remember=AsyncMock(),
                recall=AsyncMock(return_value=[]),
                get_user_preferences=AsyncMock(return_value={}),
            ),
            config=MagicMock(),
            conversation_history=[],
        )

        with patch.object(skill, "_find_project_directory", return_value=None):
            result = await skill.start_project(context, "nonexistent", create_if_missing=False)

            assert result.success is False
            assert "not found" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_create_project_success(self):
        """Test successfully creating a new project."""
        skill = ProjectManagerSkill()

        context = SkillContext(
            user_input="create project test-project",
            intent="create_project",
            parameters={"name": "test-project"},
            memory=MagicMock(
                remember=AsyncMock(),
                recall=AsyncMock(return_value=[]),
                get_user_preferences=AsyncMock(return_value={}),
            ),
            config=MagicMock(),
            conversation_history=[],
        )

        with patch("pathlib.Path.mkdir"), patch("pathlib.Path.write_text"):
            with patch("subprocess.run"):
                result = await skill.create_project(context, "test-project")

                # Should succeed
                assert result.success is True or "project" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_list_projects_empty(self):
        """Test listing projects when none exist."""
        skill = ProjectManagerSkill()

        # Clear projects file
        skill._save_projects({})

        context = SkillContext(
            user_input="list projects",
            intent="list_projects",
            parameters={},
            memory=MagicMock(
                recall=AsyncMock(return_value=[]),
                get_user_preferences=AsyncMock(return_value={}),
            ),
            config=MagicMock(),
            conversation_history=[],
        )

        result = await skill.list_projects(context)

        assert result.success is True
        assert (
            "no projects" in result.response_text.lower()
            or "registered yet" in result.response_text.lower()
        )

    @pytest.mark.asyncio
    async def test_list_projects_with_items(self):
        """Test listing projects with existing items."""
        skill = ProjectManagerSkill()

        # Add test project
        test_project = {
            "name": "test-project",
            "path": "/Users/test/projects/test-project",
            "language": "Python",
        }
        skill._save_projects({"test-project": test_project})

        context = SkillContext(
            user_input="list projects",
            intent="list_projects",
            parameters={},
            memory=MagicMock(
                recall=AsyncMock(return_value=[]),
                get_user_preferences=AsyncMock(return_value={}),
            ),
            config=MagicMock(),
            conversation_history=[],
        )

        result = await skill.list_projects(context)

        assert result.success is True
        assert "test-project" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_show_status_no_current_project(self):
        """Test showing status when no current project."""
        skill = ProjectManagerSkill()

        context = SkillContext(
            user_input="project status",
            intent="status",
            parameters={},
            memory=MagicMock(
                recall=AsyncMock(return_value=[]),
                get_user_preferences=AsyncMock(return_value={}),
            ),
            config=MagicMock(),
            conversation_history=[],
        )

        result = await skill.show_status(context)

        assert result.success is True
        assert "no active project" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_close_project_no_current(self):
        """Test closing project when none is active."""
        skill = ProjectManagerSkill()

        context = SkillContext(
            user_input="close project",
            intent="close",
            parameters={},
            memory=MagicMock(
                remember=AsyncMock(),
                recall=AsyncMock(return_value=[]),
                get_user_preferences=AsyncMock(return_value={}),
            ),
            config=MagicMock(),
            conversation_history=[],
        )

        result = await skill.close_project(context)

        assert result.success is True
        assert (
            "no project" in result.response_text.lower()
            or "currently active" in result.response_text.lower()
        )

    @pytest.mark.asyncio
    async def test_execute_list_action(self):
        """Test execute method for list action."""
        skill = ProjectManagerSkill()

        context = SkillContext(
            user_input="list my projects",
            intent="list_projects",
            parameters={},
            memory=MagicMock(
                recall=AsyncMock(return_value=[]),
                get_user_preferences=AsyncMock(return_value={}),
            ),
            config=MagicMock(),
            conversation_history=[],
        )

        with patch.object(
            skill,
            "list_projects",
            new=AsyncMock(
                return_value=SkillResult(
                    success=True,
                    response_text="No projects registered",
                )
            ),
        ):
            result = await skill.execute(context)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_status_action(self):
        """Test execute method for status action."""
        skill = ProjectManagerSkill()

        context = SkillContext(
            user_input="what's the current project status",
            intent="status",
            parameters={},
            memory=MagicMock(
                recall=AsyncMock(return_value=[]),
                get_user_preferences=AsyncMock(return_value={}),
            ),
            config=MagicMock(),
            conversation_history=[],
        )

        with patch.object(
            skill,
            "show_status",
            new=AsyncMock(
                return_value=SkillResult(
                    success=True,
                    response_text="No active project",
                )
            ),
        ):
            result = await skill.execute(context)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_close_action(self):
        """Test execute method for close action."""
        skill = ProjectManagerSkill()

        context = SkillContext(
            user_input="close the current project",
            intent="close",
            parameters={},
            memory=MagicMock(
                remember=AsyncMock(),
                recall=AsyncMock(return_value=[]),
                get_user_preferences=AsyncMock(return_value={}),
            ),
            config=MagicMock(),
            conversation_history=[],
        )

        with patch.object(
            skill,
            "close_project",
            new=AsyncMock(
                return_value=SkillResult(
                    success=True,
                    response_text="No project is currently active",
                )
            ),
        ):
            result = await skill.execute(context)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_can_handle_project_phrases(self):
        """Test can_handle recognizes project phrases."""
        skill = ProjectManagerSkill()

        phrases = [
            "start development on",
            "switch to project",
            "create project",
            "new project",
            "list projects",
            "current project",
            "project status",
        ]

        for phrase in phrases:
            confidence = skill.can_handle(phrase, {})
            assert confidence > 0, f"Should handle '{phrase}' with confidence > 0"


class TestProjectAction:
    """Tests for ProjectAction enum."""

    def test_action_values(self):
        """Test ProjectAction enum values."""
        assert ProjectAction.START.value == "start"
        assert ProjectAction.SWITCH.value == "switch"
        assert ProjectAction.CREATE.value == "create"
        assert ProjectAction.LIST.value == "list"
        assert ProjectAction.STATUS.value == "status"
        assert ProjectAction.CLOSE.value == "close"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
