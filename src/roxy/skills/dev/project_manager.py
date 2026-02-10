"""Project Manager skill for Roxy.

Manages development projects including:
- Starting development on a project
- Setting up project structure
- Managing project context
- Switching between projects
- Tracking active projects
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class ProjectAction(str, Enum):
    """Actions that can be performed on projects."""

    START = "start"
    SWITCH = "switch"
    CREATE = "create"
    LIST = "list"
    STATUS = "status"
    CLOSE = "close"


@dataclass
class ProjectInfo:
    """
    Information about a development project.

    Attributes:
        name: Project name
        path: Path to project directory
        description: Optional project description
        language: Primary programming language
        framework: Framework being used
        last_accessed: Timestamp of last access
        metadata: Additional project metadata
    """

    name: str
    path: str
    description: str = ""
    language: str = ""
    framework: str = ""
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "language": self.language,
            "framework": self.framework,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectInfo:
        """Create from dictionary."""
        return cls(**data)


class ProjectManagerSkill(RoxySkill):
    """
    Project management skill for Roxy.

    Helps users manage their development projects:
    - Start development on an existing project
    - Create new project structures
    - Switch between active projects
    - List all known projects
    - Track project context in memory
    """

    name: str = "project_manager"
    description: str = "Manage development projects and contexts"
    triggers: list[str] = [
        "start development on",
        "start working on",
        "switch to project",
        "create project",
        "new project",
        "list projects",
        "show projects",
        "what projects",
        "current project",
        "project status",
        "close project",
    ]
    permissions: list[Permission] = [Permission.FILESYSTEM, Permission.SHELL]
    requires_cloud: bool = False

    # Default project directories to search
    DEFAULT_PROJECT_DIRS: list[str] = ["~/dev", "~/projects", "~/code", "~"]

    # File to store project registry
    PROJECTS_FILE: str = "projects.json"

    def __init__(self) -> None:
        """Initialize Project Manager skill."""
        super().__init__()
        self._projects_file = self._get_projects_file_path()
        self._ensure_projects_file_exists()

    def _get_projects_file_path(self) -> Path:
        """Get the path to the projects registry file."""
        # Check ROXY_DATA_DIR environment variable first
        import os

        data_dir = os.environ.get("ROXY_DATA_DIR", "~/roxy/data")
        base_path = Path(data_dir).expanduser()
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path / self.PROJECTS_FILE

    def _ensure_projects_file_exists(self) -> None:
        """Ensure the projects registry file exists."""
        if not self._projects_file.exists():
            self._projects_file.write_text("{}")

    def _load_projects(self) -> dict[str, dict[str, Any]]:
        """Load projects from the registry file.

        Returns:
            Dictionary of project data keyed by project name.
        """
        try:
            content = self._projects_file.read_text()
            if not content.strip():
                return {}
            data = json.loads(content)
            # Ensure valid structure
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse projects file: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading projects: {e}")
            return {}

    def _save_projects(self, projects: dict[str, dict[str, Any]]) -> bool:
        """Save projects to the registry file.

        Args:
            projects: Dictionary of project data.

        Returns:
            True if successful.
        """
        try:
            self._projects_file.write_text(json.dumps(projects, indent=2))
            return True
        except Exception as e:
            logger.error(f"Failed to save projects: {e}")
            return False

    async def _get_current_project(self, context: SkillContext) -> ProjectInfo | None:
        """Get the current active project from memory.

        Args:
            context: Skill execution context.

        Returns:
            Current ProjectInfo or None.
        """
        try:
            preferences = await context.memory.get_user_preferences()
            current = preferences.get("current_project")
            if current:
                return ProjectInfo.from_dict(current)
        except Exception as e:
            logger.debug(f"Could not get current project from memory: {e}")
        return None

    async def _set_current_project(self, context: SkillContext, project: ProjectInfo) -> None:
        """Set the current active project in memory.

        Args:
            context: Skill execution context.
            project: Project to set as current.
        """
        try:
            # Update last accessed timestamp
            project.last_accessed = datetime.now().isoformat()

            # Save to registry
            projects = self._load_projects()
            projects[project.name] = project.to_dict()
            self._save_projects(projects)

            # Store in memory as current project
            # Note: We're using remember() to persist this fact
            await context.memory.remember(
                "current_project",
                json.dumps(project.to_dict()),
            )
        except Exception as e:
            logger.error(f"Failed to set current project: {e}")

    def _find_project_directory(self, name: str) -> Path | None:
        """Find a project directory by name.

        Args:
            name: Project name to search for.

        Returns:
            Path to project directory or None if not found.
        """
        # First check if it's an absolute or relative path
        path = Path(name).expanduser()
        if path.exists() and path.is_dir():
            return path

        # Search in default project directories
        for base_dir in self.DEFAULT_PROJECT_DIRS:
            base = Path(base_dir).expanduser()
            if not base.exists():
                continue

            # Check for exact name match
            project_path = base / name
            if project_path.exists() and project_path.is_dir():
                return project_path

            # Check for name as subdirectory
            for subdir in base.iterdir():
                if subdir.is_dir() and name.lower() in subdir.name.lower():
                    return subdir

        # Check if name is in registry
        projects = self._load_projects()
        if name in projects:
            registered_path = Path(projects[name]["path"]).expanduser()
            if registered_path.exists():
                return registered_path

        return None

    def _detect_project_metadata(self, path: Path) -> tuple[str, str]:
        """Detect programming language and framework from project directory.

        Args:
            path: Path to project directory.

        Returns:
            Tuple of (language, framework).
        """
        language = ""
        framework = ""

        # Check for common files and directories
        indicators: dict[str, tuple[str, str]] = {
            # Language indicators
            "package.json": ("JavaScript", "Node.js"),
            "tsconfig.json": ("TypeScript", ""),
            "pyproject.toml": ("Python", ""),
            "requirements.txt": ("Python", ""),
            "setup.py": ("Python", ""),
            "Cargo.toml": ("Rust", ""),
            "go.mod": ("Go", ""),
            "pom.xml": ("Java", "Maven"),
            "build.gradle": ("Java", "Gradle"),
            "Gemfile": ("Ruby", ""),
            "composer.json": ("PHP", ""),
            # Framework indicators
            "next.config.js": ("", "Next.js"),
            "nuxt.config.js": ("", "Nuxt"),
            "vue.config.js": ("", "Vue"),
            "angular.json": ("", "Angular"),
            "gatsby-config.js": ("", "Gatsby"),
            "Dockerfile": ("", "Docker"),
            "docker-compose.yml": ("", "Docker Compose"),
        }

        for file_name, (lang, fw) in indicators.items():
            file_path = path / file_name
            if file_path.exists():
                if lang:
                    language = lang
                if fw:
                    framework = fw

        return language, framework

    async def start_project(
        self,
        context: SkillContext,
        project_name: str,
        create_if_missing: bool = False,
    ) -> SkillResult:
        """Start development on a project.

        Args:
            context: Skill execution context.
            project_name: Name or path of the project.
            create_if_missing: Whether to create project if it doesn't exist.

        Returns:
            SkillResult with status.
        """
        project_path = self._find_project_directory(project_name)

        if not project_path:
            if create_if_missing:
                return await self.create_project(context, project_name)
            else:
                # Check all known projects for similar names
                projects = self._load_projects()
                if projects:
                    similar = [
                        name
                        for name in projects.keys()
                        if project_name.lower() in name.lower()
                        or name.lower() in project_name.lower()
                    ]
                    if similar:
                        return SkillResult(
                            success=False,
                            response_text=f"Project '{project_name}' not found. Did you mean: {', '.join(similar)}?",
                            data={"suggestions": similar},
                        )

                return SkillResult(
                    success=False,
                    response_text=f"Project '{project_name}' not found. Would you like me to create it?",
                    follow_up="Create new project?",
                )

        # Detect project metadata
        language, framework = self._detect_project_metadata(project_path)

        # Create or update project info
        projects = self._load_projects()
        project_key = project_path.name

        if project_key in projects:
            project_info = ProjectInfo.from_dict(projects[project_key])
            project_info.last_accessed = datetime.now().isoformat()
        else:
            project_info = ProjectInfo(
                name=project_key,
                path=str(project_path),
                language=language,
                framework=framework,
            )

        # Set as current project
        await self._set_current_project(context, project_info)

        response_parts = [f"Switched to project: {project_info.name}"]
        if language:
            response_parts.append(f"Language: {language}")
        if framework:
            response_parts.append(f"Framework: {framework}")

        return SkillResult(
            success=True,
            response_text=f"{' | '.join(response_parts)}\nPath: {project_path}",
            data={
                "project": project_info.to_dict(),
                "action": ProjectAction.START.value,
            },
        )

    async def create_project(
        self,
        context: SkillContext,
        project_name: str,
        base_path: str | None = None,
    ) -> SkillResult:
        """Create a new project structure.

        Args:
            context: Skill execution context.
            project_name: Name for the new project.
            base_path: Base directory for project (default: ~/dev).

        Returns:
            SkillResult with status.
        """
        if not base_path:
            base_path = self.DEFAULT_PROJECT_DIRS[0]

        base = Path(base_path).expanduser()
        base.mkdir(parents=True, exist_ok=True)

        project_path = base / project_name

        if project_path.exists():
            return SkillResult(
                success=False,
                response_text=f"Project directory already exists: {project_path}",
            )

        try:
            project_path.mkdir(parents=True, exist_ok=True)

            # Create basic project structure
            (project_path / "README.md").write_text(
                f"# {project_name}\n\nProject created by Roxy.\n"
            )
            (project_path / "src").mkdir(exist_ok=True)
            (project_path / ".gitignore").write_text(
                "# Python\n__pycache__/\n*.py[cod]\n*$py.class\n.venv/\n\n# Node\nnode_modules/\n\n# IDE\n.vscode/\n.idea/\n"
            )

            # Initialize git if git is available
            try:
                import subprocess

                subprocess.run(
                    ["git", "init"],
                    cwd=project_path,
                    capture_output=True,
                    check=True,
                )
                git_initialized = True
            except Exception:
                git_initialized = False

            # Create project info
            project_info = ProjectInfo(
                name=project_name,
                path=str(project_path),
                description=f"New project: {project_name}",
                metadata={"created_by": "roxy", "git_initialized": git_initialized},
            )

            await self._set_current_project(context, project_info)

            git_msg = "Git initialized" if git_initialized else "Git not available"

            return SkillResult(
                success=True,
                response_text=f"Created new project: {project_name}\nPath: {project_path}\n{git_msg}",
                data={"project": project_info.to_dict(), "action": ProjectAction.CREATE.value},
            )

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return SkillResult(
                success=False,
                response_text=f"Failed to create project: {e}",
            )

    async def list_projects(self, context: SkillContext) -> SkillResult:
        """List all known projects.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with project list.
        """
        projects = self._load_projects()
        current = await self._get_current_project(context)

        if not projects:
            return SkillResult(
                success=True,
                response_text="No projects registered yet. Start a project with 'start development on <name>'",
                data={"projects": [], "current": None},
            )

        project_list = []
        for name, data in projects.items():
            is_current = current and current.name == name
            marker = " (current)" if is_current else ""
            lang_fw = ""
            if data.get("language"):
                lang_fw = data["language"]
                if data.get("framework"):
                    lang_fw += f" | {data['framework']}"
            project_list.append(f"- {name}{marker}: {lang_fw or 'No language detected'}")

        response = ["Known projects:"] + project_list
        if current:
            response.append(f"\nCurrently working on: {current.name}")

        return SkillResult(
            success=True,
            response_text="\n".join(response),
            data={
                "projects": list(projects.values()),
                "current": current.to_dict() if current else None,
                "action": ProjectAction.LIST.value,
            },
        )

    async def show_status(self, context: SkillContext) -> SkillResult:
        """Show current project status.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with current project info.
        """
        current = await self._get_current_project(context)

        if not current:
            return SkillResult(
                success=True,
                response_text="No active project. Start one with 'start development on <name>'",
                data={"current": None},
            )

        status_parts = [
            f"Current project: {current.name}",
            f"Path: {current.path}",
        ]

        if current.description:
            status_parts.append(f"Description: {current.description}")
        if current.language:
            status_parts.append(f"Language: {current.language}")
        if current.framework:
            status_parts.append(f"Framework: {current.framework}")

        status_parts.append(f"Last accessed: {current.last_accessed}")

        return SkillResult(
            success=True,
            response_text="\n".join(status_parts),
            data={"current": current.to_dict(), "action": ProjectAction.STATUS.value},
        )

    async def close_project(self, context: SkillContext) -> SkillResult:
        """Close the current project.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with status.
        """
        current = await self._get_current_project(context)

        if not current:
            return SkillResult(
                success=True,
                response_text="No project is currently active.",
                data={"action": ProjectAction.CLOSE.value},
            )

        project_name = current.name
        # Clear from memory
        await context.memory.remember("current_project", "")

        return SkillResult(
            success=True,
            response_text=f"Closed project: {project_name}",
            data={"closed_project": project_name, "action": ProjectAction.CLOSE.value},
        )

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute the Project Manager skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with execution status.
        """
        user_input_lower = context.user_input.lower()

        # Determine action from input
        if any(word in user_input_lower for word in ["list", "show", "what projects"]):
            return await self.list_projects(context)

        if any(word in user_input_lower for word in ["status", "current project"]):
            return await self.show_status(context)

        if "close" in user_input_lower:
            return await self.close_project(context)

        # Check for create/new action
        create_keywords = ["create", "new", "make a", "make an", "start a new"]
        is_create = any(keyword in user_input_lower for keyword in create_keywords)

        # Extract project name from input
        # Try to get from parameters first
        project_name = context.parameters.get("project", context.parameters.get("name", ""))

        if not project_name:
            # Parse from user input - look for patterns like
            # "start development on X", "switch to X", "create project X"
            import re

            # Pattern: after common prepositions
            patterns = [
                r"(?:on|to|called|named)\s+(['\"]?)([\w\-\.]+)\1",
                r"(?:create|new|make)\s+(?:project\s+)?(['\"]?)([\w\-\.]+)\1",
            ]

            for pattern in patterns:
                match = re.search(pattern, user_input_lower)
                if match:
                    project_name = match.group(2)
                    break

        if not project_name:
            if is_create:
                return SkillResult(
                    success=False,
                    response_text="What would you like to name the new project?",
                    follow_up="Please provide a project name.",
                )
            else:
                return await self.list_projects(context)

        if is_create:
            return await self.create_project(context, project_name)

        # Default to starting/switching to project
        switch_keywords = ["switch", "change to"]
        is_switch = any(keyword in user_input_lower for keyword in switch_keywords)

        return await self.start_project(
            context,
            project_name,
            create_if_missing=False,
        )
