"""Claude Code skill for Roxy.

Launches development environment with Cursor/VSCode, Terminal, and Claude Code.
All operations local.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class ClaudeCodeSkill(RoxySkill):
    """
    Claude Code launcher skill for development mode.

    Features:
    - Launch Cursor/VSCode with project
    - Open terminal in project directory
    - Provide shortcuts for common dev tasks
    - All operations local
    """

    name: str = "claude_code"
    description: str = "Launch development environment with Claude Code"
    triggers: list[str] = [
        "start development",
        "code mode",
        "open project",
        "start coding",
        "dev mode",
        "launch ide",
    ]
    permissions: list[Permission] = [Permission.SHELL, Permission.FILESYSTEM]
    requires_cloud: bool = False

    # Default editors to try (in order of preference)
    EDITORS: list[str] = ["cursor", "code", "vim", "nano"]

    # Terminal app for macOS
    TERMINAL_APP: str = "Terminal.app"

    def __init__(self) -> None:
        """Initialize Claude Code skill."""
        super().__init__()

        # Detect available editors
        self._available_editor = self._find_editor()

    def _find_editor(self) -> str | None:
        """Find the first available code editor.

        Returns:
            Editor command or None if none found.
        """
        for editor in self.EDITORS:
            try:
                result = subprocess.run(
                    ["which", editor],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    logger.info(f"Found editor: {editor}")
                    return editor
            except Exception:
                continue

        # Also check for Cursor.app specifically (macOS app)
        try:
            result = subprocess.run(
                ["ls", "/Applications/Cursor.app"],
                capture_output=True,
            )
            if result.returncode == 0:
                return "cursor"
        except Exception:
            pass

        return None

    def _open_terminal(self, path: Path) -> bool:
        """Open macOS Terminal at specified path.

        Args:
            path: Directory to open in terminal.

        Returns:
            True if successful.
        """
        try:
            subprocess.run(
                ["open", "-a", self.TERMINAL_APP, str(path)],
                check=True,
            )
            logger.info(f"Opened terminal at: {path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to open terminal: {e}")
            return False
        except Exception as e:
            logger.error(f"Error opening terminal: {e}")
            return False

    def _open_editor(self, path: Path, editor: str | None = None) -> bool:
        """Open code editor at specified path.

        Args:
            path: Directory or file to open.
            editor: Editor command (default: auto-detect).

        Returns:
            True if successful.
        """
        editor = editor or self._available_editor

        if not editor:
            logger.warning("No code editor found")
            return False

        try:
            if editor in ["cursor", "code"]:
                # VSCode/Cursor CLI
                subprocess.run([editor, str(path)], check=True)
            elif editor == "vim":
                # VIM requires terminal, so open in terminal
                # Import the escape function to prevent AppleScript injection
                from roxy.macos.applescript import escape_applescript_string

                terminal_safe = escape_applescript_string(self.TERMINAL_APP)
                path_safe = escape_applescript_string(str(path))
                subprocess.run([
                    "osascript",
                    "-e",
                    f'tell application "{terminal_safe}"\n'
                    f'  do script "cd \\"{path_safe}\\" && vim"\n'
                    f'end tell'
                ])
            else:
                # Try opening as a file
                subprocess.run([editor, str(path)], check=True)

            logger.info(f"Opened {editor} at: {path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to open editor: {e}")
            return False
        except Exception as e:
            logger.error(f"Error opening editor: {e}")
            return False

    async def _get_project_path(self, context: SkillContext) -> Path | None:
        """Get project path from context, memory, or current directory.

        Args:
            context: Skill execution context.

        Returns:
            Project path or None.
        """
        # Check parameters first
        path_param = context.parameters.get("path", context.parameters.get("project"))
        if path_param:
            return Path(path_param).expanduser()

        # Check memory for current project
        try:
            project_memories = await context.memory.recall("current_project")
            if project_memories:
                # Parse the stored project info
                project_info = json.loads(project_memories[0])
                project_path = project_info.get("path")
                if project_path:
                    path = Path(project_path).expanduser()
                    if path.exists() and path.is_dir():
                        logger.info(f"Using current project from memory: {path}")
                        return path
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.debug(f"Could not retrieve current project from memory: {e}")

        # Default to current directory
        return Path.cwd()

    async def launch_dev_mode(
        self,
        project_path: Path,
        open_terminal: bool = True,
        open_editor: bool = True,
    ) -> str:
        """Launch development environment.

        Args:
            project_path: Path to project directory.
            open_terminal: Whether to open terminal.
            open_editor: Whether to open code editor.

        Returns:
            Status message.
        """
        if not project_path.exists():
            return f"Project path doesn't exist: {project_path}"

        results = []

        if open_editor:
            if self._available_editor:
                if self._open_editor(project_path):
                    results.append(f"Opened {self._available_editor}")
            else:
                results.append("No code editor found (install Cursor or VSCode)")

        if open_terminal:
            if self._open_terminal(project_path):
                results.append("Opened Terminal")

        if results:
            return f"Development environment ready: {', '.join(results)}\nProject: {project_path}"
        else:
            return f"Project path: {project_path}"

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute Claude Code launcher skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with launch status.
        """
        user_input = context.user_input.lower()

        # Get project path
        project_path = await self._get_project_path(context)

        if not project_path:
            return SkillResult(
                success=False,
                response_text="What project would you like to work on?",
                follow_up="Please provide the project path.",
            )

        # Check for specific flags in input
        open_terminal = "terminal" not in user_input or "no terminal" not in user_input
        open_editor = "editor" not in user_input or "no editor" not in user_input

        # Check if requesting specific editor
        editor_override = None
        for editor in ["cursor", "vscode", "vim", "code"]:
            if editor in user_input:
                editor_override = "cursor" if editor == "vscode" else editor
                break

        if editor_override:
            if not self._open_editor(project_path, editor_override):
                return SkillResult(
                    success=False,
                    response_text=f"Couldn't open {editor_override}",
                )
            return SkillResult(
                success=True,
                response_text=f"Opened {editor_override} for project: {project_path}",
                data={"project": str(project_path), "editor": editor_override},
            )

        # Launch full dev mode
        result = await self.launch_dev_mode(
            project_path=project_path,
            open_terminal=open_terminal,
            open_editor=open_editor,
        )

        return SkillResult(
            success=True,
            response_text=result,
            data={
                "project": str(project_path),
                "editor": self._available_editor,
                "terminal": open_terminal,
            },
        )
