"""Git operations skill for Roxy.

Provides git command execution via subprocess.
All operations run locally.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class GitCommand(str, Enum):
    """Common git commands."""

    STATUS = "status"
    ADD = "add"
    COMMIT = "commit"
    PUSH = "push"
    PULL = "pull"
    LOG = "log"
    DIFF = "diff"
    BRANCH = "branch"
    CHECKOUT = "checkout"


@dataclass
class GitResult:
    """Result of a git command execution."""

    success: bool
    output: str
    error: str | None = None
    command: str = ""


class GitOpsSkill(RoxySkill):
    """
    Git operations skill using subprocess.

    Features:
    - Run common git commands
    - Generate commit messages using local LLM
    - All operations local
    """

    name: str = "git_ops"
    description: str = "Execute git commands"
    triggers: list[str] = [
        "git status",
        "commit changes",
        "push",
        "git log",
        "create branch",
        "switch branch",
    ]
    permissions: list[Permission] = [Permission.SHELL]
    requires_cloud: bool = False

    # Configuration
    DEFAULT_COMMIT_MESSAGE: str = "Update from Roxy"
    MAX_LOG_ENTRIES: int = 5

    def __init__(self) -> None:
        """Initialize git ops skill."""
        super().__init__()

    def _run_git(self, args: list[str], cwd: Path | None = None) -> GitResult:
        """Run a git command.

        Args:
            args: Git command arguments (e.g., ["status", "-sb"]).
            cwd: Working directory for the command.

        Returns:
            GitResult with command output.
        """
        cmd = ["git"] + args

        logger.debug(f"Running git command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=30,
            )

            return GitResult(
                success=result.returncode == 0,
                output=result.stdout.strip(),
                error=result.stderr.strip() if result.stderr else None,
                command=" ".join(cmd),
            )

        except subprocess.TimeoutExpired:
            logger.error("Git command timed out")
            return GitResult(
                success=False,
                output="",
                error="Git command timed out after 30 seconds",
                command=" ".join(cmd),
            )
        except FileNotFoundError:
            return GitResult(
                success=False,
                output="",
                error="Git is not installed or not in PATH",
                command=" ".join(cmd),
            )
        except Exception as e:
            logger.error(f"Git command error: {e}")
            return GitResult(
                success=False,
                output="",
                error=str(e),
                command=" ".join(cmd),
            )

    def git_status(self, path: str | None = None) -> str:
        """Get git repository status.

        Args:
            path: Path to repository (default: current directory).

        Returns:
            Formatted status output.
        """
        cwd = None
        if path:
            from roxy.macos.path_validation import validate_path

            try:
                cwd = validate_path(path, must_exist=True)
            except (ValueError, FileNotFoundError) as e:
                return f"Invalid repository path '{path}': {e}"

        result = self._run_git(["status", "-sb"], cwd=cwd)

        if not result.success:
            return f"Error getting git status: {result.error}"

        # Also get branch info
        branch_result = self._run_git(["branch", "--show-current"], cwd=cwd)

        output = result.output
        if branch_result.success:
            output = f"Current branch: {branch_result.output}\n\n{output}"

        return output

    def git_log(self, count: int = 5, path: str | None = None) -> str:
        """Get recent commit log.

        Args:
            count: Number of commits to show.
            path: Path to repository.

        Returns:
            Formatted commit log.
        """
        cwd = None
        if path:
            from roxy.macos.path_validation import validate_path

            try:
                cwd = validate_path(path, must_exist=True)
            except (ValueError, FileNotFoundError) as e:
                return f"Invalid repository path '{path}': {e}"

        result = self._run_git(
            ["log", "-n", str(count), "--pretty=format:%h - %s (%an, %ar)"],
            cwd=cwd,
        )

        if not result.success:
            return f"Error getting git log: {result.error}"

        return result.output

    def git_add(self, files: list[str], path: str | None = None) -> bool:
        """Stage files for commit.

        Args:
            files: List of file paths to stage (use ["."] for all).
            path: Path to repository.

        Returns:
            True if successful.
        """
        cwd = None
        if path:
            from roxy.macos.path_validation import validate_path

            try:
                cwd = validate_path(path, must_exist=True)
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Invalid repository path '{path}': {e}")
                return False

        result = self._run_git(["add"] + files, cwd=cwd)

        return result.success

    def git_commit(self, message: str, path: str | None = None) -> bool:
        """Create a commit.

        Args:
            message: Commit message.
            path: Path to repository.

        Returns:
            True if successful.
        """
        cwd = Path(path) if path else None
        result = self._run_git(["commit", "-m", message], cwd=cwd)

        return result.success

    def git_push(self, remote: str = "origin", branch: str | None = None, path: str | None = None) -> bool:
        """Push commits to remote.

        Args:
            remote: Remote name (default: "origin").
            branch: Branch name (default: current branch).
            path: Path to repository.

        Returns:
            True if successful.
        """
        cwd = Path(path) if path else None

        if branch:
            result = self._run_git(["push", remote, branch], cwd=cwd)
        else:
            result = self._run_git(["push"], cwd=cwd)

        return result.success

    def git_pull(self, remote: str = "origin", branch: str | None = None, path: str | None = None) -> bool:
        """Pull changes from remote.

        Args:
            remote: Remote name (default: "origin").
            branch: Branch name (default: current branch).
            path: Path to repository.

        Returns:
            True if successful.
        """
        cwd = Path(path) if path else None

        if branch:
            result = self._run_git(["pull", remote, branch], cwd=cwd)
        else:
            result = self._run_git(["pull"], cwd=cwd)

        return result.success

    async def generate_commit_message(self, context: SkillContext, path: str | None = None) -> str:
        """Generate a commit message using local LLM.

        Analyzes staged changes to generate an appropriate commit message.

        Args:
            context: SkillContext with access to local LLM client.
            path: Path to repository.

        Returns:
            Generated commit message.
        """
        # Get diff of staged changes
        cwd = Path(path) if path else None
        diff_result = self._run_git(["diff", "--staged"], cwd=cwd)

        if not diff_result.success or not diff_result.output:
            return "No staged changes to commit"

        # Try to use local LLM for commit message generation
        if context.local_llm_client is not None:
            try:
                # Prepare prompt for commit message generation
                prompt = f"""Generate a concise git commit message following conventional commits format.

Staged changes:
```
{diff_result.output[:2000]}
```

Guidelines:
- Use conventional commits format: <type>(<scope>): <description>
- Types: feat, fix, docs, style, refactor, test, chore
- Keep the description under 72 characters
- Use imperative mood ("add" not "added" or "adds")
- Return ONLY the commit message, nothing else

Example formats:
- feat: add user authentication
- fix(api): resolve null pointer exception
- docs: update README with installation steps
- refactor: simplify user model

Generate the commit message:"""

                response = await context.local_llm_client.generate(
                    prompt=prompt,
                    temperature=0.3,  # Low temperature for consistent output
                    max_tokens=100,
                )

                # Extract the commit message from response
                commit_message = response.content.strip().strip('"\'').split('\n')[0]

                # Validate that we got a reasonable commit message
                if commit_message and len(commit_message) > 5 and len(commit_message) < 150:
                    logger.info(f"Generated commit message using LLM: {commit_message}")
                    return commit_message
                else:
                    logger.warning(f"LLM returned invalid commit message, falling back to simple message")

            except Exception as e:
                logger.error(f"Error generating commit message with LLM: {e}")
                # Fall through to simple message generation

        # Fallback to simple message generation
        lines = diff_result.output.split("\n")
        changed_files = [line for line in lines if line.startswith("+++ ") or line.startswith("--- ")]

        if changed_files:
            return f"Update {len(changed_files)} file{'s' if len(changed_files) > 1 else ''}"

        return self.DEFAULT_COMMIT_MESSAGE

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute git operations skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with git operation results.
        """
        user_input = context.user_input.lower()
        path = context.parameters.get("path")

        # Status
        if "status" in user_input:
            output = self.git_status(path)
            return SkillResult(
                success=True,
                response_text=output,
                data={"command": "status", "path": path},
            )

        # Log
        if "log" in user_input or "history" in user_input:
            count = context.parameters.get("count", self.MAX_LOG_ENTRIES)
            output = self.git_log(count, path)
            return SkillResult(
                success=True,
                response_text=output,
                data={"command": "log", "path": path},
            )

        # Commit
        if "commit" in user_input:
            # Get commit message or generate one
            message = context.parameters.get("message")

            if not message:
                # Stage all changes first if message is "all"
                if "all" in user_input:
                    self.git_add(["."], path)

                # Generate message
                message = await self.generate_commit_message(context, path)

            # Commit
            success = self.git_commit(message, path)

            if success:
                return SkillResult(
                    success=True,
                    response_text=f"Committed changes: {message}",
                    data={"command": "commit", "message": message, "path": path},
                )
            else:
                return SkillResult(
                    success=False,
                    response_text="Failed to commit. Maybe there are no staged changes?",
                )

        # Push
        if "push" in user_input:
            remote = context.parameters.get("remote", "origin")
            branch = context.parameters.get("branch")

            # Ask for confirmation before pushing (security measure)
            # For now, we'll just indicate what would happen
            return SkillResult(
                success=True,
                response_text=f"Ready to push to {remote}" + (f" {branch}" if branch else ""),
                follow_up="Should I proceed with the push?",
                data={"command": "push", "remote": remote, "branch": branch, "pending": True},
            )

        # Pull
        if "pull" in user_input:
            remote = context.parameters.get("remote", "origin")
            branch = context.parameters.get("branch")

            success = self.git_pull(remote, branch, path)

            if success:
                return SkillResult(
                    success=True,
                    response_text=f"Pulled changes from {remote}",
                    data={"command": "pull", "remote": remote, "branch": branch},
                )
            else:
                return SkillResult(
                    success=False,
                    response_text="Failed to pull changes",
                )

        # Branch operations
        if "branch" in user_input:
            if "create" in user_input or "new" in user_input:
                branch_name = context.parameters.get("name")
                if not branch_name:
                    return SkillResult(
                        success=False,
                        response_text="What would you like to name the new branch?",
                        follow_up="Please provide a branch name.",
                    )

                result = self._run_git(["checkout", "-b", branch_name], Path(path) if path else None)

                if result.success:
                    return SkillResult(
                        success=True,
                        response_text=f"Created and switched to new branch: {branch_name}",
                        data={"command": "branch", "name": branch_name, "path": path},
                    )
                else:
                    return SkillResult(
                        success=False,
                        response_text=f"Failed to create branch: {result.error}",
                    )

            # List branches
            result = self._run_git(["branch", "-a"], Path(path) if path else None)

            if result.success:
                return SkillResult(
                    success=True,
                    response_text=f"Branches:\n{result.output}",
                    data={"command": "branch", "path": path},
                )

        # Default: show status
        output = self.git_status(path)
        return SkillResult(
            success=True,
            response_text=output,
            data={"command": "status", "path": path},
        )
