"""Shortcuts skill for running macOS Shortcuts.

Provides ability to list and run macOS Shortcuts from the command line.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class ShortcutsSkill(RoxySkill):
    """Skill for running macOS Shortcuts."""

    name = "shortcuts"
    description = "Run and list macOS Shortcuts"
    triggers = [
        "run shortcut",
        "execute shortcut",
        "list shortcuts",
        "show shortcuts",
    ]
    permissions = [Permission.SHELL]
    requires_cloud = False

    async def execute(self, context: SkillContext) -> SkillResult:
        """
        Execute the shortcuts skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with shortcut operation result.
        """
        user_input = context.user_input.lower()
        parameters = context.parameters

        # Determine the operation
        if "list" in user_input or "show" in user_input:
            return await self._list_shortcuts()
        elif "run" in user_input or "execute" in user_input:
            shortcut_name = self._extract_shortcut_name(user_input, parameters)
            if shortcut_name:
                return await self._run_shortcut(shortcut_name, parameters)
            else:
                return SkillResult(
                    success=False,
                    response_text="Which shortcut would you like me to run?",
                    speak=False,
                )
        else:
            # Default: list shortcuts
            return await self._list_shortcuts()

    def _extract_shortcut_name(self, user_input: str, parameters: dict) -> str | None:
        """
        Extract shortcut name from user input.

        Args:
            user_input: Lowercase user input.
            parameters: Extracted parameters.

        Returns:
            Shortcut name or None if not found.
        """
        # Check if LLM extracted a shortcut_name parameter
        if "shortcut_name" in parameters:
            return parameters["shortcut_name"]

        # Try to extract shortcut name from common patterns
        import re

        patterns = [
            r"run\s+(?:shortcut\s+)?(?:called\s+)?['\"]?(.+?)['\"]?(?:\s|$|\?)",
            r"execute\s+(?:shortcut\s+)?(?:called\s+)?['\"]?(.+?)['\"]?(?:\s|$|\?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, user_input)
            if match:
                name = match.group(1).strip()
                # Remove trailing words
                name = re.split(r"\s+(?:please|now)$", name)[0].strip()
                return name if name else None

        return None

    async def _list_shortcuts(self) -> SkillResult:
        """
        List all available shortcuts.

        Returns:
            SkillResult with list of shortcuts.
        """
        try:
            # Run shortcuts list command
            process = await asyncio.create_subprocess_exec(
                "shortcuts",
                "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode().strip()
                logger.error(f"shortcuts list error: {error_msg}")
                return SkillResult(
                    success=False,
                    response_text="Sorry, I couldn't list your shortcuts. Make sure Shortcuts is enabled.",
                    speak=True,
                )

            # Parse output
            output = stdout.decode().strip()
            lines = output.split("\n")

            # Filter out empty lines and headers
            shortcuts = [line.strip() for line in lines if line.strip()]

            if not shortcuts:
                return SkillResult(
                    success=True,
                    response_text="You don't have any shortcuts yet. You can create them in the Shortcuts app.",
                    speak=True,
                )

            # Format response
            response = f"You have {len(shortcuts)} shortcut{'s' if len(shortcuts) != 1 else ''}:"

            # List shortcuts (limit to first 10 for speech)
            for shortcut in shortcuts[:10]:
                response += f"\n• {shortcut}"

            if len(shortcuts) > 10:
                response += f"\n... and {len(shortcuts) - 10} more"

            speech_response = f"You have {len(shortcuts)} shortcuts"

            return SkillResult(
                success=True,
                response_text=response,
                speak=True,
                data={"shortcuts": shortcuts, "count": len(shortcuts)},
            )

        except Exception as e:
            logger.error(f"Error listing shortcuts: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't list your shortcuts: {e}",
                speak=True,
            )

    async def _run_shortcut(
        self, shortcut_name: str, parameters: dict
    ) -> SkillResult:
        """
        Run a shortcut.

        Args:
            shortcut_name: Name of the shortcut to run.
            parameters: Parameters to pass to the shortcut.

        Returns:
            SkillResult with run operation result.
        """
        try:
            # Build command
            cmd = ["shortcuts", "run", shortcut_name]

            # Add input parameter if provided
            if "input" in parameters:
                cmd.extend(["--input", str(parameters["input"])])

            # Run shortcut
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode().strip()

                # Check if shortcut not found
                if "not found" in error_msg.lower() or "could not find" in error_msg.lower():
                    # Try to list similar shortcuts
                    similar = await self._find_similar_shortcuts(shortcut_name)
                    if similar:
                        suggestion = f"Shortcut '{shortcut_name}' not found. Did you mean: {', '.join(similar[:3])}?"
                    else:
                        suggestion = f"Shortcut '{shortcut_name}' not found."
                    return SkillResult(
                        success=False,
                        response_text=suggestion,
                        speak=True,
                    )

                logger.error(f"shortcuts run error: {error_msg}")
                return SkillResult(
                    success=False,
                    response_text=f"Sorry, I couldn't run the shortcut: {error_msg}",
                    speak=True,
                )

            # Get output if any
            output = stdout.decode().strip()

            response = f"Running shortcut '{shortcut_name}'"
            if output:
                response += f"\n{output}"

            return SkillResult(
                success=True,
                response_text=response,
                speak=True,
                data={"shortcut": shortcut_name, "output": output},
            )

        except Exception as e:
            logger.error(f"Error running shortcut: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't run the shortcut: {e}",
                speak=True,
            )

    async def _find_similar_shortcuts(self, shortcut_name: str) -> list[str]:
        """
        Find shortcuts with similar names.

        Args:
            shortcut_name: Shortcut name to match.

        Returns:
            List of similar shortcut names.
        """
        try:
            # Get all shortcuts
            process = await asyncio.create_subprocess_exec(
                "shortcuts",
                "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return []

            output = stdout.decode().strip()
            shortcuts = [line.strip() for line in output.split("\n") if line.strip()]

            # Find similar shortcuts (case-insensitive substring match)
            shortcut_lower = shortcut_name.lower()
            similar = [
                s for s in shortcuts
                if shortcut_lower in s.lower() or s.lower() in shortcut_lower
            ]

            return similar

        except Exception as e:
            logger.debug(f"Error finding similar shortcuts: {e}")
            return []

    def can_handle(self, intent: str, parameters: dict) -> float:
        """
        Calculate confidence that this skill can handle the intent.

        Args:
            intent: Classified intent string.
            parameters: Extracted parameters.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Check for direct trigger matches
        if any(trigger in intent.lower() for trigger in self.triggers):
            return 0.9

        # Check for shortcut-related keywords
        shortcut_keywords = ["shortcut", "automation"]
        if any(keyword in intent.lower() for keyword in shortcut_keywords):
            return 0.6

        return 0.0
