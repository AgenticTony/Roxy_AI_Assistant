"""Clipboard skill for managing clipboard content.

Provides read, write, and history functionality for the macOS clipboard.
"""

from __future__ import annotations

import logging
from typing import Any

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)


class ClipboardSkill(RoxySkill):
    """Skill for clipboard operations."""

    name = "clipboard"
    description = "Read, write, and manage clipboard content"
    triggers = [
        "copy that",
        "paste",
        "what's in my clipboard",
        "clear clipboard",
        "clipboard",
    ]
    permissions = [Permission.APPLESCRIPT]
    requires_cloud = False

    def __init__(self) -> None:
        """Initialize clipboard skill."""
        super().__init__()
        self._history: list[dict[str, Any]] = []
        self._max_history = 10

    async def execute(self, context: SkillContext) -> SkillResult:
        """
        Execute the clipboard skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with clipboard operation result.
        """
        user_input = context.user_input.lower()
        parameters = context.parameters

        # Determine the operation
        if "paste" in user_input:
            return await self._paste(context)
        elif "copy" in user_input:
            return await self._copy(context)
        elif "clear" in user_input:
            return await self._clear()
        elif "what's" in user_input or "show" in user_input or "read" in user_input:
            return await self._read()
        elif "history" in user_input:
            return await self._show_history()
        else:
            # Default: show current clipboard
            return await self._read()

    async def _read(self) -> SkillResult:
        """
        Read current clipboard content.

        Returns:
            SkillResult with clipboard content.
        """
        try:
            from roxy.macos.applescript import get_applescript_runner

            runner = get_applescript_runner()
            content = await runner.get_clipboard()

            if not content:
                return SkillResult(
                    success=True,
                    response_text="Your clipboard is empty",
                    speak=True,
                )

            # Add to history
            self._add_to_history(content)

            # Truncate long content for display
            display_content = content
            if len(content) > 200:
                display_content = content[:200] + "..."

            response = f"Your clipboard contains: {display_content}"

            return SkillResult(
                success=True,
                response_text=response,
                speak=True,
                response_text_for_speech=f"Your clipboard contains {len(content)} characters",
                data={"content": content, "length": len(content)},
            )

        except Exception as e:
            logger.error(f"Error reading clipboard: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't read your clipboard: {e}",
                speak=True,
            )

    async def _copy(self, context: SkillContext) -> SkillResult:
        """
        Copy content to clipboard.

        Args:
            context: Skill context containing content to copy.

        Returns:
            SkillResult with copy operation result.
        """
        # Get content to copy
        # This could be from the last assistant response or explicitly provided
        content = context.parameters.get("content")

        if not content:
            # Try to get last assistant response from conversation history
            if context.conversation_history:
                for msg in reversed(context.conversation_history):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        break

        if not content:
            return SkillResult(
                success=False,
                response_text="What would you like me to copy to the clipboard?",
                speak=False,
            )

        try:
            from roxy.macos.applescript import get_applescript_runner

            runner = get_applescript_runner()
            success = await runner.set_clipboard(content)

            if success:
                self._add_to_history(content)
                return SkillResult(
                    success=True,
                    response_text="Copied to clipboard",
                    speak=True,
                    data={"content": content, "length": len(content)},
                )
            else:
                return SkillResult(
                    success=False,
                    response_text="Sorry, I couldn't copy to the clipboard",
                    speak=True,
                )

        except Exception as e:
            logger.error(f"Error copying to clipboard: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't copy to the clipboard: {e}",
                speak=True,
            )

    async def _paste(self, context: SkillContext) -> SkillResult:
        """
        Paste clipboard content.

        Note: This simulates paste by getting clipboard content.
        Actual pasting requires GUI automation.

        Args:
            context: Skill context.

        Returns:
            SkillResult with paste operation result.
        """
        try:
            from roxy.macos.applescript import get_applescript_runner

            runner = get_applescript_runner()
            content = await runner.get_clipboard()

            if not content:
                return SkillResult(
                    success=True,
                    response_text="Your clipboard is empty, nothing to paste",
                    speak=True,
                )

            # Simulate paste using Cmd+V via AppleScript
            script = """
            tell application "System Events"
                keystroke "v" using command down
            end tell
            """

            await runner.run(script)

            return SkillResult(
                success=True,
                response_text="Pasted from clipboard",
                speak=True,
                data={"content": content},
            )

        except Exception as e:
            logger.error(f"Error pasting from clipboard: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't paste: {e}",
                speak=True,
            )

    async def _clear(self) -> SkillResult:
        """
        Clear clipboard content.

        Returns:
            SkillResult with clear operation result.
        """
        try:
            from roxy.macos.applescript import get_applescript_runner

            runner = get_applescript_runner()
            success = await runner.set_clipboard("")

            if success:
                return SkillResult(
                    success=True,
                    response_text="Clipboard cleared",
                    speak=True,
                )
            else:
                return SkillResult(
                    success=False,
                    response_text="Sorry, I couldn't clear the clipboard",
                    speak=True,
                )

        except Exception as e:
            logger.error(f"Error clearing clipboard: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't clear the clipboard: {e}",
                speak=True,
            )

    async def _show_history(self) -> SkillResult:
        """
        Show clipboard history.

        Returns:
            SkillResult with clipboard history.
        """
        if not self._history:
            return SkillResult(
                success=True,
                response_text="No clipboard history yet",
                speak=True,
            )

        response = f"Clipboard history ({len(self._history)} items):"

        for i, entry in enumerate(self._history, 1):
            content = entry.get("content", "")
            preview = content[:50] + "..." if len(content) > 50 else content
            response += f"\n{i}. {preview}"

        return SkillResult(
            success=True,
            response_text=response,
            speak=True,
            response_text_for_speech=f"You have {len(self._history)} items in clipboard history",
            data={"history": self._history},
        )

    def _add_to_history(self, content: str) -> None:
        """
        Add content to clipboard history.

        Args:
            content: Clipboard content.
        """
        import time

        self._history.append(
            {
                "content": content,
                "timestamp": time.time(),
                "length": len(content),
            }
        )

        # Trim history if needed
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

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

        # Check for clipboard-related keywords
        clipboard_keywords = ["clipboard", "copy", "paste"]
        if any(keyword in intent.lower() for keyword in clipboard_keywords):
            return 0.7

        return 0.0
