"""Reminders skill for Roxy.

Interacts with macOS Reminders.app using AppleScript.
All reminders data stays local.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult
from roxy.macos.applescript import escape_applescript_string, get_applescript_runner, get_joinlist_handler

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class RemindersSkill(RoxySkill):
    """
    Reminders skill using AppleScript to access Reminders.app.

    Features:
    - Add new reminders
    - List all reminders
    - All data stays local
    """

    name: str = "reminders"
    description: str = "Create and list reminders"
    triggers: list[str] = [
        "add reminder",
        "show reminders",
        "reminders list",
        "my reminders",
        "remind me to",
        "create reminder",
    ]
    permissions: list[Permission] = [Permission.APPLESCRIPT]
    requires_cloud: bool = False

    def __init__(self) -> None:
        """Initialize reminders skill."""
        super().__init__()
        self._applescript = get_applescript_runner()

    def _parse_due_date(self, text: str) -> str:
        """Parse due date from user input.

        Args:
            text: User input text.

        Returns:
            Date string for AppleScript.
        """
        text_lower = text.lower()

        # Today
        if "today" in text_lower:
            return "date (current date)"

        # Tomorrow
        if "tomorrow" in text_lower:
            return "date (current date) + 1 * days"

        # This week
        if "this week" in text_lower:
            return "date (current date) + 7 * days"

        # Next week
        if "next week" in text_lower:
            return "date (current date) + 14 * days"

        # Try specific dates (e.g., "January 15", "on Friday")
        date_pattern = r"(?:on\s+)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}/\d{1,2}/\d{4})"
        match = re.search(date_pattern, text)

        if match:
            # Escape user input to prevent AppleScript injection
            date_str = escape_applescript_string(match.group(1))
            return f'date "{date_str}"'

        # Default: no due date
        return "missing value"

    async def add_reminder(
        self,
        title: str,
        due_date: str | None = None,
        notes: str = "",
        list_name: str = "Reminders",
    ) -> bool:
        """Add a new reminder.

        Args:
            title: Reminder title/name.
            due_date: Due date as AppleScript date string.
            notes: Optional notes for the reminder.
            list_name: Reminders list name (default: "Reminders").

        Returns:
            True if reminder created successfully.
        """
        # Escape all user input to prevent AppleScript injection
        title_safe = escape_applescript_string(title)
        list_name_safe = escape_applescript_string(list_name)
        notes_safe = escape_applescript_string(notes) if notes else ""

        # Build due date clause
        if due_date and due_date != "missing value":
            due_clause = f"due date:{due_date}"
        else:
            due_clause = ""

        # Build notes clause
        if notes:
            notes_clause = f', body:"{notes_safe}"'
        else:
            notes_clause = ""

        script = f"""
        tell application "Reminders"
            activate
            tell list "{list_name_safe}"
                set newReminder to make new reminder with properties {{name:"{title_safe}"{notes_clause}}}
                {due_clause}
            end tell
        end tell
        """

        try:
            self._applescript.run(script)
            logger.info(f"Created reminder: {title}")
            return True
        except Exception as e:
            logger.error(f"Error creating reminder: {e}")
            return False

    async def list_reminders(self, list_name: str = "Reminders") -> list[dict]:
        """List all reminders.

        Args:
            list_name: Reminders list name (default: "Reminders").

        Returns:
            List of reminder dicts with name, due_date, completed status.
        """
        # Escape user input to prevent AppleScript injection
        list_name_safe = escape_applescript_string(list_name)

        script = f"""
        tell application "Reminders"
            activate
            tell list "{list_name_safe}"
                set allReminders to every reminder

                set reminderList to {{}}
                repeat with remRef in allReminders
                    set reminderInfo to {{}}
                    set end of reminderInfo to name of remRef

                    set dueDate to due date of remRef
                    if dueDate is not missing value then
                        set end of reminderInfo to dueDate as string
                    else
                        set end of reminderInfo to ""
                    end if

                    set end of reminderInfo to completed of remRef as boolean

                    set end of reminderList to my joinList(reminderInfo, "|||")
                end repeat

                return my joinList(reminderList, ";;;")
            end tell
        end tell
{get_joinlist_handler()}
        """

        try:
            output = self._applescript.run(script)

            if not output or output == "":
                return []

            reminders = []
            for rem_str in output.split(";;;"):
                parts = rem_str.split("|||")
                if len(parts) >= 3:
                    reminders.append({
                        "name": parts[0].strip(),
                        "due_date": parts[1].strip(),
                        "completed": parts[2].strip() == "true",
                    })

            return reminders

        except Exception as e:
            logger.error(f"Error listing reminders: {e}")
            return []

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute reminders skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with reminders information.
        """
        user_input = context.user_input.lower()

        # Check if adding a reminder
        add_keywords = ["add reminder", "create reminder", "remind me to", "new reminder"]
        if any(kw in user_input for kw in add_keywords):
            # Extract reminder title
            # Pattern: "remind me to [do something]"
            remind_match = re.search(r'remind me to\s+(.+?)(?:\s+(?:by|on|at)\s+|$)', user_input)
            if remind_match:
                title = remind_match.group(1).strip()
            else:
                # Try other patterns
                for kw in add_keywords:
                    if kw in user_input:
                        title = user_input.replace(kw, "").strip()
                        # Remove leading "to" if present
                        if title.startswith("to "):
                            title = title[3:].strip()
                        break
                else:
                    title = context.parameters.get("title", "New Reminder")

            # Get due date
            due_date = self._parse_due_date(user_input)

            # Get notes if any
            notes = context.parameters.get("notes", "")

            success = await self.add_reminder(
                title=title,
                due_date=due_date if due_date else None,
                notes=notes,
            )

            if success:
                response = f"Created reminder: {title}"
                if due_date and due_date != "missing value":
                    response += f" (due: {due_date})"
                return SkillResult(
                    success=True,
                    response_text=response,
                    data={"title": title, "due_date": due_date},
                )
            else:
                return SkillResult(
                    success=False,
                    response_text="Sorry, I couldn't create that reminder.",
                )

        # Default: list reminders
        list_name = context.parameters.get("list", "Reminders")
        reminders = await self.list_reminders(list_name)

        if not reminders:
            return SkillResult(
                success=True,
                response_text=f"You don't have any reminders in your {list_name} list.",
                data={"reminders": [], "list": list_name},
            )

        # Separate incomplete and completed
        incomplete = [r for r in reminders if not r["completed"]]
        completed = [r for r in reminders if r["completed"]]

        # Format for display
        lines = [f"Found {len(incomplete)} active reminder{'s' if len(incomplete) != 1 else ''}:\n"]

        for i, rem in enumerate(incomplete, 1):
            lines.append(f"{i}. {rem['name']}")
            if rem['due_date']:
                lines.append(f"   Due: {rem['due_date']}")
            lines.append("")

        if completed:
            lines.append(f"\nCompleted ({len(completed)}):")
            for rem in completed[:5]:  # Show max 5 completed
                lines.append(f"  - {rem['name']}")

        return SkillResult(
            success=True,
            response_text="\n".join(lines),
            data={"reminders": reminders, "list": list_name},
        )
