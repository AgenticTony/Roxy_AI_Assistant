"""Calendar skill for Roxy.

Interacts with macOS Calendar.app using AppleScript.
All data stays local - no cloud access.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from roxy.macos.applescript import (
    escape_applescript_string,
    get_applescript_runner,
    get_joinlist_handler,
)
from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class CalendarSkill(RoxySkill):
    """
    Calendar skill using AppleScript to access Calendar.app.

    Features:
    - Get events for today or specific date
    - Create new events
    - All data processed locally (no cloud)
    """

    name: str = "calendar"
    description: str = "Access and manage calendar events"
    triggers: list[str] = [
        "what's on my calendar",
        "calendar today",
        "schedule",
        "my events",
        "what's coming up",
        "create calendar event",
        "add event",
    ]
    permissions: list[Permission] = [Permission.APPLESCRIPT, Permission.CALENDAR]
    requires_cloud: bool = False

    def __init__(self) -> None:
        """Initialize calendar skill."""
        super().__init__()
        self._applescript = get_applescript_runner()

    def _parse_date(self, text: str) -> str:
        """Parse date from user input.

        Args:
            text: User input text.

        Returns:
            Date string in format for AppleScript.
        """
        text_lower = text.lower()

        # Today
        if "today" in text_lower:
            return "date (current date)"

        # Tomorrow
        if "tomorrow" in text_lower:
            return "date (current date) + 1 * days"

        # This week
        if "this week" in text_lower or "week" in text_lower:
            # Get start of week (Monday)
            return "date (current date)"

        # Next week
        if "next week" in text_lower:
            return "date (current date) + 7 * days"

        # Try to extract specific date (e.g., "January 15", "15th", "on Friday")
        date_pattern = r"(?:on\s+)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})"
        match = re.search(date_pattern, text)

        if match:
            # Escape user input to prevent AppleScript injection
            date_str = escape_applescript_string(match.group(1))
            return f'date "{date_str}"'

        # Default to today
        return "date (current date)"

    async def get_events(self, date: str | None = None) -> list[dict]:
        """Get calendar events for a date.

        Args:
            date: Date string or None for today.

        Returns:
            List of event dicts with summary, start, end, location.
        """
        if date is None:
            date = "date (current date)"

        script = f"""
        tell application "Calendar"
            activate
            set todayEvents to {{}}
            set todayDate to {date}
            set startOfDay to todayDate - (time of todayDate)
            set endOfDay to startOfDay + 1 * days

            tell calendar "Home"
                set todayEvents to every event whose start date is greater than or equal to startOfDay and start date is less than endOfDay
            end tell

            set eventList to {{}}
            repeat with evt in todayEvents
                set eventInfo to {{}}
                set end of eventInfo to summary of evt
                set end of eventInfo to start date of evt as string
                set end of eventInfo to end date of evt as string
                if location of evt is not missing value then
                    set end of eventInfo to location of evt
                else
                    set end of eventInfo to ""
                end if
                set end of eventList to my listToString(eventInfo)
            end repeat

            return my joinList(eventList, "|||")
        end tell

        on listToString(lst)
            return lst as string
        end listToString
{get_joinlist_handler()}
        """

        try:
            output = await self._applescript.run(script)

            if not output or output == "":
                return []

            # Parse the output
            events = []
            for event_str in output.split("|||"):
                parts = event_str.split(", ")
                if len(parts) >= 3:
                    events.append(
                        {
                            "summary": parts[0].strip(),
                            "start": parts[1].strip(),
                            "end": parts[2].strip(),
                            "location": parts[3].strip() if len(parts) > 3 else "",
                        }
                    )

            return events

        except Exception as e:
            logger.error(f"Error getting calendar events: {e}")
            return []

    async def create_event(
        self,
        title: str,
        start_date: str,
        end_date: str | None = None,
        location: str = "",
        notes: str = "",
    ) -> bool:
        """Create a new calendar event.

        Args:
            title: Event title.
            start_date: Start date/time.
            end_date: End date/time (defaults to 1 hour after start).
            location: Event location.
            notes: Event notes.

        Returns:
            True if event created successfully.
        """
        if end_date is None:
            # Default to 1 hour duration - start_date is expected to be
            # an AppleScript date expression (e.g., "Monday at 2pm", current date)
            # so we use it directly without escaping
            end_date = f"{start_date} + 1 * hours"

        # Escape all user input to prevent AppleScript injection
        title_safe = escape_applescript_string(title)
        location_safe = escape_applescript_string(location)
        notes_safe = escape_applescript_string(notes)

        script = f"""
        tell application "Calendar"
            activate
            tell calendar "Home"
                set newEvent to make new event at end of events with properties {{summary:"{title_safe}", start date:{start_date}, end date:{end_date}}}
                if "{location_safe}" is not "" then
                    set location of newEvent to "{location_safe}"
                end if
                if "{notes_safe}" is not "" then
                    set description of newEvent to "{notes_safe}"
                end if
            end tell
        end tell
        """

        try:
            await self._applescript.run(script)
            return True
        except Exception as e:
            logger.error(f"Error creating event: {e}")
            return False

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute calendar skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with calendar information.
        """
        user_input = context.user_input.lower()

        # Check if creating an event
        create_keywords = ["create event", "add event", "new event", "schedule"]
        if any(kw in user_input for kw in create_keywords):
            # Extract event details
            # This is simplified - real implementation would use NLP
            title = context.parameters.get("title", "New Event")
            start_date = context.parameters.get("start_date", "current date")

            success = await self.create_event(title=title, start_date=start_date)

            if success:
                return SkillResult(
                    success=True,
                    response_text=f"Created event: {title}",
                )
            else:
                return SkillResult(
                    success=False,
                    response_text="Sorry, I couldn't create that event.",
                )

        # Default: get events
        date_str = self._parse_date(user_input)
        events = await self.get_events(date_str)

        if not events:
            # Determine which date we checked
            if "today" in user_input or "today" in date_str:
                date_display = "today"
            elif "tomorrow" in user_input or "tomorrow" in date_str:
                date_display = "tomorrow"
            else:
                date_display = "that day"

            return SkillResult(
                success=True,
                response_text=f"You don't have any events scheduled {date_display}.",
                data={"events": [], "date": date_str},
            )

        # Format events for display
        lines = [f"Found {len(events)} event{'s' if len(events) > 1 else ''}:\n"]

        for i, event in enumerate(events, 1):
            lines.append(f"{i}. {event['summary']}")
            lines.append(f"   Time: {event['start']}")
            if event["location"]:
                lines.append(f"   Location: {event['location']}")
            lines.append("")

        return SkillResult(
            success=True,
            response_text="\n".join(lines),
            data={"events": events, "date": date_str},
        )
