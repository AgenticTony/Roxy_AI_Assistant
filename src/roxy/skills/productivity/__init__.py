"""Productivity skills for Roxy.

This package contains skills that interact with macOS productivity apps:
- CalendarSkill: Calendar.app integration
- EmailSkill: Mail.app integration
- NotesSkill: Notes.app integration
- RemindersSkill: Reminders.app integration

All skills use AppleScript and keep data local.
"""

from roxy.skills.productivity.calendar import CalendarSkill
from roxy.skills.productivity.email import EmailSkill
from roxy.skills.productivity.notes import NotesSkill
from roxy.skills.productivity.reminders import RemindersSkill

__all__ = [
    "CalendarSkill",
    "EmailSkill",
    "NotesSkill",
    "RemindersSkill",
]
