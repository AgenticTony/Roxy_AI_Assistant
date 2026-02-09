"""Notes skill for Roxy.

Interacts with macOS Notes.app using AppleScript.
All notes data stays local.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult
from roxy.macos.applescript import get_applescript_runner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class NotesSkill(RoxySkill):
    """
    Notes skill using AppleScript to access Notes.app.

    Features:
    - Create new notes
    - Search notes by title/body
    - All data stays local
    """

    name: str = "notes"
    description: str = "Create and search notes"
    triggers: list[str] = [
        "create note",
        "find note",
        "read notes",
        "search notes",
        "take a note",
        "new note",
    ]
    permissions: list[Permission] = [Permission.APPLESCRIPT]
    requires_cloud: bool = False

    def __init__(self) -> None:
        """Initialize notes skill."""
        super().__init__()
        self._applescript = get_applescript_runner()

    async def create_note(self, title: str, content: str, folder: str = "Notes") -> bool:
        """Create a new note.

        Args:
            title: Note title.
            content: Note body content.
            folder: Folder to create note in (default: "Notes").

        Returns:
            True if note created successfully.
        """
        script = f"""
        tell application "Notes"
            activate
            tell account "iCloud"
                set newNote to make new note at folder "{folder}" with properties {{name:"{title}", body:"{content}"}}
            end tell
        end tell
        """

        try:
            self._applescript.run(script)
            logger.info(f"Created note: {title}")
            return True
        except Exception as e:
            logger.error(f"Error creating note: {e}")
            return False

    async def search_notes(self, query: str, limit: int = 10) -> list[dict]:
        """Search notes by title or body content.

        Args:
            query: Search query.
            limit: Maximum results to return.

        Returns:
            List of note dicts with name, body, modified date.
        """
        script = f"""
        tell application "Notes"
            activate
            set foundNotes to {{}}

            tell account "iCloud"
                set allNotes to every note

                repeat with noteRef in allNotes
                    set noteName to name of noteRef
                    set noteBody to body of noteRef

                    if noteName contains "{query}" or noteBody contains "{query}" then
                        set noteInfo to {{}}
                        set end of noteInfo to noteName
                        set end of noteInfo to noteBody
                        set end of noteInfo to modification date of noteRef as string

                        set end of foundNotes to my joinList(noteInfo, "|||")

                        if (count of foundNotes) ≥ {limit} then exit repeat
                    end if
                end repeat
            end tell

            return my joinList(foundNotes, ";;;")
        end tell

        on joinList(lst, delimiter)
            set out to ""
            repeat with itemNum from 1 to count of lst
                if itemNum > 1 then set out to out & delimiter
                set out to out & item itemNum of lst
            end repeat
            return out
        end joinList
        """

        try:
            output = self._applescript.run(script)

            if not output or output == "":
                return []

            notes = []
            for note_str in output.split(";;;"):
                parts = note_str.split("|||")
                if len(parts) >= 3:
                    # Truncate body for display
                    body = parts[1].strip()
                    if len(body) > 200:
                        body = body[:200] + "..."

                    notes.append({
                        "name": parts[0].strip(),
                        "body": body,
                        "modified": parts[2].strip(),
                    })

            return notes

        except Exception as e:
            logger.error(f"Error searching notes: {e}")
            return []

    async def get_note(self, name: str) -> dict | None:
        """Get a specific note by name.

        Args:
            name: Note name.

        Returns:
            Note dict with name and body, or None if not found.
        """
        script = f"""
        tell application "Notes"
            activate
            tell account "iCloud"
                set foundNote to first note whose name is "{name}"

                set noteInfo to {{}}
                set end of noteInfo to name of foundNote
                set end of noteInfo to body of foundNote
                set end of noteInfo to modification date of foundNote as string

                return my joinList(noteInfo, "|||")
            end tell
        end tell

        on joinList(lst, delimiter)
            set out to ""
            repeat with itemNum from 1 to count of lst
                if itemNum > 1 then set out to out & delimiter
                set out to out & item itemNum of lst
            end repeat
            return out
        end joinList
        """

        try:
            output = self._applescript.run(script)

            if not output or output == "":
                return None

            parts = output.split("|||")
            if len(parts) >= 3:
                return {
                    "name": parts[0].strip(),
                    "body": parts[1].strip(),
                    "modified": parts[2].strip(),
                }

            return None

        except Exception as e:
            logger.error(f"Error getting note: {e}")
            return None

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute notes skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with notes information.
        """
        user_input = context.user_input.lower()

        # Check if creating a note
        create_keywords = ["create note", "take a note", "new note", "add note"]
        if any(kw in user_input for kw in create_keywords):
            # Extract title and content from parameters or input
            title = context.parameters.get("title", "New Note")

            # Get content - could be rest of input or from parameters
            content = context.parameters.get("content", "Note created by Roxy")

            success = await self.create_note(title=title, content=content)

            if success:
                return SkillResult(
                    success=True,
                    response_text=f"Created note: {title}",
                    data={"title": title},
                )
            else:
                return SkillResult(
                    success=False,
                    response_text="Sorry, I couldn't create that note.",
                )

        # Check if reading a specific note
        read_keywords = ["read note", "open note", "show note"]
        if any(kw in user_input for kw in read_keywords):
            # Try to extract note name
            # Look for "called X", "named X", "X note"
            name_match = re.search(r'(?:called|named)\s+"?([^"]+)"?', user_input)
            if name_match:
                note_name = name_match.group(1).strip()
            else:
                # Fallback: look for a quoted string
                quote_match = re.search(r'"([^"]+)"', context.user_input)
                if quote_match:
                    note_name = quote_match.group(1)
                else:
                    return SkillResult(
                        success=False,
                        response_text="What note would you like me to read?",
                        follow_up="Please specify the note name.",
                    )

            note = await self.get_note(note_name)

            if note:
                return SkillResult(
                    success=True,
                    response_text=f"Note: {note['name']}\n\n{note['body']}",
                    data={"note": note},
                )
            else:
                return SkillResult(
                    success=False,
                    response_text=f"Couldn't find a note called '{note_name}'.",
                )

        # Default: search notes
        # Extract search query from input
        for trigger in ["find note", "search notes", "search for"]:
            if trigger in user_input:
                query = user_input.replace(trigger, "").strip()
                for word in ["for", "in"]:
                    if query.startswith(word + " "):
                        query = query[len(word + " "):].strip()
                break
        else:
            # No trigger found, use whole input as query
            query = user_input

        if not query:
            return SkillResult(
                success=False,
                response_text="What would you like me to search for in your notes?",
                follow_up="Please provide a search term.",
            )

        notes = await self.search_notes(query)

        if not notes:
            return SkillResult(
                success=True,
                response_text=f"Couldn't find any notes matching '{query}'.",
                data={"notes": [], "query": query},
            )

        # Format results
        lines = [f"Found {len(notes)} note{'s' if len(notes) > 1 else ''} matching '{query}':\n"]

        for i, note in enumerate(notes, 1):
            lines.append(f"{i}. {note['name']}")
            lines.append(f"   {note['body'][:100]}...")
            lines.append("")

        return SkillResult(
            success=True,
            response_text="\n".join(lines),
            data={"notes": notes, "query": query},
        )
