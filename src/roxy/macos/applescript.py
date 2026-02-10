"""AppleScript runner for macOS automation.

Provides a simple interface to execute AppleScript commands from Python.
Includes pre-built template methods for common macOS automation tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def get_joinlist_handler() -> str:
    """Return AppleScript joinList handler code for use in scripts.

    This is a helper function that joins list items with a delimiter.
    It's commonly used in AppleScript scripts that return structured data.

    Returns:
        AppleScript code for the joinList handler.

    Examples:
        >>> script = f'''
        ... tell application "Finder"
        ...     set fileList to every file
        ...     return my joinList(fileList, "|||")
        ... end tell
        ...
        ... {get_joinlist_handler()}
        ... '''
    """
    return '''
        on joinList(lst, delimiter)
            set out to ""
            repeat with itemNum from 1 to count of lst
                if itemNum > 1 then set out to out & delimiter
                set out to out & item itemNum of lst
            end repeat
            return out
        end joinList
    '''


def escape_applescript_string(s: str) -> str:
    """Escape a string for safe use in AppleScript.

    AppleScript requires escaping of special characters to prevent
    injection attacks. This function handles:
    - Backslashes: \\
    - Double quotes: \\"
    - Single quotes: \\' (in some contexts)
    - Line feeds: \\n
    - Carriage returns: \\r
    - Tab characters: \\t
    - Null bytes: stripped (security risk)
    - Backticks, semicolons, pipe characters: context-dependent

    The escaping order is important - backslashes must be escaped first
    to avoid double-escaping previously escaped characters.

    Args:
        s: String to escape

    Returns:
        Safely escaped string for AppleScript

    Examples:
        >>> escape_applescript_string('Hello "World"')
        'Hello \\"World\\"'
        >>> escape_applescript_string('path\\to\\file')
        'path\\\\to\\\\file'
        >>> escape_applescript_string('line1\\nline2')
        'line1\\\\nline2'
    """
    if not isinstance(s, str):
        s = str(s)

    # Strip null bytes (security risk)
    s = s.replace('\x00', '')

    # Replace backslash first (to avoid double-escaping)
    s = s.replace("\\", "\\\\")

    # Then escape quotes
    s = s.replace('"', '\\"')
    # Note: Single quotes generally don't need escaping in AppleScript strings
    # but we escape backslash-singlequote for shell context safety
    s = s.replace("'", "\\'")

    # Handle line breaks
    s = s.replace('\n', '\\n')
    s = s.replace('\r', '\\r')

    # Handle tabs
    s = s.replace('\t', '\\t')

    # Handle other potentially dangerous characters in string context
    # Percent sign can be used for variable interpolation
    s = s.replace('%', '\\%')

    # Dollar sign (variable interpolation)
    s = s.replace('$', '\\$')

    # Backtick (command substitution in shell)
    s = s.replace('`', '\\`')

    # Remove any remaining control characters except common whitespace
    import re
    s = re.sub(r'[\x01-\x08\x0b-\x0c\x0e-\x1f]', '', s)

    return s


class AppleScriptRunner:
    """
    Runner for executing AppleScript commands.

    Provides both direct script execution and pre-built template methods
    for common macOS automation tasks like controlling applications,
    accessing system features, and getting information from apps.
    """

    def __init__(self) -> None:
        """Initialize AppleScript runner."""
        self._available: bool | None = None

    def _check_available(self) -> bool:
        """Check if osascript is available.

        Returns:
            True if osascript command is available.
        """
        if self._available is not None:
            return self._available

        try:
            result = subprocess.run(
                ["which", "osascript"],
                capture_output=True,
                text=True,
            )
            self._available = result.returncode == 0
            return self._available
        except Exception:
            self._available = False
            return False

    async def run(self, script: str) -> str:
        """
        Execute an AppleScript command asynchronously.

        Args:
            script: AppleScript code to execute.

        Returns:
            Script output as string.

        Raises:
            RuntimeError: If osascript is not available.
            subprocess.CalledProcessError: If script execution fails.
        """
        if not self._check_available():
            raise RuntimeError("osascript is not available on this system")

        logger.debug(f"Executing AppleScript: {script[:100]}...")

        try:
            process = await asyncio.create_subprocess_exec(
                "osascript",
                "-e",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode().strip()
                logger.error(f"AppleScript error: {error_msg}")
                raise subprocess.CalledProcessError(
                    process.returncode, "osascript", stderr=error_msg
                )

            output = stdout.decode().strip()
            logger.debug(f"AppleScript result: {output[:100]}...")
            return output

        except subprocess.CalledProcessError:
            raise
        except Exception as e:
            logger.error(f"Failed to execute AppleScript: {e}")
            raise

    async def run_jxa(self, script: str) -> str:
        """
        Execute JavaScript for Automation (JXA) code asynchronously.

        Args:
            script: JavaScript code to execute.

        Returns:
            Script output as string.

        Raises:
            RuntimeError: If osascript is not available.
            subprocess.CalledProcessError: If script execution fails.
        """
        if not self._check_available():
            raise RuntimeError("osascript is not available on this system")

        logger.debug(f"Executing JXA: {script[:100]}...")

        try:
            process = await asyncio.create_subprocess_exec(
                "osascript",
                "-l",
                "JavaScript",
                "-e",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode().strip()
                logger.error(f"JXA error: {error_msg}")
                raise subprocess.CalledProcessError(
                    process.returncode, "osascript", stderr=error_msg
                )

            output = stdout.decode().strip()
            logger.debug(f"JXA result: {output[:100]}...")
            return output

        except subprocess.CalledProcessError:
            raise
        except Exception as e:
            logger.error(f"Failed to execute JXA: {e}")
            raise

    async def run_and_parse_json(self, script: str) -> dict[str, Any]:
        """
        Execute AppleScript and parse JSON output.

        Args:
            script: AppleScript that outputs JSON.

        Returns:
            Parsed JSON as dict.

        Raises:
            ValueError: If output is not valid JSON.
        """
        output = await self.run(script)
        return json.loads(output)

    # Pre-built template methods for common macOS automation tasks

    async def get_running_apps(self) -> list[dict[str, Any]]:
        """
        Get list of currently running applications.

        Returns:
            List of dicts with app name, bundle ID, and if it's frontmost.
        """
        script = '''
        tell application "System Events"
            set runningApps to (name of every process whose background only is false) as list
            set appList to {}
            repeat with appName in runningApps
                try
                    set appID to bundle identifier of application appName
                    set isFront to (appName is in (name of processes whose frontmost is true))
                    set appInfo to {name:appName, bundleID:appID, frontmost:isFront}
                    set end of appList to appInfo
                end try
            end repeat
            return appList as string
        end tell
        '''
        output = await self.run(script)
        # Parse output - AppleScript returns complex structure
        # For simplicity, return basic list from JXA instead
        jxa_script = '''
        var apps = Application("System Events").processes whose({
            backgroundOnly: false
        });
        var appList = [];
        for (var i = 0; i < apps.length; i++) {
            try {
                var app = apps[i];
                appList.push({
                    name: app.name(),
                    bundleID: app.bundleIdentifier(),
                    frontmost: app.frontmost()
                });
            } catch (e) {}
        }
        JSON.stringify(appList);
        '''
        result = json.loads(await self.run_jxa(jxa_script))
        # Convert camelCase keys to snake_case
        converted = []
        for app in result:
            converted.append({
                "name": app.get("name"),
                "bundle_id": app.get("bundleID"),
                "frontmost": app.get("frontmost")
            })
        return converted

    async def get_frontmost_app(self) -> dict[str, Any]:
        """
        Get information about the frontmost application.

        Returns:
            Dict with app name, bundle ID, and window title if available.
        """
        script = '''
        tell application "System Events"
            set frontApp to (name of first application process whose frontmost is true)
            try
                set frontAppID to bundle identifier of application frontApp
            on error
                set frontAppID to "unknown"
            end try
            return frontApp & "||" & frontAppID
        end tell
        '''
        output = await self.run(script)
        parts = output.split("||")
        return {"name": parts[0], "bundle_id": parts[1] if len(parts) > 1 else "unknown"}

    async def open_url(self, url: str) -> bool:
        """
        Open a URL in the default browser.

        Args:
            url: The URL to open.

        Returns:
            True if successful.
        """
        url_esc = escape_applescript_string(url)
        script = f'''
        tell application "System Events"
            open location "{url_esc}"
        end tell
        '''
        try:
            await self.run(script)
            return True
        except Exception:
            return False

    async def send_notification(self, title: str, message: str) -> bool:
        """
        Send a macOS notification.

        Args:
            title: Notification title.
            message: Notification message body.

        Returns:
            True if successful.
        """
        title_esc = escape_applescript_string(title)
        message_esc = escape_applescript_string(message)

        script = f'''
        display notification "{message_esc}" with title "{title_esc}"
        '''
        try:
            await self.run(script)
            return True
        except Exception:
            return False

    async def get_clipboard(self) -> str:
        """
        Get the current clipboard contents.

        Returns:
            Clipboard text content.
        """
        script = '''
        tell application "System Events"
            try
                set theClipboard to the clipboard
                return theClipboard
            on error
                return ""
            end try
        end tell
        '''
        return await self.run(script)

    async def set_clipboard(self, text: str) -> bool:
        """
        Set the clipboard contents.

        Args:
            text: Text to copy to clipboard.

        Returns:
            True if successful.
        """
        text_esc = escape_applescript_string(text)

        script = f'''
        tell application "System Events"
            try
                set the clipboard to "{text_esc}"
                return "success"
            on error
                return "error"
            end try
        end tell
        '''
        try:
            result = await self.run(script)
            return result == "success"
        except Exception:
            return False

    async def get_safari_tabs(self) -> list[dict[str, Any]]:
        """
        Get URLs and titles of open Safari tabs.

        Returns:
            List of dicts with window index, tab index, title, and URL.
        """
        script = '''
        tell application "Safari"
            if not running then
                return "[]"
            end if
            set windowList to {}
            repeat with w in windows
                set tabIndex to 0
                repeat with t in tabs of w
                    set tabTitle to name of t
                    set tabURL to URL of t
                    set tabInfo to {{windowIndex:index of w as integer, tabIndex:tabIndex as integer, title:tabTitle, url:tabURL}}
                    set end of windowList to tabInfo
                    set tabIndex to tabIndex + 1
                end repeat
            end repeat
            return my jsonify(windowList)
        end tell

        on jsonify(theList)
            return do shell script "echo " & quoted form of (my deepCopy(theList) as string) & " | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()))'"
        end jsonify

        on deepCopy(theList)
            -- Simple list copy for JSON serialization
            -- This is a simplified implementation
            return theList as string
        end deepCopy
        '''
        # Use JXA for cleaner JSON handling
        jxa_script = '''
        var Safari = Application("Safari");
        if (!Safari.running()) {
            throw new Error("Safari is not running");
        }

        var windows = Safari.windows();
        var result = [];

        for (var i = 0; i < windows.length; i++) {
            var window = windows[i];
            var tabs = window.tabs();

            for (var j = 0; j < tabs.length; j++) {
                var tab = tabs[j];
                result.push({
                    windowIndex: i,
                    tabIndex: j,
                    title: tab.name(),
                    url: tab.url()
                });
            }
        }

        JSON.stringify(result);
        '''
        try:
            return json.loads(await self.run_jxa(jxa_script))
        except Exception:
            return []

    async def get_mail_unread_count(self) -> int:
        """
        Get the number of unread emails in Mail.app.

        Returns:
            Number of unread messages.
        """
        script = '''
        tell application "Mail"
            if not running then
                return "0"
            end if
            set unreadCount to (count of messages in inbox whose read status is false)
            return unreadCount as string
        end tell
        '''
        try:
            result = await self.run(script)
            return int(result)
        except Exception:
            return 0

    async def get_mail_subjects(self, limit: int = 10) -> list[str]:
        """
        Get subjects of recent unread emails.

        Args:
            limit: Maximum number of subjects to return.

        Returns:
            List of email subjects.
        """
        script = f'''
        tell application "Mail"
            if not running then
                return "[]"
            end if
            set unreadMessages to messages in inbox whose read status is false
            set subjectList to {{}}
            set counter to 0
            repeat with msg in unreadMessages
                if counter >= {limit} then
                    exit repeat
                end if
                set end of subjectList to subject of msg
                set counter to counter + 1
            end repeat
            return my jsonify(subjectList)
        end tell

        on jsonify(theList)
            set jsonString to "["
            repeat with i from 1 to count of theList
                set itemText to item i of theList
                if i > 1 then
                    set jsonString to jsonString & ","
                end if
                set jsonString to jsonString & "\"" & my escapeJson(itemText) & "\""
            end repeat
            set jsonString to jsonString & "]"
            return jsonString
        end jsonify

        on escapeJson(text)
            set text to text's text items
            set newText to ""
            repeat with t in text
                if t is "\"" then
                    set newText to newText & "\\\\\\\""
                else if t is "\\" then
                    set newText to newText & "\\\\\\\\"
                else
                    set newText to newText & t
                end if
            end repeat
            return newText
        end escapeJson
        '''
        try:
            return json.loads(await self.run(script))
        except Exception:
            return []

    async def get_calendar_events_today(self) -> list[dict[str, Any]]:
        """
        Get today's calendar events.

        Returns:
            List of dicts with event summary, start time, and end time.
        """
        # Get today's date in format for AppleScript - this is safe since
        # we control the format string and datetime.now() returns trusted data
        today = escape_applescript_string(datetime.now().strftime("%A, %B %d, %Y"))

        script = f'''
        tell application "Calendar"
            if not running then
                return "[]"
            end if
            set todayDate to date "{today}"
            set todaysEvents to every event whose start date is todayDate
            set eventList to {{}}
            repeat with evt in todaysEvents
                set eventInfo to {{
                    summary:summary of evt,
                    startDate:start date of evt as string,
                    endDate:end date of evt as string,
                    location:location of evt
                }}
                set end of eventList to eventInfo
            end repeat
            return my jsonify(eventList)
        end tell

        on jsonify(theList)
            -- Simplified JSON conversion
            set result to "["
            repeat with i from 1 to count of theList
                if i > 1 then
                    set result to result & ","
                end if
                set evt to item i of theList
                set result to result & "{{" & quote & "summary" & quote & ":" & quote & my escapeJson(summary of evt) & quote & "}}"
            end repeat
            return result & "]"
        end jsonify

        on escapeJson(text)
            -- Basic JSON escaping
            return text
        end escapeJson
        '''
        # Use JXA for better date handling
        jxa_script = f'''
        var Calendar = Application("Calendar");
        if (!Calendar.running()) {{
            throw new Error("Calendar is not running");
        }}

        var today = new Date();
        today.setHours(0, 0, 0, 0);
        var tomorrow = new Date(today);
        tomorrow.setDate(tomorrow.getDate() + 1);

        var calendars = Calendar.calendars;
        var result = [];

        for (var i = 0; i < calendars.length; i++) {{
            var events = calendars[i].events();
            for (var j = 0; j < events.length; j++) {{
                var evt = events[j];
                var startDate = evt.startDate();
                var endDate = evt.endDate();

                if (startDate >= today && startDate < tomorrow) {{
                    result.push({{
                        summary: evt.summary(),
                        startDate: startDate.toString(),
                        endDate: endDate.toString(),
                        location: evt.location()
                    }});
                }}
            }}
        }}

        JSON.stringify(result);
        '''
        try:
            return json.loads(await self.run_jxa(jxa_script))
        except Exception as e:
            logger.error(f"Error getting calendar events: {e}")
            return []

    async def get_notes_list(self) -> list[dict[str, Any]]:
        """
        Get list of notes from Notes.app.

        Returns:
            List of dicts with note name, body preview, and modification date.
        """
        jxa_script = '''
        var Notes = Application("Notes");
        if (!Notes.running()) {
            throw new Error("Notes is not running");
        }

        var folders = Note.folders();
        var result = [];

        for (var i = 0; i < folders.length; i++) {
            var notes = folders[i].notes();
            for (var j = 0; j < notes.length; j++) {
                var note = notes[j];
                var body = note.body();
                var preview = body.substring(0, 200) + (body.length > 200 ? "..." : "");

                result.push({
                    name: note.name(),
                    body: preview,
                    modified: note.modificationDate().toString(),
                    folder: folders[i].name()
                });
            }
        }

        JSON.stringify(result);
        '''
        try:
            return json.loads(await self.run_jxa(jxa_script))
        except Exception as e:
            logger.error(f"Error getting notes: {e}")
            return []

    async def create_note(self, title: str, body: str) -> bool:
        """
        Create a new note in Notes.app.

        Args:
            title: Note title.
            body: Note body content.

        Returns:
            True if successful.
        """
        title_esc = escape_applescript_string(title)
        body_esc = escape_applescript_string(body)

        jxa_script = f'''
        var Notes = Application("Notes");
        if (!Notes.running()) {{
            Notes.activate();
            delay(1);
        }}

        var newNote = Note({{
            name: "{title_esc}",
            body: "{body_esc}"
        }});

        Notes.defaultFolder().notes.push(newNote);
        '''
        try:
            await self.run_jxa(jxa_script)
            return True
        except Exception as e:
            logger.error(f"Error creating note: {e}")
            return False


# Singleton instance
_applescript_runner: AppleScriptRunner | None = None


def get_applescript_runner() -> AppleScriptRunner:
    """Get or create the AppleScript runner singleton.

    Returns:
        AppleScriptRunner instance.
    """
    global _applescript_runner

    if _applescript_runner is None:
        _applescript_runner = AppleScriptRunner()

    return _applescript_runner
