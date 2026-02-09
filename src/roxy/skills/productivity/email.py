"""Email skill for Roxy.

Interacts with macOS Mail.app using AppleScript.
Email content is processed locally - NEVER sent to cloud.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult
from roxy.macos.applescript import escape_applescript_string, get_applescript_runner, get_joinlist_handler

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class EmailSkill(RoxySkill):
    """
    Email skill using AppleScript to access Mail.app.

    Features:
    - Get latest email headers
    - Summarize emails using local LLM only
    - All email content stays local (NEVER sent to cloud)
    - Read-only access (no sending for privacy)
    """

    name: str = "email"
    description: str = "Access and summarize emails"
    triggers: list[str] = [
        "read my emails",
        "latest emails",
        "check email",
        "new emails",
        "inbox",
        "email summary",
    ]
    permissions: list[Permission] = [Permission.APPLESCRIPT, Permission.EMAIL]
    requires_cloud: bool = False  # Email stays local

    # Configuration
    DEFAULT_COUNT: int = 5
    MAX_COUNT: int = 10

    def __init__(self) -> None:
        """Initialize email skill."""
        super().__init__()
        self._applescript = get_applescript_runner()

    async def get_latest_emails(self, count: int = DEFAULT_COUNT) -> list[dict]:
        """Get latest email headers.

        Args:
            count: Number of emails to retrieve.

        Returns:
            List of email dicts with subject, sender, date, message_id.
        """
        count = min(count, self.MAX_COUNT)

        script = f"""
        tell application "Mail"
            activate
            set inbox to mailbox "INBOX"
            set recentMessages to (messages of inbox whose read status is false)

            set messageList to {{}}
            repeat with i from 1 to {count}
                if i > count of recentMessages then exit repeat
                set msg to item i of recentMessages

                set messageInfo to {{}}
                set end of messageInfo to subject of msg
                set end of messageInfo to sender of msg
                set end of messageInfo to date sent of msg as string
                set end of messageInfo to message id of msg

                set end of messageList to my joinList(messageInfo, "|||")
            end repeat

            return my joinList(messageList, ";;;")
        end tell
{get_joinlist_handler()}
        """

        try:
            output = self._applescript.run(script)

            if not output or output == "":
                return []

            emails = []
            for email_str in output.split(";;;"):
                parts = email_str.split("|||")
                if len(parts) >= 4:
                    emails.append({
                        "subject": parts[0].strip(),
                        "sender": parts[1].strip(),
                        "date": parts[2].strip(),
                        "message_id": parts[3].strip(),
                    })

            return emails

        except Exception as e:
            logger.error(f"Error getting emails: {e}")
            return []

    async def get_email_content(self, message_id: str) -> dict:
        """Get full email content by message ID.

        Args:
            message_id: Email message ID.

        Returns:
            Email dict with subject, sender, date, body.
        """
        message_id_safe = escape_applescript_string(message_id)
        script = f"""
        tell application "Mail"
            set msg to first message whose message id is "{message_id_safe}"

            set emailContent to {{}}
            set end of emailContent to subject of msg
            set end of emailContent to sender of msg
            set end of emailContent to date sent of msg as string
            set end of emailContent to content of msg

            return my joinList(emailContent, "|||")
        end tell
{get_joinlist_handler()}
        """

        try:
            output = self._applescript.run(script)

            if not output or output == "":
                return {}

            parts = output.split("|||")
            if len(parts) >= 4:
                return {
                    "subject": parts[0].strip(),
                    "sender": parts[1].strip(),
                    "date": parts[2].strip(),
                    "body": parts[3].strip(),
                }

            return {}

        except Exception as e:
            logger.error(f"Error getting email content: {e}")
            return {}

    async def summarize_email(self, context: SkillContext, message_id: str) -> str:
        """Summarize an email using local LLM.

        Args:
            context: SkillContext with access to local LLM client.
            message_id: Email message ID.

        Returns:
            Email summary as text.

        Note:
            This uses LOCAL Ollama only - never sends email to cloud.
        """
        email = await self.get_email_content(message_id)

        if not email:
            return "Could not retrieve email content."

        # Get the email body for summary
        content = email.get("body", "")

        # Try to use local LLM for email summarization
        if context.local_llm_client is not None and content:
            try:
                # Prepare prompt for email summarization
                prompt = f"""Summarize the following email concisely.

From: {email.get('sender', 'Unknown')}
Subject: {email.get('subject', 'No Subject')}
Date: {email.get('date', 'Unknown')}

Email Content:
```
{content[:3000]}
```

Provide a brief summary (2-3 sentences) covering:
1. Who sent the email
2. The main topic or purpose
3. Any key action items or deadlines

Format the response in a clear, readable way:"""

                response = await context.local_llm_client.generate(
                    prompt=prompt,
                    temperature=0.5,  # Moderate temperature for balanced summarization
                    max_tokens=200,
                )

                # Validate that we got a reasonable summary
                summary = response.content.strip()
                if summary and len(summary) > 20:
                    logger.info(f"Generated email summary using LLM for message {message_id}")
                    return f"""From: {email.get('sender', 'Unknown')}
Subject: {email.get('subject', 'No Subject')}
Date: {email.get('date', 'Unknown')}

{summary}"""
                else:
                    logger.warning(f"LLM returned invalid summary, falling back to simple format")

            except Exception as e:
                logger.error(f"Error generating email summary with LLM: {e}")
                # Fall through to simple format

        # Fallback to simple email display
        lines = [
            f"From: {email.get('sender', 'Unknown')}",
            f"Subject: {email.get('subject', 'No Subject')}",
            f"Date: {email.get('date', 'Unknown')}",
            "",
            "(Email content available - local LLM integration for summarization)",
        ]

        return "\n".join(lines)

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute email skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with email information.
        """
        user_input = context.user_input.lower()

        # Determine how many emails to retrieve
        count = self.DEFAULT_COUNT

        # Look for numbers in input
        import re
        number_match = re.search(r'(\d+)', user_input)
        if number_match:
            count = min(int(number_match.group(1)), self.MAX_COUNT)

        # Check if asking for summary of specific email
        if "summarize" in user_input or "read" in user_input:
            # For now, just get the latest email
            emails = await self.get_latest_emails(1)

            if emails:
                summary = await self.summarize_email(context, emails[0]["message_id"])
                return SkillResult(
                    success=True,
                    response_text=summary,
                    data={"email": emails[0]},
                )

        # Default: get latest email headers
        emails = await self.get_latest_emails(count)

        if not emails:
            return SkillResult(
                success=True,
                response_text="You don't have any unread emails.",
                data={"emails": []},
            )

        # Format for display
        lines = [f"Found {len(emails)} unread email{'s' if len(emails) > 1 else ''}:\n"]

        for i, email in enumerate(emails, 1):
            lines.append(f"{i}. {email['subject']}")
            lines.append(f"   From: {email['sender']}")
            lines.append(f"   Date: {email['date']}")
            lines.append("")

        lines.append("(Emails stay on your device - I can summarize specific ones if you'd like)")

        return SkillResult(
            success=True,
            response_text="\n".join(lines),
            data={"emails": emails},
        )
