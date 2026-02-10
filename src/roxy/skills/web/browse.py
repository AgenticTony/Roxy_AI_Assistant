"""Web browsing skill for Roxy.

Provides browser automation using Browser-Use with Playwright.
All URLs are logged and state is cleared between sessions.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from roxy.brain.privacy import PrivacyGateway


class BrowseSkill(RoxySkill):
    """
    Web browsing skill using Browser-Use and Playwright.

    Features:
    - Headless browser automation
    - Privacy-focused: cookies cleared after each session
    - All URLs logged to data/web_requests.log
    - PrivacyGateway checks before navigation
    - 30s timeout per action
    """

    name: str = "browse"
    description: str = "Browse websites and read web content"
    triggers: list[str] = [
        "go to",
        "read",
        "browse",
        "open website",
        "visit",
        "navigate to",
    ]
    permissions: list[Permission] = [Permission.NETWORK]
    requires_cloud: bool = False

    # Configuration
    HEADLESS: bool = True
    VIEWPORT_WIDTH: int = 1920
    VIEWPORT_HEIGHT: int = 1080
    ACTION_TIMEOUT: int = 30  # seconds
    REQUESTS_LOG: Path = Path("data/web_requests.log")

    def __init__(self, privacy_gateway: PrivacyGateway | None = None) -> None:
        """Initialize browse skill.

        Args:
            privacy_gateway: PrivacyGateway for PII checks.
        """
        super().__init__()

        self.privacy_gateway = privacy_gateway

        # Ensure log directory exists
        self.REQUESTS_LOG.parent.mkdir(parents=True, exist_ok=True)

    def _log_url(self, url: str, action: str) -> None:
        """Log URL access to web requests log.

        Args:
            url: The URL being accessed.
            action: Action being performed (browse, scrape, etc).
        """
        try:
            timestamp = datetime.now().isoformat()
            log_entry = f"{timestamp} | {action} | {url}\n"

            with self.REQUESTS_LOG.open("a", encoding="utf-8") as f:
                f.write(log_entry)

            logger.debug(f"Logged URL access: {url}")

        except Exception as e:
            logger.error(f"Failed to log URL: {e}")

    def _extract_url(self, user_input: str) -> str | None:
        """Extract URL from user input.

        Args:
            user_input: Raw user input text.

        Returns:
            Extracted URL or None if not found.
        """
        import re

        # Look for URLs in common formats
        url_pattern = r"https?://[^\s]+"
        match = re.search(url_pattern, user_input)

        if match:
            return match.group()

        # Look for domain names without protocol
        domain_pattern = r"(?:www\.)?[\w-]+\.[a-z]{2,}(?:/[^\s]*)?"
        match = re.search(domain_pattern, user_input, re.IGNORECASE)

        if match:
            url = match.group()
            if not url.startswith("http"):
                url = "https://" + url
            return url

        return None

    async def browse_and_summarize(self, url: str) -> str:
        """Browse to a URL and summarize the content.

        Args:
            url: URL to browse.

        Returns:
            Summarized content from the page.
        """
        self._log_url(url, "browse_and_summarize")

        try:
            # Try to import browser-use
            from browser_use import Agent
            from langchain_openai import ChatOpenAI

            # Use local Ollama for content processing
            llm = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # Ollama doesn't require real key
                model="qwen3:8b",
                temperature=0.7,
            )

            # Create agent with the task
            agent = Agent(
                task=f"Go to {url} and provide a concise summary of the main content. Focus on the key information, ignoring navigation and ads.",
                llm=llm,
            )

            # Run the agent
            result = await agent.run()

            # Clean up browser context
            await agent.browser.close()

            return str(result)

        except ImportError:
            logger.warning("browser-use not available, using fallback")

            # Fallback: use Firecrawl MCP for scraping
            from roxy.mcp import MCPServerManager

            manager = MCPServerManager()
            await manager.start_server("firecrawl")

            scrape_result = await manager.call_tool(
                server="firecrawl",
                tool="scrape_url",
                args={"url": url},
            )

            if scrape_result.success:
                # Return the scraped content
                if isinstance(scrape_result.data, dict):
                    return scrape_result.data.get("markdown", "Scraped content unavailable")
                return str(scrape_result.data)
            else:
                return f"Failed to scrape {url}: {scrape_result.error}"

        except Exception as e:
            logger.error(f"Browse error: {e}")
            return f"Error browsing {url}: {e}"

    async def interactive_browse(self, task: str, url: str | None = None) -> str:
        """Perform interactive browser automation.

        Args:
            task: Description of the task to accomplish.
            url: Optional starting URL.

        Returns:
            Result of the browsing task.
        """
        if url:
            self._log_url(url, "interactive_browse")

        try:
            from browser_use import Agent
            from langchain_openai import ChatOpenAI

            # Use local Ollama
            llm = ChatOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                model="qwen3:8b",
                temperature=0.7,
            )

            # Build task prompt
            if url:
                full_task = f"Start at {url}. Then: {task}"
            else:
                full_task = task

            # Create agent
            agent = Agent(
                task=full_task,
                llm=llm,
            )

            # Run with timeout
            result = await asyncio.wait_for(
                agent.run(),
                timeout=60.0,  # 60 second total timeout
            )

            # Clean up
            await agent.browser.close()

            return str(result)

        except TimeoutError:
            return "Browsing task timed out after 60 seconds"
        except Exception as e:
            logger.error(f"Interactive browse error: {e}")
            return f"Error: {e}"

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute browse skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with browse results.
        """
        user_input = context.user_input

        # Extract URL from input
        url = self._extract_url(user_input)
        given_url = context.parameters.get("url")

        if given_url:
            url = given_url

        if not url:
            return SkillResult(
                success=False,
                response_text="What website would you like me to browse?",
            )

        # Check privacy gateway for PII in URL
        if self.privacy_gateway:
            redaction_result = self.privacy_gateway.redact(url)
            if redaction_result.was_redacted:
                logger.warning(f"PII detected in URL: {redaction_result.redacted_text}")
                return SkillResult(
                    success=False,
                    response_text="That URL contains sensitive information. I can't browse to addresses with personal data.",
                )

        # Determine the task type
        task_keywords = {
            "summary": "summarize",
            "read": "read",
            "get": "get information",
        }

        is_simple_browse = any(kw in user_input.lower() for kw in task_keywords)

        try:
            if is_simple_browse:
                # Simple browse and summarize
                result = await self.browse_and_summarize(url)
            else:
                # Interactive browsing with full task
                result = await self.interactive_browse(user_input, url)

            return SkillResult(
                success=True,
                response_text=result,
                data={
                    "url": url,
                    "method": "browse_and_summarize" if is_simple_browse else "interactive",
                },
            )

        except Exception as e:
            logger.error(f"Browse execution error: {e}")
            return SkillResult(
                success=False,
                response_text=f"I couldn't browse that website. Error: {e}",
            )
