"""Web search skill for Roxy.

Provides web search capability using Brave Search MCP with SearXNG fallback.
All queries are passed through PrivacyGateway for PII redaction.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from roxy.brain.privacy import PrivacyGateway


@dataclass
class CachedResult:
    """A cached search result with expiry."""

    query: str
    result: str
    timestamp: datetime
    ttl: timedelta  # Time-to-live for this cache entry

    def is_expired(self) -> bool:
        """Check if this cached result has expired."""
        return datetime.now() > self.timestamp + self.ttl


class WebSearchSkill(RoxySkill):
    """
    Web search skill using Brave Search API via MCP.

    Features:
    - Privacy-first: All queries go through PrivacyGateway
    - Caching with configurable TTLs
    - Rate limiting (10 requests/minute)
    - SearXNG fallback for local search
    """

    name: str = "web_search"
    description: str = "Search the web for information"
    triggers: list[str] = [
        "search for",
        "look up",
        "google",
        "find on the web",
        "web search",
        "search the internet",
    ]
    permissions: list[Permission] = [Permission.NETWORK]
    requires_cloud: bool = False

    # Cache configuration
    CACHE_TTL_NEWS: timedelta = timedelta(hours=1)
    CACHE_TTL_STATIC: timedelta = timedelta(weeks=1)

    # Rate limiting
    RATE_LIMIT_MAX: int = 10  # requests per minute
    RATE_LIMIT_WINDOW: timedelta = timedelta(minutes=1)

    def __init__(self, privacy_gateway: PrivacyGateway | None = None) -> None:
        """Initialize web search skill.

        Args:
            privacy_gateway: PrivacyGateway for PII redaction.
        """
        super().__init__()

        self.privacy_gateway = privacy_gateway

        # Cache storage
        self._cache: dict[str, CachedResult] = {}

        # Rate limiting: list of timestamps for recent requests
        self._request_times: list[datetime] = []

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for a query."""
        return f"search:{query.lower().strip()}"

    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits.

        Returns:
            True if request is allowed.
        """
        now = datetime.now()

        # Remove timestamps outside the rate limit window
        self._request_times = [
            ts for ts in self._request_times
            if now - ts < self.RATE_LIMIT_WINDOW
        ]

        if len(self._request_times) >= self.RATE_LIMIT_MAX:
            logger.warning(f"Rate limit exceeded: {len(self._request_times)} requests")
            return False

        self._request_times.append(now)
        return True

    async def _check_searxng_available(self) -> bool:
        """Check if SearXNG Docker container is running.

        Returns:
            True if SearXNG is available at localhost:8888.
        """
        try:
            result = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:8888/health"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            is_healthy = result.stdout.strip() == "200"
            if is_healthy:
                logger.debug("SearXNG is available")
            else:
                logger.debug(f"SearXNG health check returned: {result.stdout.strip()}")

            return is_healthy

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"SearXNG not available: {e}")
            return False

    async def _search_brave(self, query: str) -> str:
        """Perform search using Brave Search MCP.

        Args:
            query: Search query (already redacted).

        Returns:
            Search results as formatted text.
        """
        try:
            # Import here to avoid circular dependency
            from roxy.mcp import MCPServerManager

            manager = MCPServerManager()
            await manager.start_server("brave_search")

            result = await manager.call_tool(
                server="brave_search",
                tool="brave_web_search",
                args={"query": query, "count": 5},
            )

            if result.success and result.data:
                # Format results from Brave Search
                return self._format_search_results(result.data)
            else:
                logger.error(f"Brave search failed: {result.error}")
                return f"Search failed: {result.error}"

        except Exception as e:
            logger.error(f"Error in Brave search: {e}")
            return f"Search error: {e}"

    async def _search_searxng(self, query: str) -> str:
        """Perform search using local SearXNG instance.

        Args:
            query: Search query (already redacted).

        Returns:
            Search results as formatted text.
        """
        try:
            # Query SearXNG JSON API
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:8888/search?q={query}&format=json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return "SearXNG search failed"

            # Parse JSON results
            import json

            data = json.loads(result.stdout)

            # Format results
            return self._format_searxng_results(data)

        except Exception as e:
            logger.error(f"Error in SearXNG search: {e}")
            return f"SearXNG search error: {e}"

    def _format_search_results(self, data: dict) -> str:
        """Format search results from Brave or similar API.

        Args:
            data: Raw search results data.

        Returns:
            Formatted text for user.
        """
        # Placeholder format - will be adjusted based on actual API response
        if "results" in data:
            results = data["results"][:5]
            formatted = []

            for i, result in enumerate(results, 1):
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                snippet = result.get("snippet", "")

                formatted.append(f"{i}. {title}")
                formatted.append(f"   {url}")
                if snippet:
                    formatted.append(f"   {snippet[:200]}...")
                formatted.append("")

            return "\n".join(formatted)

        return "No results found"

    def _format_searxng_results(self, data: dict) -> str:
        """Format SearXNG search results.

        Args:
            data: Raw SearXNG JSON response.

        Returns:
            Formatted text for user.
        """
        results = data.get("results", [])[:5]
        formatted = []

        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")

            formatted.append(f"{i}. {title}")
            formatted.append(f"   {url}")
            if content:
                formatted.append(f"   {content[:200]}...")
            formatted.append("")

        return "\n".join(formatted) if formatted else "No results found"

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute web search.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult with search results.
        """
        user_input = context.user_input
        query = context.parameters.get("query", user_input)

        # Extract search query from common trigger phrases
        for trigger in self.triggers:
            if trigger in user_input.lower():
                # Remove trigger phrase to get clean query
                query = user_input.lower().replace(trigger, "").strip()
                # Remove common leading words
                for word in ["for", "up", "the", "on"]:
                    if query.startswith(word + " "):
                        query = query[len(word + " "):].strip()
                break

        if not query:
            return SkillResult(
                success=False,
                response_text="What would you like me to search for?",
            )

        # Check rate limit
        if not await self._check_rate_limit():
            return SkillResult(
                success=False,
                response_text="I'm doing too many searches right now. Please wait a moment.",
            )

        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if not cached.is_expired():
                logger.info(f"Cache hit for query: {query}")
                return SkillResult(
                    success=True,
                    response_text=cached.result,
                    data={"cached": True, "query": query},
                )

        # Redact PII from query before sending externally
        redacted_query = query
        if self.privacy_gateway:
            redaction_result = self.privacy_gateway.redact(query)
            if redaction_result.was_redacted:
                logger.info(f"Redacted PII from search query: {redaction_result.redacted_text}")
                redacted_query = redaction_result.redacted_text

        # Try Brave Search first
        logger.info(f"Searching for: {redacted_query}")

        try:
            search_results = await self._search_brave(redacted_query)

            # If Brave fails, try SearXNG fallback
            if "failed" in search_results.lower() or "error" in search_results.lower():
                if await self._check_searxng_available():
                    logger.info("Brave search failed, falling back to SearXNG")
                    search_results = await self._search_searxng(redacted_query)

            # Determine TTL based on query type
            ttl = self.CACHE_TTL_STATIC
            news_keywords = ["news", "today", "latest", "breaking", "recent"]
            if any(kw in query.lower() for kw in news_keywords):
                ttl = self.CACHE_TTL_NEWS

            # Cache the result
            self._cache[cache_key] = CachedResult(
                query=query,
                result=search_results,
                timestamp=datetime.now(),
                ttl=ttl,
            )

            # Limit cache size
            if len(self._cache) > 100:
                # Remove oldest entries
                sorted_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].timestamp,
                )
                for old_key in sorted_keys[:20]:
                    del self._cache[old_key]

            return SkillResult(
                success=True,
                response_text=search_results,
                data={
                    "query": query,
                    "redacted_query": redacted_query,
                    "cached": False,
                },
            )

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't complete that search. Error: {e}",
            )
