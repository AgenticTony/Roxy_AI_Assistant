"""Unit tests for WebSearchSkill.

Tests web search functionality with mocks and fake data.
No external API calls are made.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.brain.privacy import PrivacyGateway
from roxy.skills.base import Permission, SkillContext, StubMemoryManager
from roxy.skills.web.search import WebSearchSkill


@pytest.fixture
def privacy_gateway():
    """Fixture providing a PrivacyGateway instance."""
    return PrivacyGateway()


@pytest.fixture
def web_search_skill(privacy_gateway):
    """Fixture providing a WebSearchSkill instance."""
    return WebSearchSkill(privacy_gateway=privacy_gateway)


@pytest.fixture
def skill_context():
    """Fixture providing a SkillContext for testing."""
    return SkillContext(
        user_input="search for python programming",
        intent="web_search",
        parameters={},
        memory=StubMemoryManager(),
        config=MagicMock(),
        conversation_history=[],
    )


class TestWebSearchSkill:
    """Tests for WebSearchSkill."""

    def test_init(self, privacy_gateway):
        """Test skill initialization."""
        skill = WebSearchSkill(privacy_gateway=privacy_gateway)

        assert skill.name == "web_search"
        assert len(skill.permissions) > 0
        assert Permission.NETWORK in skill.permissions
        assert len(skill.triggers) > 0
        assert "search for" in skill.triggers

    def test_cache_key_generation(self, web_search_skill):
        """Test cache key generation."""
        key1 = web_search_skill._get_cache_key("Python programming")
        key2 = web_search_skill._get_cache_key("python programming")
        key3 = web_search_skill._get_cache_key("Different query")

        # Keys should be case-insensitive and trimmed
        assert key1 == key2
        assert key1 != key3
        assert key1 == "search:python programming"

    @pytest.mark.asyncio
    async def test_rate_limit_check(self, web_search_skill):
        """Test rate limiting functionality."""
        # Should allow first request
        assert await web_search_skill._check_rate_limit() is True

        # Add many requests
        for _ in range(15):
            await web_search_skill._check_rate_limit()

        # Now should be limited
        assert await web_search_skill._check_rate_limit() is False

    @pytest.mark.asyncio
    async def test_searxng_availability_check(self, web_search_skill):
        """Test SearXNG availability check."""
        # Mock subprocess to simulate SearXNG not running
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="404")

            available = await web_search_skill._check_searxng_available()
            assert available is False

    @pytest.mark.asyncio
    async def test_search_query_extraction(self, web_search_skill, skill_context):
        """Test that search queries are extracted correctly."""
        skill_context.user_input = "search for the best pizza places"
        skill_context.parameters = {}

        # Mock the search methods
        web_search_skill._search_brave = AsyncMock(return_value="Found pizza places")
        web_search_skill.privacy_gateway = MagicMock()
        web_search_skill.privacy_gateway.redact = MagicMock(
            return_value=MagicMock(was_redacted=False, redacted_text="the best pizza places")
        )

        result = await web_search_skill.execute(skill_context)

        assert result.success is True
        assert "pizza" in result.response_text.lower() or "Found" in result.response_text

    @pytest.mark.asyncio
    async def test_pii_redaction_before_search(
        self, web_search_skill, skill_context, privacy_gateway
    ):
        """Test that PII is redacted before external searches."""
        # Input with email
        skill_context.user_input = "search for my emails at anthony@example.com"

        # Mock privacy gateway
        privacy_gateway.redact = MagicMock(
            return_value=MagicMock(
                was_redacted=True, redacted_text="search for my emails at [REDACTED_EMAIL_1]"
            )
        )
        web_search_skill.privacy_gateway = privacy_gateway

        # Mock search method
        web_search_skill._search_brave = AsyncMock(return_value="Search results")
        web_search_skill._check_rate_limit = AsyncMock(return_value=True)

        result = await web_search_skill.execute(skill_context)

        # Verify redaction was called
        privacy_gateway.redact.assert_called_once()
        assert result.success is True


class TestCachedResult:
    """Tests for CachedResult dataclass."""

    def test_cache_expiry(self):
        """Test that cached results expire correctly."""
        from roxy.skills.web.search import CachedResult

        # Create a cached result with short TTL
        cached = CachedResult(
            query="test",
            result="result",
            timestamp=datetime.now(),
            ttl=timedelta(seconds=1),
        )

        # Should not be expired immediately
        assert cached.is_expired() is False

        # Should be expired after TTL
        with patch("roxy.skills.web.search.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(seconds=2)
            assert cached.is_expired() is True
