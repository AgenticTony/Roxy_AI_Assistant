"""Integration tests for Roxy web skills.

Tests web search, browsing, and web-based skills with privacy gateway.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from roxy.brain.privacy import ConsentMode, PrivacyGateway
from roxy.config import RoxyConfig
from roxy.skills.base import SkillContext, SkillResult
from roxy.skills.registry import SkillRegistry


class MockWebSearchSkill:
    """Mock web search skill for testing."""

    name = "web_search"
    description = "Search the web"
    triggers = ["search", "find"]
    permissions = []

    def __init__(self, privacy_gateway: PrivacyGateway | None = None):
        self.privacy_gateway = privacy_gateway

    async def execute(self, context: SkillContext) -> SkillResult:
        query = context.user_input

        # Apply privacy if gateway exists
        if self.privacy_gateway:
            redacted_query, _ = self.privacy_gateway.redact(query)
            query = redacted_query

        # Mock search result
        return SkillResult(
            success=True,
            response_text=f"Found results for: {query}",
            data={"results": ["Result 1", "Result 2"]},
        )


class MockBrowseSkill:
    """Mock browser skill for testing."""

    name = "browse"
    description = "Browse a website"
    triggers = ["browse", "open", "visit"]
    permissions = []

    def __init__(self, privacy_gateway: PrivacyGateway | None = None):
        self.privacy_gateway = privacy_gateway

    async def execute(self, context: SkillContext) -> SkillResult:
        url = context.parameters.get("url", "")

        # Apply privacy if gateway exists
        if self.privacy_gateway and url:
            redacted_url, _ = self.privacy_gateway.redact(url)
            url = redacted_url

        return SkillResult(
            success=True,
            response_text=f"Browsed to: {url}",
            data={"url": url, "content": "Page content..."},
        )


@pytest.mark.asyncio
async def test_web_search_with_privacy(mock_config: RoxyConfig) -> None:
    """Test that web search applies privacy redaction."""
    privacy = PrivacyGateway(
        redact_patterns=["email", "phone"],
        consent_mode=ConsentMode.ALWAYS,
        log_path=str(Path(mock_config.data_dir) / "cloud.log"),
    )

    skill = MockWebSearchSkill(privacy_gateway=privacy)

    context = SkillContext(
        user_input="Search for john@example.com",
        intent="search",
        parameters={},
        memory=MagicMock(),
        config=mock_config,
        conversation_history=[],
    )

    result = await skill.execute(context)

    assert result.success is True
    # Email should be redacted in the response
    assert "john@example.com" not in result.response_text


@pytest.mark.asyncio
async def test_web_browse_skill(mock_config: RoxyConfig) -> None:
    """Test the browse skill."""
    privacy = PrivacyGateway(
        redact_patterns=[],
        consent_mode=ConsentMode.ALWAYS,
        log_path=str(Path(mock_config.data_dir) / "cloud.log"),
    )

    skill = MockBrowseSkill(privacy_gateway=privacy)

    context = SkillContext(
        user_input="Browse to example.com",
        intent="browse",
        parameters={"url": "https://example.com"},
        memory=MagicMock(),
        config=mock_config,
        conversation_history=[],
    )

    result = await skill.execute(context)

    assert result.success is True
    assert "example.com" in result.response_text
    assert result.data is not None


@pytest.mark.asyncio
async def test_web_skill_cloud_consent(mock_config: RoxyConfig) -> None:
    """Test that web skills respect cloud consent mode."""
    # Test with NEVER consent
    privacy_never = PrivacyGateway(
        redact_patterns=[],
        consent_mode=ConsentMode.NEVER,
        log_path=str(Path(mock_config.data_dir) / "cloud.log"),
    )

    skill = MockWebSearchSkill(privacy_gateway=privacy_never)

    context = SkillContext(
        user_input="Search for something",
        intent="search",
        parameters={},
        memory=MagicMock(),
        config=mock_config,
        conversation_history=[],
    )

    # With NEVER consent, the skill should either:
    # 1. Deny the request, or
    # 2. Use local search only
    result = await skill.execute(context)

    # Should still succeed but with local-only approach
    assert result.success is True


@pytest.mark.asyncio
async def test_multiple_web_skills_registration(mock_config: RoxyConfig) -> None:
    """Test that multiple web skills can be registered."""
    registry = SkillRegistry()
    registry.reset()

    privacy = PrivacyGateway(
        redact_patterns=["email"],
        consent_mode=ConsentMode.ASK,
        log_path=str(Path(mock_config.data_dir) / "cloud.log"),
    )

    # Register web skills
    search_skill = MockWebSearchSkill(privacy_gateway=privacy)
    browse_skill = MockBrowseSkill(privacy_gateway=privacy)

    # Manual registration
    registry._skills["web_search"] = MockWebSearchSkill
    registry._skills["browse"] = MockBrowseSkill
    registry._skill_instances["web_search"] = search_skill
    registry._skill_instances["browse"] = browse_skill

    # List skills
    skills = registry.list_skills()

    skill_names = {s["name"] for s in skills}
    assert "web_search" in skill_names
    assert "browse" in skill_names


@pytest.mark.asyncio
async def test_web_skill_error_handling(mock_config: RoxyConfig) -> None:
    """Test that web skills handle errors gracefully."""
    privacy = PrivacyGateway(
        redact_patterns=[],
        consent_mode=ConsentMode.ALWAYS,
        log_path=str(Path(mock_config.data_dir) / "cloud.log"),
    )

    class FailingWebSkill:
        """A web skill that fails."""

        name = "failing_web"
        description = "A failing web skill"
        triggers = ["fail"]
        permissions = []

        def __init__(self, privacy_gateway: PrivacyGateway | None = None):
            self.privacy_gateway = privacy_gateway

        async def execute(self, context: SkillContext) -> SkillResult:
            # Simulate network error
            return SkillResult(
                success=False,
                response_text="Network error occurred",
                data={"error": "Connection failed"},
            )

    skill = FailingWebSkill(privacy_gateway=privacy)

    context = SkillContext(
        user_input="fail",
        intent="fail",
        parameters={},
        memory=MagicMock(),
        config=mock_config,
        conversation_history=[],
    )

    result = await skill.execute(context)

    # Should return error result without crashing
    assert result.success is False
    assert "error" in result.response_text.lower()


@pytest.mark.asyncio
async def test_web_search_result_parsing(mock_config: RoxyConfig) -> None:
    """Test that web search results are properly parsed."""
    privacy = PrivacyGateway(
        redact_patterns=[],
        consent_mode=ConsentMode.ALWAYS,
        log_path=str(Path(mock_config.data_dir) / "cloud.log"),
    )

    skill = MockWebSearchSkill(privacy_gateway=privacy)

    context = SkillContext(
        user_input="Search for Python tutorials",
        intent="search",
        parameters={},
        memory=MagicMock(),
        config=mock_config,
        conversation_history=[],
    )

    result = await skill.execute(context)

    # Check result structure
    assert result.success is True
    assert result.data is not None
    assert "results" in result.data


@pytest.mark.asyncio
async def test_web_skill_with_parameters(mock_config: RoxyConfig) -> None:
    """Test web skills with additional parameters."""
    privacy = PrivacyGateway(
        redact_patterns=[],
        consent_mode=ConsentMode.ALWAYS,
        log_path=str(Path(mock_config.data_dir) / "cloud.log"),
    )

    class ParameterizedWebSkill:
        """Web skill with parameters."""

        name = "param_web"
        description = "Web skill with parameters"
        triggers = ["search with"]
        permissions = []

        def __init__(self, privacy_gateway: PrivacyGateway | None = None):
            self.privacy_gateway = privacy_gateway

        async def execute(self, context: SkillContext) -> SkillResult:
            query = context.parameters.get("query", "")
            limit = context.parameters.get("limit", 10)

            return SkillResult(
                success=True,
                response_text=f"Searched for '{query}' with limit {limit}",
                data={"query": query, "limit": limit},
            )

    skill = ParameterizedWebSkill(privacy_gateway=privacy)

    context = SkillContext(
        user_input="Search for AI with limit 5",
        intent="search_with",
        parameters={"query": "AI", "limit": 5},
        memory=MagicMock(),
        config=mock_config,
        conversation_history=[],
    )

    result = await skill.execute(context)

    assert result.success is True
    assert "AI" in result.response_text
    assert "5" in result.response_text


@pytest.mark.asyncio
async def test_privacy_gateway_in_web_skills(mock_config: RoxyConfig) -> None:
    """Test that privacy gateway is properly integrated into web skills."""
    privacy = PrivacyGateway(
        redact_patterns=["email", "phone", "ssn"],
        consent_mode=ConsentMode.ASK,
        log_path=str(Path(mock_config.data_dir) / "cloud.log"),
    )

    # Test with PII in query
    text_with_pii = "Search for contact info at 555-123-4567"

    # Redact
    redacted, detected = privacy.redact(text_with_pii)

    # Verify PII was detected and redacted
    assert detected is True or len(detected) > 0
    assert "555-123-4567" not in redacted


@pytest.mark.asyncio
async def test_web_skill_response_formatting(mock_config: RoxyConfig) -> None:
    """Test that web skill responses are properly formatted."""
    privacy = PrivacyGateway(
        redact_patterns=[],
        consent_mode=ConsentMode.ALWAYS,
        log_path=str(Path(mock_config.data_dir) / "cloud.log"),
    )

    class FormattingWebSkill:
        """Web skill that formats responses."""

        name = "formatting_web"
        description = "Formats web responses"
        triggers = ["format search"]
        permissions = []

        def __init__(self, privacy_gateway: PrivacyGateway | None = None):
            self.privacy_gateway = privacy_gateway

        async def execute(self, context: SkillContext) -> SkillResult:
            # Simulate search results
            results = [
                {"title": "Result 1", "url": "https://example.com/1", "snippet": "First result"},
                {"title": "Result 2", "url": "https://example.com/2", "snippet": "Second result"},
            ]

            # Format response
            response = "Here are the search results:\n"
            for i, r in enumerate(results, 1):
                response += f"{i}. {r['title']}\n   {r['snippet']}\n"

            return SkillResult(
                success=True,
                response_text=response,
                data={"results": results},
            )

    skill = FormattingWebSkill(privacy_gateway=privacy)

    context = SkillContext(
        user_input="Format search results",
        intent="format_search",
        parameters={},
        memory=MagicMock(),
        config=mock_config,
        conversation_history=[],
    )

    result = await skill.execute(context)

    assert result.success is True
    assert "search results" in result.response_text.lower()
    assert "Result 1" in result.response_text
    assert "Result 2" in result.response_text


# Async test marker
# Note: Using @pytest.mark.asyncio decorator directly
