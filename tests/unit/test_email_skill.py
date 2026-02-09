"""Unit tests for email skill."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.skills.base import Permission, SkillContext, StubMemoryManager
from roxy.skills.productivity.email import EmailSkill
from roxy.brain.llm_clients import LLMResponse


# Fixtures
@pytest.fixture
def skill_context():
    """Create a skill context for testing."""
    from roxy.config import LocalLLMConfig, CloudLLMConfig, PrivacyConfig, RoxyConfig

    config = RoxyConfig(
        name="TestRoxy",
        data_dir="/tmp/test_roxy",
        llm_local=LocalLLMConfig(),
        llm_cloud=CloudLLMConfig(),
        privacy=PrivacyConfig(),
    )

    # Create mock LLM client
    mock_llm_client = MagicMock()
    mock_llm_client.generate = AsyncMock(return_value=LLMResponse(
        content="This email from John Smith discusses the project timeline and requests a meeting by Friday to review deliverables.",
        model="qwen3:8b",
        provider="local",
    ))

    return SkillContext(
        user_input="test input",
        intent="test",
        parameters={},
        memory=StubMemoryManager(),
        config=config,
        conversation_history=[],
        local_llm_client=mock_llm_client,
    )


@pytest.fixture
def sample_email():
    """Return a sample email."""
    return {
        "subject": "Project Timeline Review",
        "sender": "John Smith <john.smith@example.com>",
        "date": "Mon, 9 Feb 2026 10:30:00 +0000",
        "message_id": "test-message-id-123",
        "body": """Hi Team,

I wanted to follow up on the project timeline. We need to review the current deliverables and make sure we're on track for the Q1 release.

Key action items:
- Review API documentation by Wednesday
- Complete testing module by Thursday
- Schedule final review meeting for Friday

Please let me know your availability for the meeting.

Best regards,
John"""
    }


class TestEmailSkill:
    """Test EmailSkill."""

    def test_skill_metadata(self):
        """Test skill has correct metadata."""
        skill = EmailSkill()

        assert skill.name == "email"
        assert "check email" in skill.triggers
        assert Permission.APPLESCRIPT in skill.permissions
        assert Permission.EMAIL in skill.permissions
        assert skill.requires_cloud is False  # Email stays local

    @pytest.mark.asyncio
    async def test_summarize_email_with_llm(self, skill_context, sample_email):
        """Test email summarization using LLM."""
        skill = EmailSkill()

        # Mock get_email_content
        with patch.object(skill, "get_email_content", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_email

            # Generate summary
            summary = await skill.summarize_email(skill_context, "test-message-id")

            # Verify LLM was called
            assert skill_context.local_llm_client.generate.called

            # Verify summary contains key info
            assert "From:" in summary
            assert sample_email["sender"] in summary
            assert "Subject:" in summary
            assert sample_email["subject"] in summary

    @pytest.mark.asyncio
    async def test_summarize_email_fallback(self, skill_context, sample_email):
        """Test email summarization fallback when LLM fails."""
        skill = EmailSkill()

        # Mock get_email_content
        with patch.object(skill, "get_email_content", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_email

            # Mock LLM failure
            skill_context.local_llm_client.generate = AsyncMock(side_effect=Exception("LLM failed"))

            # Generate summary
            summary = await skill.summarize_email(skill_context, "test-message-id")

            # Verify we got a fallback format
            assert "From:" in summary
            assert sample_email["sender"] in summary
            assert "Subject:" in summary
            assert sample_email["subject"] in summary

    @pytest.mark.asyncio
    async def test_summarize_email_no_content(self, skill_context):
        """Test email summarization with no email content."""
        skill = EmailSkill()

        # Mock get_email_content to return empty
        with patch.object(skill, "get_email_content", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {}

            # Generate summary
            summary = await skill.summarize_email(skill_context, "test-message-id")

            # Verify error message
            assert "Could not retrieve" in summary

    @pytest.mark.asyncio
    async def test_execute_read_command(self, skill_context):
        """Test execute method with read command."""
        skill = EmailSkill()
        skill_context.user_input = "read my emails"

        # Mock get_latest_emails
        with patch.object(skill, "get_latest_emails", new_callable=AsyncMock) as mock_emails:
            mock_emails.return_value = [
                {
                    "subject": "Test Email",
                    "sender": "test@example.com",
                    "date": "Mon, 9 Feb 2026 10:30:00 +0000",
                    "message_id": "test-123",
                }
            ]

            result = await skill.execute(skill_context)

            # Verify result structure
            assert result.success is True
            assert isinstance(result.response_text, str)

    @pytest.mark.asyncio
    async def test_execute_summarize_command(self, skill_context):
        """Test execute method with summarize command."""
        skill = EmailSkill()
        skill_context.user_input = "summarize my latest email"

        # Mock get_latest_emails and summarize_email
        with patch.object(skill, "get_latest_emails", new_callable=AsyncMock) as mock_emails, \
             patch.object(skill, "summarize_email", new_callable=AsyncMock) as mock_summarize:
            mock_emails.return_value = [
                {
                    "subject": "Test Email",
                    "sender": "test@example.com",
                    "date": "Mon, 9 Feb 2026 10:30:00 +0000",
                    "message_id": "test-123",
                }
            ]
            mock_summarize.return_value = "From: test@example.com\nSubject: Test Email\n\nTest summary"

            result = await skill.execute(skill_context)

            # Verify result structure
            assert result.success is True
            assert "test@example.com" in result.response_text

    @pytest.mark.asyncio
    async def test_can_handle_confidence_scoring(self):
        """Test can_handle method returns appropriate confidence scores."""
        skill = EmailSkill()

        # High confidence for exact match
        confidence = skill.can_handle("read my emails", {})
        assert confidence > 0.5

        # Lower confidence for partial match
        confidence = skill.can_handle("check my inbox", {})
        assert confidence > 0.0

        # No confidence for unrelated input
        confidence = skill.can_handle("what's the weather", {})
        assert confidence == 0.0
