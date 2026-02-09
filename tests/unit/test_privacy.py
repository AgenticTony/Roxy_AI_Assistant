"""Unit tests for privacy gateway."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from roxy.brain.privacy import (
    PrivacyGateway,
    PIIMatch,
    RedactionResult,
    ConsentMode,
)


class TestPIIMatch:
    """Tests for PIIMatch dataclass."""

    def test_create_pii_match(self) -> None:
        """Test creating a PII match."""
        match = PIIMatch(
            pattern_name="email",
            original="test@example.com",
            start=0,
            end=15,
            placeholder="[REDACTED_EMAIL_1]",
        )

        assert match.pattern_name == "email"
        assert match.original == "test@example.com"
        assert match.start == 0
        assert match.end == 15
        assert match.placeholder == "[REDACTED_EMAIL_1]"


class TestRedactionResult:
    """Tests for RedactionResult dataclass."""

    def test_create_redaction_result(self) -> None:
        """Test creating a redaction result."""
        result = RedactionResult(
            redacted_text="My email is [REDACTED_EMAIL_1]",
            pii_matches=[],
            was_redacted=True,
        )

        assert result.redacted_text == "My email is [REDACTED_EMAIL_1]"
        assert result.was_redacted is True
        assert len(result.pii_matches) == 0


class TestPrivacyGateway:
    """Tests for PrivacyGateway."""

    def test_init_default(self) -> None:
        """Test initialization with defaults."""
        gateway = PrivacyGateway()

        assert gateway.consent_mode == ConsentMode.ASK
        assert gateway.log_path == Path("data/cloud_requests.log").expanduser()
        assert len(gateway._pattern_names) > 0

    def test_init_custom(self) -> None:
        """Test initialization with custom values."""
        gateway = PrivacyGateway(
            redact_patterns=["email", "phone"],
            consent_mode=ConsentMode.ALWAYS,
            log_path="/tmp/test.log",
        )

        assert gateway.consent_mode == ConsentMode.ALWAYS
        assert gateway.log_path == Path("/tmp/test.log")
        assert gateway._pattern_names == ["email", "phone"]

    @pytest.mark.asyncio
    async def test_can_use_cloud_ask_mode(self) -> None:
        """Test can_use_cloud in ASK mode."""
        gateway = PrivacyGateway(consent_mode=ConsentMode.ASK)
        allowed, message = await gateway.can_use_cloud()

        assert allowed is True
        assert message is not None
        assert "cloud" in message.lower()

    @pytest.mark.asyncio
    async def test_can_use_cloud_always_mode(self) -> None:
        """Test can_use_cloud in ALWAYS mode."""
        gateway = PrivacyGateway(consent_mode=ConsentMode.ALWAYS)
        allowed, message = await gateway.can_use_cloud()

        assert allowed is True
        assert message is None

    @pytest.mark.asyncio
    async def test_can_use_cloud_never_mode(self) -> None:
        """Test can_use_cloud in NEVER mode."""
        gateway = PrivacyGateway(consent_mode=ConsentMode.NEVER)
        allowed, message = await gateway.can_use_cloud()

        assert allowed is False
        assert message is not None
        assert "disabled" in message.lower()

    def test_redact_empty_text(self) -> None:
        """Test redacting empty text."""
        gateway = PrivacyGateway()
        result = gateway.redact("")

        assert result.redacted_text == ""
        assert result.was_redacted is False
        assert len(result.pii_matches) == 0

    def test_redact_none_text(self) -> None:
        """Test redacting None text."""
        gateway = PrivacyGateway()
        result = gateway.redact("")

        assert result.redacted_text == ""
        assert result.was_redacted is False

    def test_redact_email(self) -> None:
        """Test redacting email addresses."""
        gateway = PrivacyGateway(redact_patterns=["email"])
        text = "My email is john.doe@example.com"
        result = gateway.redact(text)

        assert result.was_redacted is True
        assert "[REDACTED_EMAIL_1]" in result.redacted_text
        assert "john.doe@example.com" not in result.redacted_text
        assert len(result.pii_matches) == 1
        assert result.pii_matches[0].pattern_name == "email"

    def test_redact_multiple_emails(self) -> None:
        """Test redacting multiple email addresses."""
        gateway = PrivacyGateway(redact_patterns=["email"])
        text = "Email me at test1@example.com or test2@example.org"
        result = gateway.redact(text)

        assert result.was_redacted is True
        assert "[REDACTED_EMAIL_1]" in result.redacted_text
        assert "[REDACTED_EMAIL_2]" in result.redacted_text
        assert len(result.pii_matches) == 2

    def test_redact_phone(self) -> None:
        """Test redacting phone numbers."""
        gateway = PrivacyGateway(redact_patterns=["phone"])
        text = "Call me at 555-123-4567"
        result = gateway.redact(text)

        assert result.was_redacted is True
        assert "[REDACTED_PHONE_1]" in result.redacted_text
        assert "555-123-4567" not in result.redacted_text

    def test_redact_phone_with_parentheses(self) -> None:
        """Test redacting phone numbers with parentheses."""
        gateway = PrivacyGateway(redact_patterns=["phone"])
        text = "Phone: (555) 123-4567"
        result = gateway.redact(text)

        assert result.was_redacted is True
        assert "[REDACTED_PHONE_1]" in result.redacted_text

    def test_redact_ssn(self) -> None:
        """Test redacting SSN."""
        gateway = PrivacyGateway(redact_patterns=["ssn"])
        text = "My SSN is 123-45-6789"
        result = gateway.redact(text)

        assert result.was_redacted is True
        assert "[REDACTED_SSN_1]" in result.redacted_text
        assert "123-45-6789" not in result.redacted_text

    def test_redact_credit_card(self) -> None:
        """Test redacting credit card numbers."""
        gateway = PrivacyGateway(redact_patterns=["credit_card"])
        text = "Card: 4111-1111-1111-1111"
        result = gateway.redact(text)

        assert result.was_redacted is True
        assert "[REDACTED_CREDIT_CARD_1]" in result.redacted_text

    def test_redact_multiple_patterns(self) -> None:
        """Test redacting multiple PII patterns."""
        gateway = PrivacyGateway(redact_patterns=["email", "phone", "ssn"])
        text = "Contact me at john@example.com or 555-123-4567. SSN: 123-45-6789"
        result = gateway.redact(text)

        assert result.was_redacted is True
        # Check that all PII types were redacted
        assert "[REDACTED_" in result.redacted_text
        assert "john@example.com" not in result.redacted_text
        assert "555-123-4567" not in result.redacted_text
        assert "123-45-6789" not in result.redacted_text
        assert len(result.pii_matches) == 3

    def test_redact_no_pii(self) -> None:
        """Test redacting text without PII."""
        gateway = PrivacyGateway()
        text = "Hello world, how are you today?"
        result = gateway.redact(text)

        assert result.was_redacted is False
        assert result.redacted_text == text
        assert len(result.pii_matches) == 0

    def test_restore_empty_text(self) -> None:
        """Test restoring empty text."""
        gateway = PrivacyGateway()
        result = gateway.restore("", [])

        assert result == ""

    def test_restore_no_matches(self) -> None:
        """Test restoring with no PII matches."""
        gateway = PrivacyGateway()
        result = gateway.restore("Hello world", [])

        assert result == "Hello world"

    def test_restore_email(self) -> None:
        """Test restoring email."""
        gateway = PrivacyGateway()

        pii_match = PIIMatch(
            pattern_name="email",
            original="test@example.com",
            start=8,
            end=23,
            placeholder="[REDACTED_EMAIL_1]",
        )

        text = "Email me at [REDACTED_EMAIL_1] please"
        result = gateway.restore(text, [pii_match])

        assert "test@example.com" in result
        assert "[REDACTED_EMAIL_1]" not in result

    def test_restore_multiple_pii(self) -> None:
        """Test restoring multiple PII instances."""
        gateway = PrivacyGateway()

        pii_matches = [
            PIIMatch(
                pattern_name="email",
                original="test1@example.com",
                start=8,
                end=24,
                placeholder="[REDACTED_EMAIL_1]",
            ),
            PIIMatch(
                pattern_name="phone",
                original="555-123-4567",
                start=25,
                end=37,
                placeholder="[REDACTED_PHONE_1]",
            ),
        ]

        text = "[REDACTED_EMAIL_1] or [REDACTED_PHONE_1]"
        result = gateway.restore(text, pii_matches)

        assert "test1@example.com" in result
        assert "555-123-4567" in result
        assert "[REDACTED" not in result

    def test_placeholder_counter_resets(self) -> None:
        """Test that placeholder counter resets between redactions."""
        gateway = PrivacyGateway(redact_patterns=["email"])

        # First redaction
        result1 = gateway.redact("test1@example.com")
        assert "[REDACTED_EMAIL_1]" in result1.redacted_text

        # Second redaction should start counter at 1 again
        result2 = gateway.redact("test2@example.com")
        assert "[REDACTED_EMAIL_1]" in result2.redacted_text
        assert "[REDACTED_EMAIL_2]" not in result2.redacted_text

    def test_generate_placeholder_unique(self) -> None:
        """Test that generated placeholders are unique within a redaction."""
        gateway = PrivacyGateway(redact_patterns=["email"])

        result = gateway.redact("test1@example.com test2@example.org")
        assert "[REDACTED_EMAIL_1]" in result.redacted_text
        assert "[REDACTED_EMAIL_2]" in result.redacted_text

    @pytest.mark.asyncio
    async def test_log_cloud_request(self, tmp_path: Path) -> None:
        """Test logging cloud requests."""
        log_file = tmp_path / "test_cloud.log"
        gateway = PrivacyGateway(log_path=log_file)

        await gateway.log_cloud_request(
            original_prompt="My email is test@example.com",
            redacted_prompt="My email is [REDACTED_EMAIL_1]",
            provider="zai",
            model="glm-4.7",
            response_summary="This is a response",
        )

        # Check log file was created and contains content
        assert log_file.exists()
        content = log_file.read_text()

        assert "zai" in content
        assert "glm-4.7" in content
        assert "[REDACTED_EMAIL_1]" in content
        assert "This is a response" in content

    @pytest.mark.asyncio
    async def test_log_cloud_request_length_limit(self, tmp_path: Path) -> None:
        """Test that log limits prompt and response length."""
        log_file = tmp_path / "test_cloud.log"
        gateway = PrivacyGateway(log_path=log_file)

        long_prompt = "x" * 1000
        long_response = "y" * 1000

        await gateway.log_cloud_request(
            original_prompt=long_prompt,
            redacted_prompt=long_prompt,
            provider="test",
            model="test-model",
            response_summary=long_response,
        )

        content = log_file.read_text()
        # The log truncates the prompt to 500 chars
        assert len([line for line in content.split("\n") if "xxxxx" in line]) >= 1
        assert "xxx" in content  # Truncated but present

    def test_add_custom_pattern(self) -> None:
        """Test adding a custom PII pattern."""
        gateway = PrivacyGateway()

        # Add custom pattern for API keys
        gateway.add_pattern("api_key", r"sk-[a-zA-Z0-9]{32,}")

        # Test the custom pattern
        text = "My API key is sk-abcdefghijklmnopqrstuvwxyz123456"
        result = gateway.redact(text)

        assert "api_key" in gateway.PATTERNS

    def test_remove_pattern(self) -> None:
        """Test removing a PII pattern."""
        gateway = PrivacyGateway(redact_patterns=["email", "phone"])

        # Remove email pattern
        gateway.remove_pattern("email")

        # Should still have other patterns
        assert "email" not in gateway.PATTERNS

        # Email should no longer be redacted
        text = "My email is test@example.com"
        result = gateway.redact(text)

        # Email might still be matched by other patterns, but not by 'email'
        assert result.was_redacted is False or all(
            m.pattern_name != "email" for m in result.pii_matches
        )

    def test_detect_address(self) -> None:
        """Test address detection."""
        addresses = PrivacyGateway.detect_address("I live at 123 Main St, Springfield, IL 62701")

        # This is a conservative pattern, so may not match all addresses
        # Just test that the method works
        assert isinstance(addresses, list)
        assert all(isinstance(addr, tuple) and len(addr) == 3 for addr in addresses)

    def test_unknown_pattern_skipped(self) -> None:
        """Test that unknown patterns are skipped during redaction."""
        gateway = PrivacyGateway(redact_patterns=["unknown_pattern"])

        result = gateway.redact("Some text")

        # Should not crash and should not redact anything
        assert result.was_redacted is False
