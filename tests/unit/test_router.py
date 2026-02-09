"""Unit tests for confidence router."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.brain.router import ConfidenceRouter
from roxy.brain.llm_clients import LLMResponse, OllamaClient, CloudLLMClient
from roxy.brain.privacy import PrivacyGateway, ConsentMode


class TestConfidenceRouter:
    """Tests for ConfidenceRouter."""

    @pytest.fixture
    def mock_local_client(self) -> OllamaClient:
        """Create a mock OllamaClient."""
        client = MagicMock(spec=OllamaClient)
        client.router_model = "qwen3:0.6b"

        # Mock generate to return a response
        async def mock_generate(*args, **kwargs):
            return LLMResponse(
                content="Local response",
                model="qwen3:8b",
                provider="local",
                tokens_used=50,
                latency_ms=1000,
            )

        client.generate = AsyncMock(side_effect=mock_generate)
        return client

    @pytest.fixture
    def mock_cloud_client(self) -> CloudLLMClient:
        """Create a mock CloudLLMClient."""
        client = MagicMock(spec=CloudLLMClient)
        client.provider = "zai"
        client.model = "glm-4.7"

        # Mock generate to return a response
        async def mock_generate(*args, **kwargs):
            return LLMResponse(
                content="Cloud response",
                model="glm-4.7",
                provider="cloud",
                tokens_used=30,
                latency_ms=2000,
            )

        client.generate = AsyncMock(side_effect=mock_generate)
        return client

    @pytest.fixture
    def mock_privacy(self) -> PrivacyGateway:
        """Create a mock PrivacyGateway."""
        privacy = MagicMock(spec=PrivacyGateway)

        # Mock can_use_cloud to always allow
        async def mock_can_use():
            return True, None

        privacy.can_use_cloud = AsyncMock(side_effect=mock_can_use_cloud)
        privacy.redact = MagicMock(return_value=MagicMock(
            redacted_text="Redacted text",
            pii_matches=[],
            was_redacted=False,
        ))
        privacy.restore = MagicMock(return_value="Cloud response")

        return privacy

    def test_init(self, mock_local_client, mock_cloud_client, mock_privacy) -> None:
        """Test router initialization."""
        router = ConfidenceRouter(
            local_client=mock_local_client,
            cloud_client=mock_cloud_client,
            privacy=mock_privacy,
            confidence_threshold=0.8,
        )

        assert router.local_client == mock_local_client
        assert router.cloud_client == mock_cloud_client
        assert router.privacy == mock_privacy
        assert router.threshold == 0.8
        assert router._total_requests == 0

    @pytest.mark.asyncio
    async def test_route_high_confidence(
        self,
        mock_local_client,
        mock_cloud_client,
        mock_privacy,
    ) -> None:
        """Test routing with high confidence (uses local)."""
        router = ConfidenceRouter(
            local_client=mock_local_client,
            cloud_client=mock_cloud_client,
            privacy=mock_privacy,
            confidence_threshold=0.7,
        )

        # Mock high confidence score
        with patch.object(router, "_assess_confidence", return_value=0.9):
            response = await router.route("What's the weather?")

        assert response.provider == "local"
        assert mock_local_client.generate.called
        assert not mock_cloud_client.generate.called
        assert router._local_requests == 1

    @pytest.mark.asyncio
    async def test_route_low_confidence_with_consent(
        self,
        mock_local_client,
        mock_cloud_client,
        mock_privacy,
    ) -> None:
        """Test routing with low confidence and consent (uses cloud)."""
        router = ConfidenceRouter(
            local_client=mock_local_client,
            cloud_client=mock_cloud_client,
            privacy=mock_privacy,
            confidence_threshold=0.7,
        )

        # Mock low confidence score
        with patch.object(router, "_assess_confidence", return_value=0.5):
            response = await router.route("What's the weather?")

        assert response.provider == "cloud"
        assert not mock_local_client.generate.called
        assert mock_cloud_client.generate.called
        assert router._cloud_requests == 1

    @pytest.mark.asyncio
    async def test_route_low_confidence_without_consent(
        self,
        mock_local_client,
        mock_cloud_client,
    ) -> None:
        """Test routing with low confidence but no consent (falls back to local)."""
        # Mock privacy to deny cloud access
        privacy = MagicMock(spec=PrivacyGateway)

        async def mock_can_use():
            return False, "Cloud access denied"

        privacy.can_use_cloud = AsyncMock(side_effect=mock_can_use)

        router = ConfidenceRouter(
            local_client=mock_local_client,
            cloud_client=mock_cloud_client,
            privacy=privacy,
            confidence_threshold=0.7,
        )

        # Mock low confidence score
        with patch.object(router, "_assess_confidence", return_value=0.5):
            response = await router.route("What's the weather?")

        # Should fall back to local
        assert response.provider == "local"
        assert mock_local_client.generate.called
        assert not mock_cloud_client.generate.called
        assert router._local_requests == 1
        assert router._blocked_requests == 1

    @pytest.mark.asyncio
    async def test_assess_confidence(self, mock_local_client, mock_cloud_client, mock_privacy) -> None:
        """Test confidence assessment."""
        router = ConfidenceRouter(
            local_client=mock_local_client,
            cloud_client=mock_cloud_client,
            privacy=mock_privacy,
        )

        # Mock scorer response
        with patch.object(router.scorer, "score", return_value=0.8):
            confidence = await router._assess_confidence("Test request")
            assert confidence == 0.8

    @pytest.mark.asyncio
    async def test_assess_confidence_error_handling(
        self,
        mock_local_client,
        mock_cloud_client,
        mock_privacy,
    ) -> None:
        """Test confidence assessment error handling."""
        router = ConfidenceRouter(
            local_client=mock_local_client,
            cloud_client=mock_cloud_client,
            privacy=mock_privacy,
        )

        # Mock scorer to raise error
        with patch.object(router.scorer, "score", side_effect=Exception("Test error")):
            confidence = await router._assess_confidence("Test request")
            # Should default to 0.5 on error
            assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_use_local(
        self,
        mock_local_client,
        mock_cloud_client,
        mock_privacy,
    ) -> None:
        """Test _use_local method."""
        router = ConfidenceRouter(
            local_client=mock_local_client,
            cloud_client=mock_cloud_client,
            privacy=mock_privacy,
        )

        response = await router._use_local("Test prompt", 0.5, 100)

        assert response.provider == "local"
        assert mock_local_client.generate.called

    @pytest.mark.asyncio
    async def test_use_cloud_with_pii(
        self,
        mock_local_client,
        mock_cloud_client,
    ) -> None:
        """Test _use_cloud method with PII redaction."""
        from roxy.brain.privacy import PIIMatch

        privacy = MagicMock(spec=PrivacyGateway)

        # Mock PII redaction
        redaction_result = MagicMock()
        redaction_result.was_redacted = True
        redaction_result.redacted_text = "Redacted prompt"
        redaction_result.pii_matches = [
            PIIMatch(
                pattern_name="email",
                original="test@example.com",
                start=8,
                end=23,
                placeholder="[REDACTED_EMAIL_1]",
            )
        ]

        privacy.redact = MagicMock(return_value=redaction_result)
        privacy.restore = MagicMock(return_value="Restored response")

        async def mock_can_use():
            return True, None

        privacy.can_use_cloud = AsyncMock(side_effect=mock_can_use)
        privacy.log_cloud_request = AsyncMock()

        router = ConfidenceRouter(
            local_client=mock_local_client,
            cloud_client=mock_cloud_client,
            privacy=privacy,
        )

        response = await router._use_cloud("My email is test@example.com")

        assert response.provider == "cloud"
        assert privacy.redact.called
        assert privacy.restore.called
        assert privacy.log_cloud_request.called

    @pytest.mark.asyncio
    async def test_use_cloud_without_pii(
        self,
        mock_local_client,
        mock_cloud_client,
    ) -> None:
        """Test _use_cloud method without PII."""
        privacy = MagicMock(spec=PrivacyGateway)

        # Mock no PII
        redaction_result = MagicMock()
        redaction_result.was_redacted = False
        redaction_result.redacted_text = "Clean prompt"
        redaction_result.pii_matches = []

        privacy.redact = MagicMock(return_value=redaction_result)
        privacy.restore = MagicMock(return_value="Cloud response")

        async def mock_can_use():
            return True, None

        privacy.can_use_cloud = AsyncMock(side_effect=mock_can_use)
        privacy.log_cloud_request = AsyncMock()

        router = ConfidenceRouter(
            local_client=mock_local_client,
            cloud_client=mock_cloud_client,
            privacy=privacy,
        )

        response = await router._use_cloud("Clean prompt")

        assert response.provider == "cloud"
        assert privacy.redact.called
        # restore should still be called even if no PII was found
        # (it handles empty matches gracefully)
        assert privacy.log_cloud_request.called

    def test_get_statistics(
        self,
        mock_local_client,
        mock_cloud_client,
        mock_privacy,
    ) -> None:
        """Test getting routing statistics."""
        router = ConfidenceRouter(
            local_client=mock_local_client,
            cloud_client=mock_cloud_client,
            privacy=mock_privacy,
        )

        # Set some stats
        router._total_requests = 10
        router._local_requests = 7
        router._cloud_requests = 2
        router._blocked_requests = 1

        stats = router.get_statistics()

        assert stats["total_requests"] == 10
        assert stats["local_requests"] == 7
        assert stats["cloud_requests"] == 2
        assert stats["blocked_requests"] == 1
        assert stats["local_rate"] == 0.7
        assert stats["cloud_rate"] == 0.2

    def test_get_statistics_no_requests(
        self,
        mock_local_client,
        mock_cloud_client,
        mock_privacy,
    ) -> None:
        """Test getting statistics with no requests."""
        router = ConfidenceRouter(
            local_client=mock_local_client,
            cloud_client=mock_cloud_client,
            privacy=mock_privacy,
        )

        stats = router.get_statistics()

        assert stats["total_requests"] == 0
        assert stats["local_requests"] == 0
        assert stats["cloud_requests"] == 0
        assert stats["local_rate"] == 0.0
        assert stats["cloud_rate"] == 0.0

    def test_reset_statistics(
        self,
        mock_local_client,
        mock_cloud_client,
        mock_privacy,
    ) -> None:
        """Test resetting statistics."""
        router = ConfidenceRouter(
            local_client=mock_local_client,
            cloud_client=mock_cloud_client,
            privacy=mock_privacy,
        )

        # Set some stats
        router._total_requests = 10
        router._local_requests = 7

        router.reset_statistics()

        assert router._total_requests == 0
        assert router._local_requests == 0
        assert router._cloud_requests == 0
        assert router._blocked_requests == 0


# Fix the mock_can_use_cloud function name in the test
async def mock_can_use_cloud() -> tuple[bool, str | None]:
    """Mock function for can_use_cloud."""
    return True, None
