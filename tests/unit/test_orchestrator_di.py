"""Test dependency injection in RoxyOrchestrator.

These tests verify that the orchestrator can work with injected dependencies,
making it easier to test and swap implementations.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from roxy.brain.llm_clients import LLMResponse
from roxy.brain.orchestrator import RoxyOrchestrator
from roxy.config import RoxyConfig


class MockPrivacyGateway:
    """Mock privacy gateway for testing."""

    def __init__(self) -> None:
        self._consent_mode = MagicMock()
        self.redact_called = False
        self.restore_called = False

    async def can_use_cloud(self) -> tuple[bool, str | None]:
        return True, None

    def redact(self, text: str) -> Any:
        self.redact_called = True
        from dataclasses import dataclass

        @dataclass
        class RedactionResult:
            redacted_text: str
            pii_matches: list = None
            was_redacted: bool = False

        return RedactionResult(redacted_text=text, pii_matches=[], was_redacted=False)

    def restore(self, text: str, pii_matches: list) -> str:
        self.restore_called = True
        return text

    async def log_cloud_request(
        self,
        original_prompt: str,
        redacted_prompt: str,
        provider: str,
        model: str,
        response_summary: str,
    ) -> None:
        pass

    @property
    def consent_mode(self) -> Any:
        return self._consent_mode


class MockLocalLLMClient:
    """Mock local LLM client for testing."""

    def __init__(self, response: str = "Test response") -> None:
        self.response = response
        self._model = "test-model"
        self._router_model = "test-router-model"
        self._host = "http://localhost:11434"
        self.closed = False

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=self.response,
            model=self._model,
            provider="local",
        )

    async def close(self) -> None:
        self.closed = True

    @property
    def model(self) -> str:
        return self._model

    @property
    def router_model(self) -> str:
        return self._router_model

    @property
    def host(self) -> str:
        return self._host


class MockCloudLLMClient:
    """Mock cloud LLM client for testing."""

    def __init__(self, response: str = "Cloud response") -> None:
        self.response = response
        self._provider = "test-cloud"
        self._model = "test-cloud-model"
        self.closed = False

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=self.response,
            model=self._model,
            provider="cloud",
        )

    async def close(self) -> None:
        self.closed = True

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model


class MockConfidenceRouter:
    """Mock confidence router for testing."""

    def __init__(self, response: str = "Routed response") -> None:
        self.response = response
        self.route_called = False
        self._stats = {
            "total_requests": 0,
            "local_requests": 0,
            "cloud_requests": 0,
            "blocked_requests": 0,
            "local_rate": 1.0,
            "cloud_rate": 0.0,
        }

    async def route(
        self,
        request: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.route_called = True
        self._stats["total_requests"] += 1
        return LLMResponse(
            content=self.response,
            model="routed-model",
            provider="local",
        )

    def get_statistics(self) -> dict[str, int | float]:
        return self._stats.copy()


class MockMemoryManager:
    """Mock memory manager for testing."""

    def __init__(self) -> None:
        self.initialized = False
        self.memories: dict[str, str] = {}
        self.session_messages: list[dict] = []
        self.ended_conversations = 0

    async def initialize(self) -> None:
        self.initialized = True

    async def build_context_for_llm(self, query: str) -> dict[str, Any]:
        return {
            "session_messages": self.session_messages,
            "relevant_history": [],
            "relevant_memories": [],
            "user_preferences": {},
            "current_conversation_id": "test-conv-id",
        }

    async def add_to_session(self, role: str, content: str) -> None:
        self.session_messages.append({"role": role, "content": content})

    async def remember(self, key: str, value: str) -> None:
        self.memories[key] = value

    async def recall(self, query: str, limit: int = 5) -> list[str]:
        return [v for k, v in self.memories.items() if query.lower() in k.lower()][:limit]

    async def end_conversation(self) -> None:
        self.ended_conversations += 1


class MockSkillRegistry:
    """Mock skill registry for testing."""

    def __init__(self) -> None:
        self.skills: list[dict] = []
        self.find_skill_result = (None, 0.0)

    def find_skill(self, intent: str, parameters: dict) -> tuple[Any, float]:
        return self.find_skill_result

    def list_skills(self) -> list[dict]:
        return self.skills.copy()

    async def initialize(self) -> None:
        pass


@pytest.fixture
def mock_config() -> RoxyConfig:
    """Create a mock RoxyConfig for testing."""
    return RoxyConfig(
        name="TestRoxy",
        version="0.1.0",
        data_dir="/tmp/test_roxy",
        log_level="DEBUG",
    )


@pytest.fixture
def all_mocks() -> dict[str, Any]:
    """Create all mock dependencies."""
    return {
        "privacy": MockPrivacyGateway(),
        "local_client": MockLocalLLMClient(),
        "cloud_client": MockCloudLLMClient(),
        "router": MockConfidenceRouter(),
        "memory": MockMemoryManager(),
        "skill_registry": MockSkillRegistry(),
    }


class TestOrchestratorDI:
    """Test dependency injection in RoxyOrchestrator."""

    def test_orchestrator_accepts_all_dependencies(
        self,
        mock_config: RoxyConfig,
        all_mocks: dict[str, Any],
    ) -> None:
        """Test that orchestrator can be created with all dependencies injected."""
        orchestrator = RoxyOrchestrator(
            config=mock_config,
            skill_registry=all_mocks["skill_registry"],
            local_client=all_mocks["local_client"],
            cloud_client=all_mocks["cloud_client"],
            privacy=all_mocks["privacy"],
            router=all_mocks["router"],
            memory=all_mocks["memory"],
        )

        assert orchestrator.config == mock_config
        assert orchestrator.skill_registry is all_mocks["skill_registry"]
        assert orchestrator.local_client is all_mocks["local_client"]
        assert orchestrator.cloud_client is all_mocks["cloud_client"]
        assert orchestrator.privacy is all_mocks["privacy"]
        assert orchestrator.router is all_mocks["router"]
        assert orchestrator.memory is all_mocks["memory"]

    def test_orchestrator_backward_compatible(
        self,
        mock_config: RoxyConfig,
    ) -> None:
        """Test that orchestrator still works without injected dependencies."""
        orchestrator = RoxyOrchestrator(
            config=mock_config,
        )

        # Dependencies should be created
        assert orchestrator.local_client is not None
        assert orchestrator.cloud_client is not None
        assert orchestrator.privacy is not None
        assert orchestrator.router is not None
        assert orchestrator.memory is not None

    @pytest.mark.asyncio
    async def test_orchestrator_uses_injected_memory(
        self,
        mock_config: RoxyConfig,
        all_mocks: dict[str, Any],
    ) -> None:
        """Test that orchestrator uses injected memory manager."""
        orchestrator = RoxyOrchestrator(
            config=mock_config,
            memory=all_mocks["memory"],
        )

        await orchestrator.initialize()

        # Memory should be initialized
        assert all_mocks["memory"].initialized is True

    @pytest.mark.asyncio
    async def test_orchestrator_uses_injected_router(
        self,
        mock_config: RoxyConfig,
        all_mocks: dict[str, Any],
    ) -> None:
        """Test that orchestrator uses injected router for LLM calls."""
        orchestrator = RoxyOrchestrator(
            config=mock_config,
            skill_registry=all_mocks["skill_registry"],
            router=all_mocks["router"],
            memory=all_mocks["memory"],
        )

        await orchestrator.initialize()

        # Process a request
        response = await orchestrator.process("Hello")

        # Router should have been called
        assert all_mocks["router"].route_called is True
        assert response == "Routed response"

    @pytest.mark.asyncio
    async def test_orchestrator_shutdown_closes_clients(
        self,
        mock_config: RoxyConfig,
        all_mocks: dict[str, Any],
    ) -> None:
        """Test that shutdown closes injected LLM clients."""
        orchestrator = RoxyOrchestrator(
            config=mock_config,
            local_client=all_mocks["local_client"],
            cloud_client=all_mocks["cloud_client"],
            memory=all_mocks["memory"],
        )

        await orchestrator.initialize()
        await orchestrator.shutdown()

        # Clients should be closed
        assert all_mocks["local_client"].closed is True
        assert all_mocks["cloud_client"].closed is True

    @pytest.mark.asyncio
    async def test_orchestrator_statistics_include_router_stats(
        self,
        mock_config: RoxyConfig,
        all_mocks: dict[str, Any],
    ) -> None:
        """Test that orchestrator statistics include router statistics."""
        orchestrator = RoxyOrchestrator(
            config=mock_config,
            skill_registry=all_mocks["skill_registry"],
            router=all_mocks["router"],
            memory=all_mocks["memory"],
        )

        await orchestrator.initialize()

        # Process some requests
        await orchestrator.process("Request 1")
        await orchestrator.process("Request 2")

        stats = await orchestrator.get_statistics()

        # Router stats should be included
        assert "routing_stats" in stats
        assert stats["routing_stats"]["total_requests"] == 2


class TestOrchestratorIsolation:
    """Test that injected dependencies provide proper isolation."""

    @pytest.mark.asyncio
    async def test_mock_memory_isolated_from_real_storage(
        self,
        mock_config: RoxyConfig,
    ) -> None:
        """Test that using mock memory doesn't touch real storage."""
        mock_memory = MockMemoryManager()
        mock_router = MockConfidenceRouter(response="Test response")
        mock_registry = MockSkillRegistry()

        orchestrator = RoxyOrchestrator(
            config=mock_config,
            memory=mock_memory,
            router=mock_router,
            skill_registry=mock_registry,
        )

        await orchestrator.initialize()
        await orchestrator.process("Hello")

        # Mock memory should have been used
        assert mock_memory.initialized is True
        assert len(mock_memory.session_messages) > 0

    @pytest.mark.asyncio
    async def test_mock_llm_no_network_calls(
        self,
        mock_config: RoxyConfig,
    ) -> None:
        """Test that using mock LLM doesn't make network calls."""
        mock_client = MockLocalLLMClient(response="Mocked!")
        mock_router = MockConfidenceRouter(response="Routed!")
        mock_memory = MockMemoryManager()
        mock_registry = MockSkillRegistry()

        orchestrator = RoxyOrchestrator(
            config=mock_config,
            local_client=mock_client,
            router=mock_router,
            memory=mock_memory,
            skill_registry=mock_registry,
        )

        await orchestrator.initialize()
        response = await orchestrator.process("Hello")

        # Should use mocked response, not make real LLM call
        assert response == "Routed!"
