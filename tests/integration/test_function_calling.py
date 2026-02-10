"""Integration tests for Agno function calling with Roxy skills.

Tests the 10 test inputs specified in Task 1 of the hardening session.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.brain.orchestrator import RoxyOrchestrator
from roxy.brain.tool_adapter import IntentClassifier, SkillToolAdapter
from roxy.config import CloudLLMConfig, LocalLLMConfig, PrivacyConfig, RoxyConfig
from roxy.skills.registry import SkillRegistry

# Test inputs and expected outputs from Task 1
TEST_CASES = [
    {
        "input": "open Safari",
        "expected_skill": "AppLauncherSkill",
        "expected_intent": "open",
        "description": "Simple app launch command",
    },
    {
        "input": "can you check if I've got anything on tomorrow",
        "expected_skill": "CalendarSkill",
        "expected_intent": "calendar",
        "description": "Calendar check for tomorrow",
    },
    {
        "input": "find me cheap flights to London next month",
        "expected_skill": "FlightSearchSkill",
        "expected_intent": "flight_search",
        "description": "Flight search request",
    },
    {
        "input": "what's the latest news about AI",
        "expected_skill": "WebSearchSkill",
        "expected_intent": "web_search",
        "description": "Web search for news",
    },
    {
        "input": "remember that my dentist is Dr. Lindström",
        "expected_skill": "memory",  # Handled by orchestrator, not a specific skill
        "expected_intent": "remember",
        "description": "Memory storage request",
    },
    {
        "input": "what's running on my system",
        "expected_skill": "SystemInfoSkill",
        "expected_intent": "system_info",
        "description": "System info request",
    },
    {
        "input": "I need to set up my coding workspace",
        "expected_skill": "WindowManagerSkill",
        "expected_intent": "window_manager",
        "description": "Workspace setup request",
    },
    {
        "input": "how's my disk space looking",
        "expected_skill": "SystemInfoSkill",
        "expected_intent": "system_info",
        "description": "Disk space query",
    },
    {
        "input": "any new emails?",
        "expected_skill": "EmailSkill",
        "expected_intent": "email",
        "description": "Email check request",
    },
    {
        "input": "tell me a joke",
        "expected_skill": None,  # General conversation, no skill
        "expected_intent": "general_conversation",
        "description": "General conversation (no skill)",
    },
]


@pytest.fixture
def integration_config(temp_dir):
    """Return test configuration for integration tests."""
    return RoxyConfig(
        name="TestRoxy",
        version="0.1.0-test",
        data_dir=str(temp_dir / "data"),
        log_level="DEBUG",
        llm_local=LocalLLMConfig(
            model="qwen3:8b",
            router_model="qwen3:0.6b",
            temperature=0.7,
            max_tokens=2048,
            host="http://localhost:11434",
        ),
        llm_cloud=CloudLLMConfig(
            provider="zai",
            model="glm-4.7",
            confidence_threshold=0.7,
            api_key="test-api-key",
        ),
        privacy=PrivacyConfig(
            redact_patterns=["email", "phone", "ssn"],
            cloud_consent="never",  # Never use cloud in tests
            log_cloud_requests=True,
            pii_redaction_enabled=True,
        ),
    )


@pytest.fixture
async def mock_orchestrator(integration_config):
    """Create a mock orchestrator for testing function calling."""
    SkillRegistry.reset()

    # Mock the local client
    with patch("roxy.brain.llm_clients.AsyncOpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 30

        mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

        # Mock the Ollama availability check
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": []}

            # Create orchestrator
            orchestrator = RoxyOrchestrator(config=integration_config)

            # Initialize
            await orchestrator.initialize()

            yield orchestrator

            # Cleanup
            await orchestrator.shutdown()


class TestIntentClassification:
    """Test intent classification for the 10 test cases."""

    @pytest.fixture
    def mock_llm_response(self):
        """Mock responses for different test inputs."""
        return {
            "open Safari": '{"intent": "app_launcher", "confidence": 0.9, "parameters": {"user_input": "open Safari", "intent": "open", "app_name": "Safari"}}',
            "can you check if I've got anything on tomorrow": '{"intent": "calendar", "confidence": 0.85, "parameters": {"user_input": "check tomorrow", "intent": "calendar"}}',
            "find me cheap flights to London next month": '{"intent": "flight_search", "confidence": 0.9, "parameters": {"user_input": "flights to London", "intent": "flight_search", "destination": "London"}}',
            "what's the latest news about AI": '{"intent": "web_search", "confidence": 0.85, "parameters": {"user_input": "news about AI", "intent": "web_search", "query": "latest news about AI"}}',
            "remember that my dentist is Dr. Lindström": '{"intent": "memory_store", "confidence": 0.8, "parameters": {"user_input": "remember dentist", "intent": "remember", "fact": "my dentist is Dr. Lindström"}}',
            "what's running on my system": '{"intent": "system_info", "confidence": 0.9, "parameters": {"user_input": "system info", "intent": "system_info"}}',
            "I need to set up my coding workspace": '{"intent": "window_manager", "confidence": 0.85, "parameters": {"user_input": "coding workspace", "intent": "window_manager"}}',
            "how's my disk space looking": '{"intent": "system_info", "confidence": 0.9, "parameters": {"user_input": "disk space", "intent": "system_info"}}',
            "any new emails?": '{"intent": "email", "confidence": 0.9, "parameters": {"user_input": "check emails", "intent": "email"}}',
            "tell me a joke": '{"intent": "general_conversation", "confidence": 0.5, "parameters": {"user_input": "tell joke", "intent": "general_conversation"}}',
        }

    @pytest.fixture
    def classifier(self, mock_llm_response, integration_config):
        """Create an IntentClassifier for testing with mock responses."""
        from roxy.brain.llm_clients import OllamaClient

        # Create OllamaClient with mocked _client
        client = OllamaClient(
            host=integration_config.llm_local.host,
            model=integration_config.llm_local.model,
            router_model=integration_config.llm_local.router_model,
        )

        # Mock the internal client's create method
        async def mock_create(*args, **kwargs):
            # Get the user input from the prompt
            messages = kwargs.get("messages", [])
            content = messages[-1].get("content", "") if messages else ""

            # Find which test case matches
            for user_input, response in mock_llm_response.items():
                if user_input.lower() in content.lower():
                    mock_response = MagicMock()
                    mock_response.choices = [MagicMock()]
                    mock_response.choices[0].message.content = response
                    mock_response.choices[0].finish_reason = "stop"
                    return mock_response

            # Default fallback
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = mock_llm_response.get("tell me a joke")
            mock_response.choices[0].finish_reason = "stop"
            return mock_response

        # Replace the _client's create method
        client._client.chat.completions.create = mock_create

        return IntentClassifier(
            local_client=client,
            model=integration_config.llm_local.router_model,
        )

    @pytest.mark.asyncio
    async def test_classify_open_safari(self, classifier):
        """Test: 'open Safari' → AppLauncherSkill."""
        result = await classifier.classify("open Safari")

        assert result["intent"] in ["app_launcher", "open"]
        assert result["confidence"] >= 0.7
        assert "Safari" in result["parameters"].get("app_name", "")

    @pytest.mark.asyncio
    async def test_classify_calendar_tomorrow(self, classifier):
        """Test: 'can you check if I've got anything on tomorrow' → CalendarSkill."""
        result = await classifier.classify("can you check if I've got anything on tomorrow")

        assert result["intent"] in ["calendar", "calendar_check"]
        assert result["confidence"] >= 0.7

    @pytest.mark.asyncio
    async def test_classify_flights_london(self, classifier):
        """Test: 'find me cheap flights to London next month' → FlightSearchSkill."""
        result = await classifier.classify("find me cheap flights to London next month")

        assert result["intent"] in ["flight_search", "flights"]
        assert result["confidence"] >= 0.7
        assert "London" in result["parameters"].get("destination", "")

    @pytest.mark.asyncio
    async def test_classify_news_ai(self, classifier):
        """Test: 'what's the latest news about AI' → WebSearchSkill."""
        result = await classifier.classify("what's the latest news about AI")

        assert result["intent"] in ["web_search", "search"]
        assert result["confidence"] >= 0.7

    @pytest.mark.asyncio
    async def test_classify_remember_dentist(self, classifier):
        """Test: 'remember that my dentist is Dr. Lindström' → memory store."""
        result = await classifier.classify("remember that my dentist is Dr. Lindström")

        assert result["intent"] in ["memory_store", "remember", "memory"]
        assert result["confidence"] >= 0.7

    @pytest.mark.asyncio
    async def test_classify_whats_running(self, classifier):
        """Test: 'what's running on my system' → SystemInfoSkill."""
        result = await classifier.classify("what's running on my system")

        assert result["intent"] in ["system_info", "system"]
        assert result["confidence"] >= 0.7

    @pytest.mark.asyncio
    async def test_classify_coding_workspace(self, classifier):
        """Test: 'I need to set up my coding workspace' → WindowManagerSkill."""
        result = await classifier.classify("I need to set up my coding workspace")

        assert result["intent"] in ["window_manager", "workspace", "layout"]
        assert result["confidence"] >= 0.7

    @pytest.mark.asyncio
    async def test_classify_disk_space(self, classifier):
        """Test: 'how's my disk space looking' → SystemInfoSkill."""
        result = await classifier.classify("how's my disk space looking")

        assert result["intent"] in ["system_info", "system"]
        assert result["confidence"] >= 0.7

    @pytest.mark.asyncio
    async def test_classify_new_emails(self, classifier):
        """Test: 'any new emails?' → EmailSkill."""
        result = await classifier.classify("any new emails?")

        assert result["intent"] in ["email", "emails", "mail"]
        assert result["confidence"] >= 0.7

    @pytest.mark.asyncio
    async def test_classify_tell_joke(self, classifier):
        """Test: 'tell me a joke' → general conversation (no skill)."""
        result = await classifier.classify("tell me a joke")

        assert result["intent"] == "general_conversation"
        # Lower confidence for general conversation is acceptable
        assert 0.0 <= result["confidence"] <= 1.0


class TestToolAdapterIntegration:
    """Test the SkillToolAdapter with actual skills."""

    @pytest.fixture
    def adapter(self, integration_config):
        """Create a SkillToolAdapter with registered skills."""
        # Reset and register skills
        SkillRegistry.reset()
        registry = SkillRegistry.get_instance()

        # Import and register some actual skills
        from roxy.skills.productivity.calendar import CalendarSkill
        from roxy.skills.system.app_launcher import AppLauncherSkill
        from roxy.skills.system.file_search import FileSearchSkill

        registry.register(AppLauncherSkill)
        registry.register(CalendarSkill)
        registry.register(FileSearchSkill)

        # Create mock orchestrator context
        mock_orchestrator = MagicMock()
        mock_orchestrator.config = integration_config
        mock_orchestrator.memory = MagicMock()
        mock_orchestrator.local_client = MagicMock()
        mock_orchestrator.privacy = MagicMock()
        mock_orchestrator.conversation_history = []

        return SkillToolAdapter(
            skill_registry=registry,
            orchestrator_context=mock_orchestrator,
        )

    def test_get_all_tools(self, adapter):
        """Test getting all tools from the adapter."""
        tools = adapter.get_all_tools()

        # Should have tools for registered skills
        tool_names = {tool.name for tool in tools}
        assert len(tool_names) > 0

        # Check for some expected skills
        expected_skills = ["app_launcher", "calendar", "file_search"]
        for skill in expected_skills:
            assert skill in tool_names, f"Expected skill '{skill}' not found in tools: {tool_names}"

    def test_tool_metadata(self, adapter):
        """Test that tools have proper metadata."""
        tools = adapter.get_all_tools()

        for tool in tools:
            assert tool.name, f"Tool missing name: {tool}"
            assert tool.description, f"Tool '{tool.name}' missing description"
            # Agno Function stores callable internally, verify parameters exist
            assert isinstance(tool.parameters, dict), f"Tool '{tool.name}' parameters is not a dict"


class TestEndToEndFunctionCalling:
    """End-to-end tests for function calling with the orchestrator."""

    @pytest.mark.skip(reason="Requires running Ollama server for full E2E test")
    @pytest.mark.asyncio
    async def test_open_safari_e2e(self, mock_orchestrator):
        """End-to-end test: 'open Safari' should trigger AppLauncherSkill."""
        # This would require a real Ollama server to test fully
        # Marked as skip for CI environments
        response = await mock_orchestrator.process("open Safari")
        assert "Safari" in response or "open" in response.lower()

    @pytest.mark.skip(reason="Requires running Ollama server for full E2E test")
    @pytest.mark.asyncio
    async def test_general_conversation_e2e(self, mock_orchestrator):
        """End-to-end test: 'tell me a joke' should use LLM directly."""
        response = await mock_orchestrator.process("tell me a joke")
        assert response  # Should get some response
