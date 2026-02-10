"""Tests for the SkillToolAdapter and IntentClassifier.

Tests the integration between Roxy skills and Agno function calling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.brain.tool_adapter import IntentClassifier, SkillToolAdapter
from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult
from roxy.skills.registry import SkillRegistry


class DummyTestSkill(RoxySkill):
    """A dummy skill for testing."""

    name = "test_skill"
    description = "A test skill for unit testing"
    triggers = ["test", "dummy", "example"]
    permissions = []

    async def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            response_text=f"Test skill executed with: {context.user_input}",
        )


class DummyAppLauncherSkill(RoxySkill):
    """A dummy app launcher skill for testing."""

    name = "app_launcher"
    description = "Launch applications by name"
    triggers = ["open", "launch", "start"]
    permissions = [Permission.APPLESCRIPT]

    async def execute(self, context: SkillContext) -> SkillResult:
        app_name = context.parameters.get("app_name", "unknown")
        return SkillResult(
            success=True,
            response_text=f"Opening {app_name}",
        )


class DummyCalendarSkill(RoxySkill):
    """A dummy calendar skill for testing."""

    name = "calendar"
    description = "Access and manage calendar events"
    triggers = ["what's on my calendar", "calendar today", "schedule"]
    permissions = [Permission.APPLESCRIPT, Permission.CALENDAR]

    async def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            success=True,
            response_text="You have 2 events today.",
            data={"events": ["Meeting at 2pm", "Dinner at 7pm"]},
        )


class TestSkillToolAdapter:
    """Tests for SkillToolAdapter."""

    @pytest.fixture
    def registry(self) -> SkillRegistry:
        """Return a test skill registry."""
        SkillRegistry.reset()
        registry = SkillRegistry.get_instance()
        registry.register(DummyTestSkill)
        registry.register(DummyAppLauncherSkill)
        registry.register(DummyCalendarSkill)
        return registry

    @pytest.fixture
    def mock_orchestrator(self, mock_config):
        """Return a mock orchestrator."""
        orchestrator = MagicMock()
        orchestrator.config = mock_config
        orchestrator.memory = MagicMock()
        orchestrator.local_client = MagicMock()
        orchestrator.privacy = MagicMock()
        orchestrator.conversation_history = []
        return orchestrator

    @pytest.fixture
    def adapter(self, registry, mock_orchestrator) -> SkillToolAdapter:
        """Return a SkillToolAdapter instance."""
        return SkillToolAdapter(
            skill_registry=registry,
            orchestrator_context=mock_orchestrator,
        )

    def test_init(self, adapter, registry, mock_orchestrator):
        """Test adapter initialization."""
        assert adapter.registry == registry
        assert adapter.orchestrator == mock_orchestrator
        assert adapter._tool_cache == {}

    def test_skill_to_tool(self, adapter):
        """Test converting a skill to an Agno tool."""
        skill = DummyTestSkill()
        tool = adapter.skill_to_tool(skill)

        assert tool.name == "test_skill"
        assert "A test skill for unit testing" in tool.description
        assert "test" in tool.description or "Triggered" in tool.description
        # Agno Function stores the callable internally, check entrypoint exists
        # The function is wrapped by Agno's mechanism

    def test_tool_cache(self, adapter):
        """Test that tools are cached."""
        skill = DummyTestSkill()

        tool1 = adapter.skill_to_tool(skill)
        tool2 = adapter.skill_to_tool(skill)

        assert tool1 is tool2  # Should be the same cached object
        # Cache key includes version, check for key with skill name prefix
        assert any(k.startswith("test_skill:") for k in adapter._tool_cache.keys())

    def test_extract_parameters(self, adapter):
        """Test parameter extraction from a skill."""
        skill = DummyTestSkill()
        params = adapter._extract_parameters(skill)

        assert params["type"] == "object"
        assert "properties" in params
        assert "user_input" in params["properties"]
        assert "intent" in params["properties"]
        assert params["properties"]["user_input"]["type"] == "string"
        assert params["properties"]["intent"]["type"] == "string"

    def test_clear_cache(self, adapter):
        """Test clearing the tool cache."""
        skill = DummyTestSkill()
        adapter.skill_to_tool(skill)
        # Cache should not be empty after creating a tool
        assert len(adapter._tool_cache) > 0

        adapter.clear_cache()
        assert adapter._tool_cache == {}

    @pytest.mark.asyncio
    async def test_execute_from_tool_call_success(self, adapter, mock_orchestrator):
        """Test executing a skill from a tool call."""
        parameters = {
            "user_input": "open Safari",
            "intent": "open",
        }

        # Mock the memory's build_context_for_llm method
        mock_orchestrator.memory.build_context_for_llm = AsyncMock(return_value=[])

        result = await adapter.execute_from_tool_call("app_launcher", parameters)

        assert "Opening" in result
        assert "success" not in result.lower()  # Should not be an error message

    @pytest.mark.asyncio
    async def test_execute_from_tool_call_skill_not_found(self, adapter):
        """Test executing a non-existent skill."""
        parameters = {
            "user_input": "test",
            "intent": "test",
        }

        result = await adapter.execute_from_tool_call("nonexistent_skill", parameters)

        assert "not found" in result.lower() or "error" in result.lower()

    def test_get_all_tools(self, adapter):
        """Test getting all tools from the registry."""
        tools = adapter.get_all_tools()

        # Should have tools for all registered skills
        tool_names = {tool.name for tool in tools}
        assert "test_skill" in tool_names
        assert "app_launcher" in tool_names
        assert "calendar" in tool_names

        # All tools should have required attributes
        for tool in tools:
            assert tool.name
            assert tool.description
            # Agno Function has parameters as dict
            assert isinstance(tool.parameters, dict)

    def test_get_tools_for_request(self, adapter):
        """Test getting relevant tools for a request."""
        # Currently returns all tools
        tools = adapter.get_tools_for_request("open Safari")

        tool_names = {tool.name for tool in tools}
        assert "app_launcher" in tool_names


class TestIntentClassifier:
    """Tests for IntentClassifier."""

    @pytest.fixture
    def mock_client(self):
        """Return a mock OllamaClient."""
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = (
            '{"intent": "app_launcher", "confidence": 0.9, "parameters": {"app_name": "Safari"}}'
        )
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)
        return client

    @pytest.fixture
    def classifier(self, mock_client):
        """Return an IntentClassifier instance."""
        return IntentClassifier(
            local_client=mock_client,
            model="qwen3:0.6b",
        )

    def test_init(self, classifier, mock_client):
        """Test classifier initialization."""
        assert classifier.client == mock_client
        assert classifier.model == "qwen3:0.6b"
        assert classifier._classification_prompt

    def test_build_classification_prompt(self, classifier):
        """Test building the classification prompt."""
        prompt = classifier._classification_prompt

        assert "intent classifier" in prompt.lower()
        assert "json" in prompt.lower()
        assert "app_launcher" in prompt or "calendar" in prompt  # Should list skills

    def test_get_all_skills_info(self, classifier):
        """Test getting skill descriptions."""
        with patch.object(SkillRegistry, "get_instance") as mock_get:
            mock_registry = MagicMock()
            mock_registry.list_skills.return_value = [
                {
                    "name": "test_skill",
                    "description": "A test skill",
                    "triggers": ["test"],
                }
            ]
            mock_get.return_value = mock_registry

            info = classifier._get_all_skills_info()

            assert "test_skill" in info
            assert "A test skill" in info

    @pytest.mark.asyncio
    async def test_classify_success(self, classifier, mock_client):
        """Test successful intent classification."""
        result = await classifier.classify("open Safari")

        assert result["intent"] == "app_launcher"
        assert result["confidence"] == 0.9
        assert "app_name" in result["parameters"]
        assert result["parameters"]["app_name"] == "Safari"

    @pytest.mark.asyncio
    async def test_classify_json_error(self, classifier, mock_client):
        """Test classification with JSON parse error."""
        # Return invalid JSON
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not valid JSON"
        mock_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await classifier.classify("test input")

        # Should fall back to general_conversation
        assert result["intent"] == "general_conversation"
        assert result["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_classify_api_error(self, classifier, mock_client):
        """Test classification with API error."""
        mock_client._client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        result = await classifier.classify("test input")

        # Should fall back to general_conversation
        assert result["intent"] == "general_conversation"
        assert result["confidence"] == 0.5
        assert "user_input" in result["parameters"]

    @pytest.mark.asyncio
    async def test_classify_confidence_clamping(self, classifier, mock_client):
        """Test that confidence values are clamped to [0, 1]."""
        # Return confidence > 1
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"intent": "test", "confidence": 1.5, "parameters": {}}'
        mock_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await classifier.classify("test")

        assert result["confidence"] == 1.0  # Clamped to max

    @pytest.mark.asyncio
    async def test_classify_negative_confidence(self, classifier, mock_client):
        """Test that negative confidence values are clamped."""
        # Return confidence < 0
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"intent": "test", "confidence": -0.5, "parameters": {}}'
        mock_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await classifier.classify("test")

        assert result["confidence"] == 0.0  # Clamped to min

    @pytest.mark.asyncio
    async def test_classify_with_partial_json(self, classifier, mock_client):
        """Test classification with extra text around JSON."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = (
            'Here is the result: {"intent": "test", "confidence": 0.7, "parameters": {}}'
        )
        mock_client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await classifier.classify("test")

        assert result["intent"] == "test"
        assert result["confidence"] == 0.7


class TestSkillCallStats:
    """Tests for SkillCallStats dataclass."""

    def test_initial_state(self):
        """Test initial state of stats."""
        from roxy.brain.tool_adapter import SkillCallStats

        stats = SkillCallStats()

        assert stats.total_calls == 0
        assert stats.failures == 0
        assert stats.consecutive_failures == 0
        assert stats.last_failure is None
        assert stats.last_success is None

    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        from roxy.brain.tool_adapter import SkillCallStats

        stats = SkillCallStats()
        assert stats.failure_rate == 0.0

        stats.total_calls = 10
        stats.failures = 3
        assert stats.failure_rate == 0.3

    def test_is_healthy_with_no_failures(self):
        """Test that new stats are healthy."""
        from roxy.brain.tool_adapter import SkillCallStats

        stats = SkillCallStats()
        assert stats.is_healthy is True

    def test_circuit_breaker_trips_on_consecutive_failures(self):
        """Test that circuit breaker trips after 5 consecutive failures."""

        from roxy.brain.tool_adapter import SkillCallStats

        stats = SkillCallStats()
        stats.total_calls = 10
        stats.failures = 6
        stats.consecutive_failures = 5

        assert stats.is_healthy is False

    def test_circuit_breaker_trips_on_high_failure_rate(self):
        """Test that circuit breaker trips on high failure rate."""
        from datetime import datetime, timedelta

        from roxy.brain.tool_adapter import SkillCallStats

        stats = SkillCallStats()
        stats.total_calls = 10
        stats.failures = 8
        stats.consecutive_failures = 4
        stats.last_failure = datetime.now() - timedelta(seconds=30)

        # High failure rate + recent failure = unhealthy
        assert stats.is_healthy is False


class TestHardenedSkillToolAdapter:
    """Tests for hardened SkillToolAdapter features."""

    @pytest.fixture
    def registry(self) -> SkillRegistry:
        """Return a test skill registry."""
        SkillRegistry.reset()
        registry = SkillRegistry.get_instance()
        registry.register(DummyTestSkill)
        return registry

    @pytest.fixture
    def mock_orchestrator(self, mock_config):
        """Return a mock orchestrator."""
        orchestrator = MagicMock()
        orchestrator.config = mock_config
        orchestrator.memory = MagicMock()
        orchestrator.local_client = MagicMock()
        orchestrator.privacy = MagicMock()
        orchestrator.conversation_history = []
        return orchestrator

    @pytest.fixture
    def adapter(self, registry, mock_orchestrator) -> SkillToolAdapter:
        """Return a SkillToolAdapter instance."""
        return SkillToolAdapter(
            skill_registry=registry,
            orchestrator_context=mock_orchestrator,
            skill_timeout=5.0,  # Short timeout for testing
        )

    def test_permission_info_in_tool_description(self, adapter):
        """Test that permission requirements are added to tool descriptions."""
        skill = DummyAppLauncherSkill()
        tool = adapter.skill_to_tool(skill)

        assert "permissions" in tool.description.lower()
        assert "applescript" in tool.description.lower()

    def test_skill_name_validation_raises_on_invalid_name(self, adapter):
        """Test that invalid skill names raise ValueError."""

        # Create a valid skill then modify name to be invalid
        # (can't create with invalid name directly due to RoxySkill.__init__ validation)
        class InvalidSkill(RoxySkill):
            name = "temp_name"
            description = "Invalid"
            triggers = []
            permissions = []

            async def execute(self, context):
                return SkillResult(success=True, response_text="")

        skill = InvalidSkill()
        skill.name = ""  # Bypass __post_init__ by modifying after creation

        with pytest.raises(ValueError, match="Invalid skill name"):
            adapter.skill_to_tool(skill)

    @pytest.mark.asyncio
    async def test_timeout_protection(self, adapter, mock_orchestrator):
        """Test that skills are protected from hanging indefinitely."""
        import asyncio

        class HangingSkill(RoxySkill):
            name = "hanging"
            description = "A skill that hangs"
            triggers = ["hang"]
            permissions = []

            async def execute(self, context):
                await asyncio.sleep(100)  # Hang for 100 seconds
                return SkillResult(success=True, response_text="Done")

        # Register hanging skill
        adapter.registry.register(HangingSkill)

        # Should timeout with short timeout
        result = await adapter.execute_from_tool_call("hanging", {"user_input": "hang"})

        assert "timeout" in result.lower()
        assert "5.0s" in result or "timeout" in result.lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_disables_failing_skill(self, adapter, mock_orchestrator):
        """Test that circuit breaker disables skills after repeated failures."""

        class FailingSkill(RoxySkill):
            name = "failing"
            description = "A skill that always fails"
            triggers = ["fail"]
            permissions = []

            async def execute(self, context):
                raise RuntimeError("Always fails")

        # Register failing skill
        adapter.registry.register(FailingSkill)

        # Fail 5 times consecutively
        for _ in range(5):
            result = await adapter.execute_from_tool_call("failing", {"user_input": "fail"})
            assert "error" in result.lower()

        # 6th call should be blocked by circuit breaker
        result = await adapter.execute_from_tool_call("failing", {"user_input": "fail"})
        assert "disabled" in result.lower() or "unavailable" in result.lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_allows_manual_reset(self, adapter, mock_orchestrator):
        """Test that skills can be manually re-enabled after circuit breaker trip."""

        class FailingSkill(RoxySkill):
            name = "temporarily_broken"
            description = "A skill that temporarily fails"
            triggers = ["break"]
            permissions = []

            async def execute(self, context):
                raise RuntimeError("Temporary failure")

        # Register the skill
        adapter.registry.register(FailingSkill)

        # Trip the circuit breaker
        for _ in range(5):
            await adapter.execute_from_tool_call("temporarily_broken", {"user_input": "break"})

        # Verify it's disabled
        stats = adapter.get_skill_stats("temporarily_broken")
        assert stats["is_healthy"] is False

        # Manually reset
        adapter.reset_skill_stats("temporarily_broken")

        # Should be healthy again
        stats = adapter.get_skill_stats("temporarily_broken")
        assert stats["is_healthy"] is True
        assert stats["total_calls"] == 0
        assert stats["failures"] == 0

    def test_get_skill_stats(self, adapter):
        """Test getting skill statistics."""
        # Stats for a skill that hasn't been executed yet should be empty
        stats = adapter.get_skill_stats("nonexistent_skill")
        assert stats == {}

        # Getting stats with no argument should return all stats (empty dict if none)
        all_stats = adapter.get_skill_stats()
        assert isinstance(all_stats, dict)

    def test_get_all_skills_stats(self, adapter):
        """Test getting stats for all skills."""
        # Initially, no stats are collected
        all_stats = adapter.get_skill_stats()
        assert isinstance(all_stats, dict)

        # After executing a skill, stats should be collected
        import asyncio

        async def run_test():
            result = await adapter.execute_from_tool_call(
                "test_skill",
                {"user_input": "test", "intent": "test"},
            )
            return result

        # Run sync (the skill is very simple)
        asyncio.run(run_test())

        # Now test_skill should have stats
        all_stats = adapter.get_skill_stats()
        assert "test_skill" in all_stats
        assert all_stats["test_skill"]["total_calls"] == 1

    @pytest.mark.asyncio
    async def test_unhealthy_skills_excluded_from_tools(self, adapter, mock_orchestrator):
        """Test that unhealthy skills are not included in tool list."""

        class FlakySkill(RoxySkill):
            name = "flaky"
            description = "A skill that often fails"
            triggers = ["flaky"]
            permissions = []

            async def execute(self, context):
                raise RuntimeError("Flaky")

        # Register and trip circuit breaker
        adapter.registry.register(FlakySkill)
        for _ in range(5):
            await adapter.execute_from_tool_call("flaky", {"user_input": "flaky"})

        # Get all tools - should exclude unhealthy skill
        tools = adapter.get_all_tools()
        tool_names = {tool.name for tool in tools}
        assert "flaky" not in tool_names


class TestHardenedIntentClassifier:
    """Tests for hardened IntentClassifier features."""

    @pytest.fixture
    def mock_client(self):
        """Return a mock OllamaClient."""
        client = MagicMock()
        return client

    @pytest.fixture
    def classifier(self, mock_client):
        """Return an IntentClassifier instance."""
        return IntentClassifier(
            local_client=mock_client,
            model="qwen3:0.6b",
        )

    @pytest.mark.asyncio
    async def test_input_validation_with_none(self, classifier):
        """Test classification with None input."""
        result = await classifier.classify(None)

        assert result["intent"] == "general_conversation"
        assert result["confidence"] == 0.5
        # None is converted to the string "None" by str(None)
        assert result["parameters"]["user_input"] == "None"

    @pytest.mark.asyncio
    async def test_input_validation_with_empty_string(self, classifier):
        """Test classification with empty string input."""
        result = await classifier.classify("")

        assert result["intent"] == "general_conversation"
        assert result["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_extract_json_from_code_block(self, classifier):
        """Test JSON extraction from markdown code blocks."""

        # Test with code block
        content = '```json\n{"intent": "test", "confidence": 0.8}\n```'
        result = classifier._extract_json(content)

        assert result is not None
        assert result["intent"] == "test"
        assert result["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_extract_json_from_nested_content(self, classifier):
        """Test JSON extraction from content with nested braces."""

        # Test with extra text before JSON
        content = 'Here is the result: {"intent": "app_launcher", "confidence": 0.9, "parameters": {"app_name": "Safari"}}'
        result = classifier._extract_json(content)

        assert result is not None
        assert result["intent"] == "app_launcher"

    @pytest.mark.asyncio
    async def test_extract_json_with_multiple_brace_levels(self, classifier):
        """Test JSON extraction from deeply nested JSON."""

        # Test with nested JSON (2 levels of braces)
        content = '{"intent": "test", "parameters": {"nested": {"value": 5}}}'
        result = classifier._extract_json(content)

        assert result is not None
        assert result["intent"] == "test"

    @pytest.mark.asyncio
    async def test_extract_json_returns_none_on_invalid(self, classifier):
        """Test that _extract_json returns None for invalid JSON."""

        result = classifier._extract_json("This is definitely not JSON content")
        assert result is None

        result = classifier._extract_json("")
        assert result is None

        result = classifier._extract_json("{incomplete json")
        assert result is None
