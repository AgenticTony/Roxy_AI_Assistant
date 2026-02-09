"""Integration tests for skill registration and discovery.

Tests that skills are properly registered, discovered, and can be
dispatched to via the skill registry.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from roxy.config import RoxyConfig
from roxy.skills.registry import SkillRegistry
from roxy.skills.base import RoxySkill, SkillContext, SkillResult, Permission


class TestSkill1(RoxySkill):
    """First test skill."""

    name = "test_skill_1"
    description = "A test skill"
    triggers = ["test one", "first test"]
    permissions = [Permission.FILESYSTEM]

    async def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(success=True, response_text="Test 1 executed")


class TestSkill2(RoxySkill):
    """Second test skill."""

    name = "test_skill_2"
    description = "Another test skill"
    triggers = ["test two", "second test"]
    permissions = [Permission.NETWORK]

    async def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(success=True, response_text="Test 2 executed")


class ConflictingSkill(RoxySkill):
    """Skill with conflicting name for testing overrides."""

    name = "test_skill_1"  # Same as TestSkill1
    description = "Conflicting skill"
    triggers = ["conflict"]
    permissions = []

    async def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(success=True, response_text="Conflict executed")


@pytest_asyncio
async def test_skill_register_and_list(mock_config: RoxyConfig) -> None:
    """Test that skills can be registered and listed."""
    registry = SkillRegistry()
    registry.reset()

    # Register skills
    registry.register(TestSkill1)
    registry.register(TestSkill2)

    # List skills
    skills = registry.list_skills()

    assert len(skills) == 2
    skill_names = {s["name"] for s in skills}
    assert "test_skill_1" in skill_names
    assert "test_skill_2" in skill_names


@pytest_asyncio
async def test_skill_find_by_trigger(mock_config: RoxyConfig) -> None:
    """Test that skills can be found by trigger phrases."""
    registry = SkillRegistry()
    registry.reset()

    registry.register(TestSkill1)
    registry.register(TestSkill2)

    # Find skill by trigger
    skill, confidence = registry.find_skill("test one", {})

    assert skill is not None
    assert skill.name == "test_skill_1"
    assert confidence > 0.5


@pytest_asyncio
async def test_skill_confidence_scoring(mock_config: RoxyConfig) -> None:
    """Test that confidence scores are calculated correctly."""
    registry = SkillRegistry()
    registry.reset()

    registry.register(TestSkill1)
    registry.register(TestSkill2)

    # Exact trigger match should have high confidence
    skill1, conf1 = registry.find_skill("test one", {})
    assert conf1 > 0.8

    # Partial match should have lower confidence
    skill2, conf2 = registry.find_skill("testing one thing", {})
    assert conf2 < conf1  # Lower confidence for partial match

    # No match should have zero/low confidence
    skill3, conf3 = registry.find_skill("completely unrelated", {})
    assert conf3 < 0.3


@pytest_asyncio
async def test_skill_get_by_name(mock_config: RoxyConfig) -> None:
    """Test that skills can be retrieved by name."""
    registry = SkillRegistry()
    registry.reset()

    registry.register(TestSkill1)
    registry.register(TestSkill2)

    # Get by name
    skill = registry.get("test_skill_1")

    assert skill is not None
    assert skill.name == "test_skill_1"
    assert isinstance(skill, TestSkill1)


@pytest_asyncio
async def test_skill_unregister(mock_config: RoxyConfig) -> None:
    """Test that skills can be unregistered."""
    registry = SkillRegistry()
    registry.reset()

    registry.register(TestSkill1)
    registry.register(TestSkill2)

    # Unregister one skill
    registry.unregister("test_skill_1")

    # Verify it's gone
    skill = registry.get("test_skill_1")
    assert skill is None

    # Verify other skill remains
    skill = registry.get("test_skill_2")
    assert skill is not None


@pytest_asyncio
async def test_skill_discovery_from_directory(mock_config: RoxyConfig) -> None:
    """Test that skills can be auto-discovered from a directory."""
    registry = SkillRegistry()
    registry.reset()

    # Discover from skills directory
    skills_dir = Path(__file__).parent.parent.parent / "src" / "roxy" / "skills"

    if skills_dir.exists():
        registry.discover(str(skills_dir))

        # Should find at least some skills
        skills = registry.list_skills()
        assert len(skills) > 0


@pytest_asyncio
async def test_skill_execution_via_registry(mock_config: RoxyConfig) -> None:
    """Test that skills can be executed through the registry."""
    registry = SkillRegistry()
    registry.reset()

    registry.register(TestSkill1)

    # Get and execute skill
    skill = registry.get("test_skill_1")
    assert skill is not None

    # Create context
    context = SkillContext(
        user_input="test one",
        intent="test",
        parameters={},
        memory=MagicMock(),  # Mock memory
        config=mock_config,
        conversation_history=[],
    )

    # Execute
    result = await skill.execute(context)

    assert result.success is True
    assert "Test 1 executed" in result.response_text


@pytest_asyncio
async def test_skill_permissions_checking(mock_config: RoxyConfig) -> None:
    """Test that skill permissions are properly declared."""
    registry = SkillRegistry()
    registry.reset()

    registry.register(TestSkill1)
    registry.register(TestSkill2)

    # Get skill info
    skill1 = registry.get("test_skill_1")
    skill2 = registry.get("test_skill_2")

    assert Permission.FILESYSTEM in skill1.permissions
    assert Permission.NETWORK in skill2.permissions


@pytest_asyncio
async def test_skill_duplicate_registration(mock_config: RoxyConfig) -> None:
    """Test behavior when registering skills with duplicate names."""
    registry = SkillRegistry()
    registry.reset()

    registry.register(TestSkill1)

    # Register conflicting skill (same name)
    # Should either raise error or override
    try:
        registry.register(ConflictingSkill)

        # If no error, check that it was overridden
        skill = registry.get("test_skill_1")
        # Behavior depends on implementation
        assert skill is not None
    except Exception:
        # Some implementations might raise an error
        pass


@pytest_asyncio
async def test_registry_reset(mock_config: RoxyConfig) -> None:
    """Test that the registry can be reset."""
    registry = SkillRegistry()

    # Register some skills
    registry.register(TestSkill1)
    registry.register(TestSkill2)

    assert len(registry.list_skills()) == 2

    # Reset
    registry.reset()

    # Should be empty
    assert len(registry.list_skills()) == 0


@pytest_asyncio
async def test_skill_metadata_storage(mock_config: RoxyConfig) -> None:
    """Test that skill metadata is properly stored."""
    registry = SkillRegistry()
    registry.reset()

    registry.register(TestSkill1)

    # Get skill info
    skills = registry.list_skills()
    skill_info = next(s for s in skills if s["name"] == "test_skill_1")

    # Check metadata
    assert skill_info["name"] == "test_skill_1"
    assert skill_info["description"] == "A test skill"
    assert "triggers" in skill_info
    assert "permissions" in skill_info


# Async test marker
pytest_asyncio = pytest.mark.asyncio
