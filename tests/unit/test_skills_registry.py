"""Unit tests for skill registry and permissions."""

from __future__ import annotations

import pytest

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult
from roxy.skills.registry import SkillRegistry, get_registry
from roxy.skills.permissions import PermissionManager, get_permission_manager


class MockSkill(RoxySkill):
    """Mock skill for testing."""

    name = "mock_skill"
    description = "A mock skill for testing"
    triggers = ["test", "mock"]
    permissions = [Permission.APPLESCRIPT]
    requires_cloud = False

    async def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(success=True, response_text="Mock executed")


class AnotherMockSkill(RoxySkill):
    """Another mock skill for testing."""

    name = "another_mock"
    description = "Another mock skill"
    triggers = ["another"]
    permissions = [Permission.FILESYSTEM]
    requires_cloud = False

    async def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(success=True, response_text="Another mock executed")


class TestSkillRegistry:
    """Test SkillRegistry functionality."""

    def setup_method(self):
        """Reset registry before each test."""
        SkillRegistry.reset()

    def test_register_skill(self):
        """Test registering a skill."""
        registry = SkillRegistry()
        registry.register(MockSkill)

        skills = registry.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "mock_skill"

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate skills raises error."""
        registry = SkillRegistry()
        registry.register(MockSkill)

        with pytest.raises(TypeError, match="already registered"):
            registry.register(MockSkill)

    def test_register_invalid_skill_raises_error(self):
        """Test that registering non-RoxySkill class raises error."""
        registry = SkillRegistry()

        class NotASkill:
            pass

        with pytest.raises(ValueError, match="must be a RoxySkill subclass"):
            registry.register(NotASkill)

    def test_get_skill(self):
        """Test getting a skill by name."""
        registry = SkillRegistry()
        registry.register(MockSkill)

        skill = registry.get_skill("mock_skill")
        assert skill is not None
        assert skill.name == "mock_skill"

    def test_get_skill_not_found(self):
        """Test getting a non-existent skill returns None."""
        registry = SkillRegistry()
        skill = registry.get_skill("nonexistent")
        assert skill is None

    def test_find_skill_by_intent(self):
        """Test finding a skill by intent."""
        registry = SkillRegistry()
        registry.register(MockSkill)
        registry.register(AnotherMockSkill)

        # Test direct trigger match
        skill, confidence = registry.find_skill("test this", {})
        assert skill is not None
        assert skill.name == "mock_skill"
        assert confidence > 0.5

        # Test another trigger
        skill, confidence = registry.find_skill("another test", {})
        assert skill is not None
        assert skill.name == "another_mock"

    def test_find_skill_no_match(self):
        """Test finding skill when no match exists."""
        registry = SkillRegistry()
        registry.register(MockSkill)

        skill, confidence = registry.find_skill("completely unrelated", {})
        assert skill is None
        assert confidence == 0.0

    def test_list_skills(self):
        """Test listing all registered skills."""
        registry = SkillRegistry()
        registry.register(MockSkill)
        registry.register(AnotherMockSkill)

        skills = registry.list_skills()
        assert len(skills) == 2

        # Check skill metadata
        mock_skill = next((s for s in skills if s["name"] == "mock_skill"), None)
        assert mock_skill is not None
        assert mock_skill["description"] == "A mock skill for testing"
        assert "applescript" in mock_skill["permissions"]
        assert mock_skill["requires_cloud"] is False

    def test_singleton(self):
        """Test that get_registry returns singleton instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_clear(self):
        """Test clearing the registry."""
        registry = SkillRegistry()
        registry.register(MockSkill)
        registry.register(AnotherMockSkill)

        assert len(registry.list_skills()) == 2

        registry.clear()

        assert len(registry.list_skills()) == 0


class TestPermissionManager:
    """Test PermissionManager functionality."""

    def test_check_no_permissions_required(self):
        """Test checking skill with no permissions required."""
        class NoPermsSkill(RoxySkill):
            name = "no_perms"
            description = "Skill with no permissions"
            triggers = []
            permissions = []
            requires_cloud = False

            async def execute(self, context: SkillContext) -> SkillResult:
                return SkillResult(success=True, response_text="Done")

        manager = PermissionManager()
        skill = NoPermsSkill()

        # Should return True even with no granted permissions
        assert manager.check(skill) is True

    def test_grant_and_check(self):
        """Test granting and checking permissions."""
        manager = PermissionManager()
        skill = MockSkill()

        # Initially should not have permission
        assert manager.check(skill) is False

        # Grant permission
        manager.grant("mock_skill", Permission.APPLESCRIPT)

        # Now should have permission
        assert manager.check(skill) is True

    def test_revoke(self):
        """Test revoking permissions."""
        manager = PermissionManager()
        skill = MockSkill()

        # Grant permission
        manager.grant("mock_skill", Permission.APPLESCRIPT)
        assert manager.check(skill) is True

        # Revoke permission
        manager.revoke("mock_skill", Permission.APPLESCRIPT)
        assert manager.check(skill) is False

    def test_get_granted_permissions(self):
        """Test getting granted permissions."""
        manager = PermissionManager()

        # Grant multiple permissions
        manager.grant("test_skill", Permission.APPLESCRIPT)
        manager.grant("test_skill", Permission.FILESYSTEM)

        perms = manager.get_granted_permissions("test_skill")
        assert Permission.APPLESCRIPT in perms
        assert Permission.FILESYSTEM in perms
        assert len(perms) == 2

    def test_revoke_all(self):
        """Test revoking all permissions for a skill."""
        manager = PermissionManager()
        skill = MockSkill()

        manager.grant("mock_skill", Permission.APPLESCRIPT)
        manager.grant("mock_skill", Permission.FILESYSTEM)

        assert manager.check(skill) is True

        manager.revoke_all("mock_skill")

        assert manager.check(skill) is False

    def test_list_permissions(self):
        """Test listing all permissions."""
        manager = PermissionManager()

        manager.grant("skill1", Permission.APPLESCRIPT)
        manager.grant("skill2", Permission.FILESYSTEM)

        perms = manager.list_permissions()
        assert "skill1" in perms
        assert "skill2" in perms
        assert "applescript" in perms["skill1"]
        assert "filesystem" in perms["skill2"]

    def test_singleton(self):
        """Test that get_permission_manager returns singleton instance."""
        manager1 = get_permission_manager()
        manager2 = get_permission_manager()

        assert manager1 is manager2
