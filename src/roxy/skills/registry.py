"""Skill registry for managing Roxy's skill system.

Provides automatic skill discovery, registration, and routing based on intent matching.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SkillRegistry:
    """
    Central registry for all Roxy skills.

    Implements singleton pattern to ensure only one registry exists.
    Skills are automatically discovered from specified directories and
    can be queried for intent matching.
    """

    _instance: SkillRegistry | None = None

    def __init__(self) -> None:
        """Initialize the skill registry.

        Raises:
            RuntimeError: If trying to create a second instance (use get_instance() instead).
        """
        if SkillRegistry._instance is not None:
            raise RuntimeError(
                "SkillRegistry is a singleton. Use SkillRegistry.get_instance() instead."
            )

        self._skills: dict[str, type] = {}
        self._skill_instances: dict[str, Any] = {}
        logger.info("SkillRegistry initialized")

    @classmethod
    def get_instance(cls) -> SkillRegistry:
        """
        Get the singleton SkillRegistry instance.

        Returns:
            The singleton SkillRegistry instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.

        This is primarily useful for testing. Use with caution.
        """
        cls._instance = None
        logger.debug("SkillRegistry reset")

    def register(self, skill: type) -> None:
        """
        Register a skill class.

        Args:
            skill: A RoxySkill subclass to register.

        Raises:
            ValueError: If skill is not a valid RoxySkill subclass.
            TypeError: If skill name already exists.
        """
        from roxy.skills.base import RoxySkill

        # Verify it's a RoxySkill subclass
        if not inspect.isclass(skill) or not issubclass(skill, RoxySkill):
            raise ValueError(f"Skill must be a RoxySkill subclass, got {skill}")

        # Check for skill name
        skill_name = getattr(skill, "name", None)
        if not skill_name:
            raise ValueError(f"Skill {skill.__name__} must define a 'name' attribute")

        # Check for duplicates
        if skill_name in self._skills:
            raise TypeError(f"Skill '{skill_name}' is already registered")

        # Register the skill class
        self._skills[skill_name] = skill
        logger.info(f"Registered skill: {skill_name} ({skill.__name__})")

    def discover(self, skills_dir: str | Path) -> None:
        """
        Auto-discover and register skills from a directory.

        Searches for Python modules in the specified directory and
        registers any RoxySkill subclasses found.

        Args:
            skills_dir: Path to directory containing skill modules.
        """
        skills_path = Path(skills_dir)

        if not skills_path.exists():
            logger.warning(f"Skills directory does not exist: {skills_path}")
            return

        if not skills_path.is_dir():
            logger.warning(f"Skills path is not a directory: {skills_path}")
            return

        # Get the parent package path
        # If skills_dir is "src/roxy/skills/system", we need to import as "roxy.skills.system.*"
        # Find the roxy package root
        root = self._find_package_root(skills_path)
        if root is None:
            logger.error(f"Could not find package root for {skills_path}")
            return

        # Convert path to module prefix
        # e.g., /path/to/roxy/src/roxy/skills/system -> roxy.skills.system
        relative_path = skills_path.relative_to(root.parent)
        module_prefix = ".".join(relative_path.parts)

        logger.debug(f"Discovering skills in {module_prefix} from {skills_path}")

        # Find all Python files in the directory
        for py_file in skills_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            # Convert file path to module name
            module_name = f"{module_prefix}.{py_file.stem}"

            try:
                # Import the module
                module = importlib.import_module(module_name)

                # Find all RoxySkill subclasses in the module
                from roxy.skills.base import RoxySkill

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Skip the base class itself
                    if obj is RoxySkill:
                        continue

                    # Check if it's a RoxySkill subclass and defined in this module
                    if issubclass(obj, RoxySkill) and obj.__module__ == module_name:
                        self.register(obj)

            except Exception as e:
                logger.error(f"Failed to import module {module_name}: {e}")
                continue

    def _find_package_root(self, path: Path) -> Path | None:
        """
        Find the root of the roxy package by looking for __init__.py files.

        Args:
            path: A path within the package.

        Returns:
            Path to the package root directory, or None if not found.
        """
        current = path
        while current != current.parent:
            init_file = current / "__init__.py"
            if init_file.exists():
                # Check if this is the roxy package
                if current.name == "roxy":
                    return current
            current = current.parent
        return None

    def find_skill(self, intent: str, parameters: dict[str, Any]) -> tuple[Any | None, float]:
        """
        Find the best matching skill for a given intent.

        Args:
            intent: The classified intent string.
            parameters: Extracted parameters from user input.

        Returns:
            Tuple of (skill_instance, confidence_score).
            Returns (None, 0.0) if no skill matches.
        """
        best_skill: Any | None = None
        best_confidence = 0.0

        for skill_name, skill_class in self._skills.items():
            # Get or create skill instance
            skill = self._skill_instances.get(skill_name)
            if skill is None:
                try:
                    skill = skill_class()
                    self._skill_instances[skill_name] = skill
                except Exception as e:
                    logger.error(f"Failed to instantiate skill {skill_name}: {e}")
                    continue

            # Check if skill can handle this intent
            try:
                confidence = skill.can_handle(intent, parameters)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_skill = skill
            except Exception as e:
                logger.error(f"Error checking skill {skill_name}: {e}")
                continue

        if best_skill is not None:
            logger.debug(
                f"Found matching skill: {best_skill.name} with confidence {best_confidence:.2f}"
            )
        else:
            logger.debug(f"No matching skill found for intent: {intent}")

        return best_skill, best_confidence

    def list_skills(self) -> list[dict[str, Any]]:
        """
        List all registered skills with their metadata.

        Returns:
            List of dicts containing skill metadata (name, description, triggers, permissions).
        """
        skills_info = []

        for skill_name, skill_class in self._skills.items():
            skill_info = {
                "name": getattr(skill_class, "name", skill_name),
                "description": getattr(skill_class, "description", ""),
                "triggers": getattr(skill_class, "triggers", []),
                "permissions": [p.value for p in getattr(skill_class, "permissions", [])],
                "requires_cloud": getattr(skill_class, "requires_cloud", False),
                "class_name": skill_class.__name__,
                "module": skill_class.__module__,
            }
            skills_info.append(skill_info)

        return skills_info

    def get_skill(self, name: str) -> Any | None:
        """
        Get a skill instance by name.

        Args:
            name: The skill name.

        Returns:
            Skill instance or None if not found.
        """
        skill_class = self._skills.get(name)
        if skill_class is None:
            return None

        # Return cached instance or create new one
        if name not in self._skill_instances:
            try:
                self._skill_instances[name] = skill_class()
            except Exception as e:
                logger.error(f"Failed to instantiate skill {name}: {e}")
                return None

        return self._skill_instances[name]

    def clear(self) -> None:
        """Clear all registered skills.

        This is primarily useful for testing.
        """
        self._skills.clear()
        self._skill_instances.clear()
        logger.debug("SkillRegistry cleared")


# Convenience functions
def get_registry() -> SkillRegistry:
    """Get the singleton SkillRegistry instance."""
    return SkillRegistry.get_instance()


def register_skill(skill: type) -> None:
    """Register a skill class."""
    registry = get_registry()
    registry.register(skill)


def discover_skills(skills_dir: str | Path) -> None:
    """Discover and register skills from a directory."""
    registry = get_registry()
    registry.discover(skills_dir)


def find_skill(intent: str, parameters: dict[str, Any]) -> tuple[Any | None, float]:
    """Find the best matching skill for an intent."""
    registry = get_registry()
    return registry.find_skill(intent, parameters)


def list_skills() -> list[dict[str, Any]]:
    """List all registered skills."""
    registry = get_registry()
    return registry.list_skills()


def get_skill(name: str) -> Any | None:
    """Get a skill instance by name."""
    registry = get_registry()
    return registry.get_skill(name)
