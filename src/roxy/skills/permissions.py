"""Permission management for Roxy skills.

Handles permission checks, user prompts, and persistent storage of granted permissions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.prompt import Confirm

from roxy.skills.base import Permission, RoxySkill

if TYPE_CHECKING:
    from roxy.config import RoxyConfig

logger = logging.getLogger(__name__)

# Rich console for prompts
console = Console()


class PermissionManager:
    """
    Manages permissions for Roxy skills.

    Tracks granted permissions per skill and provides UI for requesting
    permissions from users. Persists state to a JSON file.
    """

    def __init__(self, config: RoxyConfig | None = None) -> None:
        """
        Initialize the permission manager.

        Args:
            config: Roxy configuration. If None, will attempt to load default.
        """
        if config is None:
            from roxy.config import RoxyConfig

            config = RoxyConfig.load()

        self._config = config
        self._permissions_file = Path(config.data_dir) / "permissions.json"
        self._granted: dict[str, set[Permission]] = {}

        # Load existing permissions
        self._load_permissions()

        logger.info(f"PermissionManager initialized with {len(self._granted)} skill permissions")

    def _load_permissions(self) -> None:
        """Load granted permissions from JSON file."""
        if not self._permissions_file.exists():
            logger.debug(f"Permissions file does not exist: {self._permissions_file}")
            return

        try:
            with self._permissions_file.open("r") as f:
                data = json.load(f)

            # Convert string permissions back to Permission enum
            for skill_name, perm_list in data.items():
                self._granted[skill_name] = {Permission(p) for p in perm_list}

            logger.debug(f"Loaded permissions for {len(self._granted)} skills")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode permissions file: {e}")
        except Exception as e:
            logger.error(f"Failed to load permissions: {e}")

    def _save_permissions(self) -> None:
        """Save granted permissions to JSON file."""
        try:
            # Ensure data directory exists
            self._permissions_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert Permission enum to strings
            data = {
                skill_name: [p.value for p in perm_set]
                for skill_name, perm_set in self._granted.items()
            }

            with self._permissions_file.open("w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved permissions for {len(self._granted)} skills")

        except Exception as e:
            logger.error(f"Failed to save permissions: {e}")

    def check(self, skill: RoxySkill) -> bool:
        """
        Check if a skill has all required permissions granted.

        Args:
            skill: The skill to check.

        Returns:
            True if all required permissions are granted.
        """
        if not skill.permissions:
            # No permissions required
            return True

        granted = self._granted.get(skill.name, set())
        has_all = all(p in granted for p in skill.permissions)

        if not has_all:
            missing = [p.value for p in skill.permissions if p not in granted]
            logger.debug(f"Skill {skill.name} missing permissions: {missing}")

        return has_all

    def request(self, skill: RoxySkill) -> bool:
        """
        Request permissions from the user for a skill.

        Prompts the user via CLI for each missing permission using Click/rich
        for consistency with main.py.

        Args:
            skill: The skill requesting permissions.

        Returns:
            True if user granted all permissions, False otherwise.
        """
        if not skill.permissions:
            # No permissions required
            return True

        # Check which permissions are already granted
        currently_granted = self._granted.get(skill.name, set())
        missing = [p for p in skill.permissions if p not in currently_granted]

        if not missing:
            # All permissions already granted
            return True

        # Display skill information
        console.print()
        console.print(f"[bold cyan]Skill: {skill.name}[/bold cyan]\n{skill.description}")

        # Request each missing permission
        for permission in missing:
            granted = self._prompt_permission(skill.name, permission)
            if not granted:
                console.print(f"[yellow]Permission denied: {permission.value}[/yellow]")
                return False

        # All permissions granted
        return True

    def _prompt_permission(self, skill_name: str, permission: Permission) -> bool:
        """
        Prompt user for a single permission.

        Args:
            skill_name: Name of the skill requesting permission.
            permission: The permission being requested.

        Returns:
            True if user granted the permission.
        """
        # Get permission description
        description = self._get_permission_description(permission)

        # Prompt using rich/Click
        console.print()
        console.print(
            f"[bold yellow]Permission Request[/bold yellow]\n"
            f"  Skill: {skill_name}\n"
            f"  Permission: [bold]{permission.value}[/bold]\n"
            f"  Description: {description}\n"
        )

        try:
            result = Confirm.ask(
                "[bold green]Grant this permission?[/bold green]",
                console=console,
                default=False,
            )

            if result:
                # Grant the permission
                if skill_name not in self._granted:
                    self._granted[skill_name] = set()

                self._granted[skill_name].add(permission)
                self._save_permissions()

                console.print(f"[green]✓[/green] Permission granted: {permission.value}")
                logger.info(f"Permission {permission.value} granted for {skill_name}")
            else:
                console.print(f"[red]✗[/red] Permission denied: {permission.value}")
                logger.info(f"Permission {permission.value} denied for {skill_name}")

            return result

        except Exception as e:
            logger.error(f"Error prompting for permission: {e}")
            return False

    def _get_permission_description(self, permission: Permission) -> str:
        """
        Get a human-readable description for a permission.

        Args:
            permission: The permission to describe.

        Returns:
            Description string.
        """
        descriptions = {
            Permission.FILESYSTEM: "Access to read and write files on your system",
            Permission.NETWORK: "Access to make network requests and external API calls",
            Permission.SHELL: "Ability to execute shell commands and scripts",
            Permission.MICROPHONE: "Access to record audio from the microphone",
            Permission.NOTIFICATIONS: "Ability to display system notifications",
            Permission.APPLESCRIPT: "Ability to execute AppleScript commands (macOS automation)",
            Permission.CLOUD_LLM: "Permission to send data to external cloud AI services",
            Permission.CALENDAR: "Access to read your calendar events",
            Permission.EMAIL: "Access to read and send emails",
            Permission.CONTACTS: "Access to read your contacts",
        }

        return descriptions.get(permission, "Unknown permission")

    def grant(self, skill_name: str, permission: Permission) -> None:
        """
        Grant a permission to a skill.

        Args:
            skill_name: Name of the skill.
            permission: Permission to grant.
        """
        if skill_name not in self._granted:
            self._granted[skill_name] = set()

        if permission not in self._granted[skill_name]:
            self._granted[skill_name].add(permission)
            self._save_permissions()
            logger.info(f"Granted {permission.value} to {skill_name}")

    def revoke(self, skill_name: str, permission: Permission) -> None:
        """
        Revoke a permission from a skill.

        Args:
            skill_name: Name of the skill.
            permission: Permission to revoke.
        """
        if skill_name in self._granted and permission in self._granted[skill_name]:
            self._granted[skill_name].remove(permission)
            self._save_permissions()
            logger.info(f"Revoked {permission.value} from {skill_name}")

            # Clean up empty skill entries
            if not self._granted[skill_name]:
                del self._granted[skill_name]

    def get_granted_permissions(self, skill_name: str) -> set[Permission]:
        """
        Get all granted permissions for a skill.

        Args:
            skill_name: Name of the skill.

        Returns:
            Set of granted permissions.
        """
        return self._granted.get(skill_name, set()).copy()

    def revoke_all(self, skill_name: str) -> None:
        """
        Revoke all permissions for a skill.

        Args:
            skill_name: Name of the skill.
        """
        if skill_name in self._granted:
            del self._granted[skill_name]
            self._save_permissions()
            logger.info(f"Revoked all permissions for {skill_name}")

    def reset(self) -> None:
        """Reset all permissions (clear all granted permissions).

        This is primarily useful for testing.
        """
        self._granted.clear()
        self._save_permissions()
        logger.info("Reset all permissions")

    def list_permissions(self) -> dict[str, list[str]]:
        """
        List all granted permissions for all skills.

        Returns:
            Dict mapping skill names to lists of permission strings.
        """
        return {
            skill_name: [p.value for p in perm_set]
            for skill_name, perm_set in self._granted.items()
        }


# Singleton instance
_permission_manager: PermissionManager | None = None


def get_permission_manager(config: RoxyConfig | None = None) -> PermissionManager:
    """
    Get or create the PermissionManager singleton.

    Args:
        config: Roxy configuration. Only used on first call.

    Returns:
        PermissionManager instance.
    """
    global _permission_manager

    if _permission_manager is None:
        _permission_manager = PermissionManager(config)

    return _permission_manager


def reset_permission_manager() -> None:
    """Reset the PermissionManager singleton.

    This is primarily useful for testing.
    """
    global _permission_manager
    _permission_manager = None
