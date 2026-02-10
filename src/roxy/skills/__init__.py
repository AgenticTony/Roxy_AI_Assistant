"""Roxy skills module.

Contains the base skill class and all skill implementations.
"""

from roxy.skills.base import (
    HookType,
    MemoryManager,
    Permission,
    RoxySkill,
    SkillContext,
    SkillResult,
    StubMemoryManager,
    clear_hooks,
    get_hooks,
    register_hook,
)

# Developer skills
from roxy.skills.dev import ClaudeCodeSkill, GitOpsSkill, ProjectManagerSkill
from roxy.skills.permissions import PermissionManager, get_permission_manager

# Productivity skills
from roxy.skills.productivity import (
    CalendarSkill,
    EmailSkill,
    NotesSkill,
    RemindersSkill,
)

# Skill registry and permissions
from roxy.skills.registry import SkillRegistry, get_registry

# System skills
from roxy.skills.system.app_launcher import AppLauncherSkill
from roxy.skills.system.clipboard import ClipboardSkill
from roxy.skills.system.file_search import FileSearchSkill
from roxy.skills.system.shortcuts import ShortcutsSkill
from roxy.skills.system.system_info import SystemInfoSkill
from roxy.skills.system.window_manager import WindowManagerSkill

# Web skills
from roxy.skills.web import BrowseSkill, FlightSearchSkill, WebSearchSkill

__all__ = [
    # Base classes
    "Permission",
    "SkillContext",
    "SkillResult",
    "MemoryManager",
    "StubMemoryManager",
    "RoxySkill",
    # Lifecycle hooks
    "HookType",
    "register_hook",
    "clear_hooks",
    "get_hooks",
    # Registry and permissions
    "SkillRegistry",
    "get_registry",
    "PermissionManager",
    "get_permission_manager",
    # Web skills
    "WebSearchSkill",
    "BrowseSkill",
    "FlightSearchSkill",
    # Productivity skills
    "CalendarSkill",
    "EmailSkill",
    "NotesSkill",
    "RemindersSkill",
    # Developer skills
    "GitOpsSkill",
    "ClaudeCodeSkill",
    "ProjectManagerSkill",
    # System skills
    "AppLauncherSkill",
    "FileSearchSkill",
    "WindowManagerSkill",
    "SystemInfoSkill",
    "ClipboardSkill",
    "ShortcutsSkill",
]
