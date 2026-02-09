"""Roxy skills module.

Contains the base skill class and all skill implementations.
"""

from roxy.skills.base import (
    Permission,
    SkillContext,
    SkillResult,
    MemoryManager,
    StubMemoryManager,
    RoxySkill,
)

# Skill registry and permissions
from roxy.skills.registry import SkillRegistry, get_registry
from roxy.skills.permissions import PermissionManager, get_permission_manager

# Web skills
from roxy.skills.web import WebSearchSkill, BrowseSkill, FlightSearchSkill

# Productivity skills
from roxy.skills.productivity import (
    CalendarSkill,
    EmailSkill,
    NotesSkill,
    RemindersSkill,
)

# Developer skills
from roxy.skills.dev import GitOpsSkill, ClaudeCodeSkill

# System skills
from roxy.skills.system.app_launcher import AppLauncherSkill
from roxy.skills.system.file_search import FileSearchSkill
from roxy.skills.system.window_manager import WindowManagerSkill
from roxy.skills.system.system_info import SystemInfoSkill
from roxy.skills.system.clipboard import ClipboardSkill
from roxy.skills.system.shortcuts import ShortcutsSkill

__all__ = [
    # Base classes
    "Permission",
    "SkillContext",
    "SkillResult",
    "MemoryManager",
    "StubMemoryManager",
    "RoxySkill",
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
    # System skills
    "AppLauncherSkill",
    "FileSearchSkill",
    "WindowManagerSkill",
    "SystemInfoSkill",
    "ClipboardSkill",
    "ShortcutsSkill",
]
