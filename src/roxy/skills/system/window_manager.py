"""Window manager skill using Hammerspoon.

Provides window layout management and window manipulation capabilities.
"""

from __future__ import annotations

import logging
from typing import Any

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)


# Predefined window layouts
LAYOUTS: dict[str, dict] = {
    "coding": {
        "name": "Coding Layout",
        "description": "Set up windows for coding: IDE on left, terminal on right",
        "windows": [
            {"app": "Visual Studio Code", "unit": {"x": 0, "y": 0, "w": 0.6, "h": 1.0}},
            {"app": "Terminal", "unit": {"x": 0.6, "y": 0, "w": 0.4, "h": 0.5}},
            {"app": "Safari", "unit": {"x": 0.6, "y": 0.5, "w": 0.4, "h": 0.5}},
        ],
    },
    "meeting": {
        "name": "Meeting Layout",
        "description": "Set up windows for meetings: video call centered, notes nearby",
        "windows": [
            {"app": "Zoom", "unit": {"x": 0.25, "y": 0, "w": 0.5, "h": 0.6}},
            {"app": "Notes", "unit": {"x": 0.75, "y": 0, "w": 0.25, "h": 1.0}},
        ],
    },
    "fullscreen": {
        "name": "Fullscreen",
        "description": "Maximize the frontmost window",
        "windows": [],  # Special case - maximize current window
    },
    "split": {
        "name": "Split Screen",
        "description": "Split screen between two apps",
        "windows": [
            {"app": None, "unit": {"x": 0, "y": 0, "w": 0.5, "h": 1.0}},  # Current app
            {"app": None, "unit": {"x": 0.5, "y": 0, "w": 0.5, "h": 1.0}},  # Will be second app
        ],
    },
    "focus": {
        "name": "Focus Mode",
        "description": "Center the frontmost window",
        "windows": [
            {"app": None, "unit": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}},
        ],
    },
}


class WindowManagerSkill(RoxySkill):
    """Skill for managing windows using Hammerspoon."""

    name = "window_manager"
    description = "Manage windows and layouts using Hammerspoon"
    triggers = [
        "coding layout",
        "meeting layout",
        "set up my workspace",
        "arrange windows",
        "window layout",
        "fullscreen",
        "split screen",
        "focus mode",
    ]
    permissions = [Permission.SHELL]
    requires_cloud = False

    def __init__(self) -> None:
        """Initialize window manager skill."""
        super().__init__()
        self._hammerspoon_available: bool | None = None

    async def execute(self, context: SkillContext) -> SkillResult:
        """
        Execute the window manager skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult indicating success/failure.
        """
        user_input = context.user_input.lower()
        parameters = context.parameters

        # Determine which layout to apply
        layout_name = self._determine_layout(user_input, parameters)

        if not layout_name:
            return SkillResult(
                success=False,
                response_text="Which layout would you like me to set up? Available: coding, meeting, fullscreen, split, focus",
                speak=False,
            )

        logger.info(f"Applying window layout: {layout_name}")

        # Check if Hammerspoon is available
        if not await self._check_hammerspoon():
            # Fallback to AppleScript-based window management
            return await self._apply_layout_applescript(layout_name, user_input)

        # Apply layout using Hammerspoon
        success = await self._apply_layout_hammerspoon(layout_name)

        if success:
            layout = LAYOUTS.get(layout_name, {})
            return SkillResult(
                success=True,
                response_text=f"{layout.get('name', layout_name)} applied",
                speak=True,
                data={"layout": layout_name},
            )
        else:
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't apply the {layout_name} layout",
                speak=True,
            )

    def _determine_layout(self, user_input: str, parameters: dict) -> str | None:
        """
        Determine which layout to apply based on user input.

        Args:
            user_input: Lowercase user input.
            parameters: Extracted parameters.

        Returns:
            Layout name or None if not determined.
        """
        # Check if LLM extracted a layout parameter
        if "layout" in parameters:
            return parameters["layout"]

        # Map common phrases to layouts
        layout_map = {
            "coding": "coding",
            "dev": "coding",
            "develop": "coding",
            "meeting": "meeting",
            "call": "meeting",
            "video": "meeting",
            "fullscreen": "fullscreen",
            "maximize": "fullscreen",
            "split": "split",
            "side by side": "split",
            "focus": "focus",
            "center": "focus",
        }

        for phrase, layout in layout_map.items():
            if phrase in user_input:
                return layout

        return None

    async def _check_hammerspoon(self) -> bool:
        """
        Check if Hammerspoon is available.

        Returns:
            True if Hammerspoon server is responding.
        """
        if self._hammerspoon_available is not None:
            return self._hammerspoon_available

        try:
            from roxy.macos.hammerspoon import get_hammerspoon_client

            client = get_hammerspoon_client()
            self._hammerspoon_available = await client._check_available()
            return self._hammerspoon_available
        except Exception as e:
            logger.debug(f"Hammerspoon not available: {e}")
            self._hammerspoon_available = False
            return False

    async def _apply_layout_hammerspoon(self, layout_name: str) -> bool:
        """
        Apply a window layout using Hammerspoon.

        Args:
            layout_name: Name of the layout to apply.

        Returns:
            True if successful.
        """
        try:
            from roxy.macos.hammerspoon import get_hammerspoon_client

            client = get_hammerspoon_client()
            layout = LAYOUTS.get(layout_name)

            if not layout:
                logger.warning(f"Unknown layout: {layout_name}")
                return False

            # Special handling for fullscreen
            if layout_name == "fullscreen":
                # Get frontmost app and maximize it
                success = await client.maximize_window("")
                return success

            # Apply layout for each window
            for window_config in layout["windows"]:
                app_name = window_config.get("app")
                unit = window_config.get("unit")

                if app_name:
                    await client.move_window(app_name, 0, unit)
                else:
                    # Move current window - need to get frontmost app first
                    from roxy.macos.pyobjc_bridge import get_macos_bridge

                    bridge = get_macos_bridge()
                    app_info = await bridge.get_active_window_info()
                    if app_info and app_info.get("name"):
                        await client.move_window(app_info["name"], 0, unit)

            return True

        except Exception as e:
            logger.error(f"Error applying Hammerspoon layout: {e}")
            return False

    async def _apply_layout_applescript(
        self, layout_name: str, user_input: str
    ) -> SkillResult:
        """
        Fallback: Apply layout using AppleScript.

        This is more limited than Hammerspoon but provides basic functionality.

        Args:
            layout_name: Name of the layout.
            user_input: Original user input.

        Returns:
            SkillResult with outcome.
        """
        try:
            from roxy.macos.applescript import get_applescript_runner

            runner = get_applescript_runner()

            # For fullscreen
            if layout_name == "fullscreen":
                script = """
                tell application "System Events"
                    set frontApp to (name of first application process whose frontmost is true)
                    tell application frontApp
                        activate
                        set bounds of front window to {0, 0, 1920, 1080}
                    end tell
                end tell
                """
                await runner.run(script)
                return SkillResult(
                    success=True,
                    response_text="Window maximized",
                    speak=True,
                )

            # For other layouts, provide guidance
            return SkillResult(
                success=False,
                response_text=f"Window layouts work best with Hammerspoon installed. For now, you can use macOS split view by holding the green button on a window.",
                speak=True,
            )

        except Exception as e:
            logger.error(f"Error applying AppleScript layout: {e}")
            return SkillResult(
                success=False,
                response_text=f"Sorry, I couldn't apply the layout. Hammerspoon provides the best window management experience.",
                speak=True,
            )
