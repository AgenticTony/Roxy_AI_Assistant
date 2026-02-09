"""Application launcher skill for macOS.

Allows launching applications by name with common aliases and fuzzy matching.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

from roxy.skills.base import Permission, RoxySkill, SkillContext, SkillResult

logger = logging.getLogger(__name__)


# Common app name aliases for better matching
APP_ALIASES: dict[str, str] = {
    "chrome": "Google Chrome",
    "firefox": "Firefox",
    "safari": "Safari",
    "edge": "Microsoft Edge",
    "slack": "Slack",
    "discord": "Discord",
    "zoom": "zoom.us",
    "teams": "Microsoft Teams",
    "vscode": "Visual Studio Code",
    "code": "Visual Studio Code",
    "xcode": "Xcode",
    "terminal": "Terminal",
    "iterm": "iTerm",
    "iterm2": "iTerm",
    "notes": "Notes",
    "mail": "Mail",
    "calendar": "Calendar",
    "photos": "Photos",
    "music": "Music",
    "spotify": "Spotify",
    "finder": "Finder",
}


class AppLauncherSkill(RoxySkill):
    """Skill for launching macOS applications."""

    name = "app_launcher"
    description = "Launch applications by name"
    triggers = [
        "open",
        "launch",
        "start",
        "run",
    ]
    permissions = [Permission.APPLESCRIPT]
    requires_cloud = False

    async def execute(self, context: SkillContext) -> SkillResult:
        """
        Execute the application launcher skill.

        Args:
            context: Skill execution context.

        Returns:
            SkillResult indicating success/failure and response.
        """
        user_input = context.user_input.lower()
        parameters = context.parameters

        # Extract app name from user input
        app_name = self._extract_app_name(user_input, parameters)

        if not app_name:
            return SkillResult(
                success=False,
                response_text="What application would you like me to open?",
                speak=False,
            )

        # Normalize app name using aliases
        normalized_name = APP_ALIASES.get(app_name.lower(), app_name)

        logger.info(f"Attempting to launch application: {normalized_name}")

        # Try to launch using multiple methods
        success = await self._launch_app(normalized_name)

        if success:
            return SkillResult(
                success=True,
                response_text=f"Opening {normalized_name}",
                speak=True,
            )
        else:
            # Try to find similar apps using Spotlight
            suggestions = await self._find_similar_apps(app_name)

            if suggestions:
                suggestion_text = f"Couldn't find '{app_name}'. Did you mean: {', '.join(suggestions[:3])}?"
            else:
                suggestion_text = f"Couldn't find an application named '{app_name}'."

            return SkillResult(
                success=False,
                response_text=suggestion_text,
                speak=True,
            )

    def _extract_app_name(self, user_input: str, parameters: dict) -> str | None:
        """
        Extract application name from user input.

        Args:
            user_input: Lowercase user input string.
            parameters: Extracted parameters from LLM.

        Returns:
            Application name or None if not found.
        """
        # First check if LLM extracted an app_name parameter
        if "app_name" in parameters:
            return parameters["app_name"]

        # Try to extract app name from common patterns
        import re

        # Pattern: "open [app]", "launch [app]", "start [app]"
        patterns = [
            r"open\s+(?:the\s+)?(?:application\s+)?(.+?)(?:\s+please|$|\?)",
            r"launch\s+(?:the\s+)?(?:application\s+)?(.+?)(?:\s+please|$|\?)",
            r"start\s+(?:the\s+)?(?:application\s+)?(.+?)(?:\s+please|$|\?)",
            r"run\s+(?:the\s+)?(?:application\s+)?(.+?)(?:\s+please|$|\?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, user_input)
            if match:
                app_name = match.group(1).strip()
                # Remove trailing words
                app_name = app_name.split()[0] if app_name.split() else app_name
                return app_name

        return None

    async def _launch_app(self, app_name: str) -> bool:
        """
        Launch an application using multiple methods.

        Args:
            app_name: Name of the application.

        Returns:
            True if successfully launched.
        """
        # Method 1: Try PyObjC bridge if available
        try:
            from roxy.macos.pyobjc_bridge import get_macos_bridge

            bridge = get_macos_bridge()
            if bridge.is_available:
                result = await bridge.launch_application(app_name)
                if result:
                    return True
        except Exception as e:
            logger.debug(f"PyObjC launch failed: {e}")

        # Method 2: Try using the 'open' command with app name
        try:
            result = subprocess.run(
                ["open", "-a", app_name],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info(f"Launched {app_name} using open command")
                return True
        except Exception as e:
            logger.debug(f"Open command failed: {e}")

        # Method 3: Try AppleScript
        try:
            from roxy.macos.applescript import escape_applescript_string, get_applescript_runner

            runner = get_applescript_runner()
            app_name_safe = escape_applescript_string(app_name)
            script = f'tell application "{app_name_safe}" to activate'
            await runner.run(script)
            return True
        except Exception as e:
            logger.debug(f"AppleScript launch failed: {e}")

        # Method 4: Try using Spotlight to find the app
        try:
            from roxy.macos.spotlight import get_spotlight_search

            spotlight = get_spotlight_search()
            app_paths = await spotlight.find_applications(app_name)

            if app_paths:
                # Try the first match
                app_path = app_paths[0]
                result = subprocess.run(
                    ["open", app_path],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    logger.info(f"Launched {app_name} from Spotlight result")
                    return True
        except Exception as e:
            logger.debug(f"Spotlight launch failed: {e}")

        return False

    async def _find_similar_apps(self, app_name: str) -> list[str]:
        """
        Find similar application names using Spotlight.

        Args:
            app_name: App name to search for.

        Returns:
            List of similar app names.
        """
        try:
            from roxy.macos.spotlight import get_spotlight_search

            spotlight = get_spotlight_search()
            app_paths = await spotlight.find_applications(app_name)

            # Extract app names from paths
            import re

            app_names = []
            for path in app_paths:
                match = re.search(r"/([^/]+)\.app$", path)
                if match:
                    app_names.append(match.group(1))

            return app_names[:5]

        except Exception as e:
            logger.debug(f"Error finding similar apps: {e}")
            return []
