"""macOS bridge using PyObjC for native system integration.

Provides direct access to macOS frameworks like NSWorkspace, NSProcessInfo,
and other system APIs. Falls back to AppleScript if PyObjC is not available.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# Try to import PyObjC frameworks
try:
    import AppKit
    import Foundation
    from AppKit import NSWorkspace
    from Foundation import NSProcessInfo, NSUserDefaults

    PYOBJC_AVAILABLE = True
    logger.info("PyObjC frameworks loaded successfully")
except ImportError:
    PYOBJC_AVAILABLE = False
    logger.warning("PyObjC not available, will use AppleScript fallback")


class MacOSBridge:
    """
    Bridge to native macOS APIs using PyObjC.

    Provides methods for system integration that are faster and more reliable
    than AppleScript. Falls back to AppleScriptRunner methods when PyObjC
    is not available.
    """

    def __init__(self) -> None:
        """Initialize the macOS bridge."""
        self._pyobjc_available = PYOBJC_AVAILABLE

        if not self._pyobjc_available:
            logger.debug("PyObjC not available, will use AppleScript fallback")
            return

        # Get shared workspace instance
        self._workspace = NSWorkspace.sharedWorkspace()

    @property
    def is_available(self) -> bool:
        """Check if PyObjC bridge is available."""
        return self._pyobjc_available

    async def get_running_applications(self) -> list[dict[str, Any]]:
        """
        Get list of currently running applications.

        Returns:
            List of dicts with app name, bundle ID, PID, and if it's frontmost.
        """
        if not self._pyobjc_available:
            # Fallback to AppleScript
            from .applescript import get_applescript_runner

            runner = get_applescript_runner()
            return await runner.get_running_apps()

        try:
            running_apps = self._workspace.runningApplications()

            apps = []
            for app in running_apps:
                # Only include visible apps (not background-only)
                if app.activationPolicy() == 0:  # NSApplicationActivationPolicyRegular
                    apps.append(
                        {
                            "name": app.localizedName(),
                            "bundle_id": app.bundleIdentifier(),
                            "pid": app.processIdentifier(),
                            "frontmost": app.isActive(),
                        }
                    )

            return apps

        except Exception as e:
            logger.error(f"Error getting running applications: {e}")
            return []

    async def launch_application(self, name: str) -> bool:
        """
        Launch an application by name.

        Args:
            name: Application name (can be partial, e.g., "Chrome" for "Google Chrome").

        Returns:
            True if successfully launched or already running.

        Note:
            This method will try common app name aliases and bundle IDs.
        """
        if not self._pyobjc_available:
            # Fallback to open command
            try:
                result = subprocess.run(
                    ["open", "-a", name],
                    capture_output=True,
                    text=True,
                )
                return result.returncode == 0
            except Exception as e:
                logger.error(f"Error launching application {name}: {e}")
                return False

        try:
            # First, try to find the app URL
            app_url = self._workspace.URLForApplicationWithBundleIdentifier_(name)
            if app_url is None:
                # Try by app name
                app_url = self._workspace.absoluteURLForAppBundleWithIdentifier_(name)

            if app_url:
                return self._workspace.launchApplicationAtURL_options_configuration_error_(
                    app_url, 0, None, None
                )

            # Try using open command as fallback
            result = subprocess.run(
                ["open", "-a", name],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Error launching application {name}: {e}")
            # Last resort: try open command
            try:
                result = subprocess.run(
                    ["open", "-a", name],
                    capture_output=True,
                    text=True,
                )
                return result.returncode == 0
            except Exception:
                return False

    async def get_active_window_info(self) -> dict[str, Any]:
        """
        Get information about the active/frontmost window.

        Returns:
            Dict with app name, bundle ID, and window info if available.
        """
        if not self._pyobjc_available:
            # Fallback to AppleScript
            from .applescript import get_applescript_runner

            runner = get_applescript_runner()
            return await runner.get_frontmost_app()

        try:
            frontmost_app = self._workspace.frontmostApplication()

            if frontmost_app is None:
                return {"name": "unknown", "bundle_id": "unknown", "pid": 0}

            return {
                "name": frontmost_app.localizedName(),
                "bundle_id": frontmost_app.bundleIdentifier(),
                "pid": frontmost_app.processIdentifier(),
            }

        except Exception as e:
            logger.error(f"Error getting active window info: {e}")
            return {"name": "unknown", "bundle_id": "unknown", "pid": 0}

    async def get_system_info(self) -> dict[str, Any]:
        """
        Get system information.

        Returns:
            Dict with OS version, RAM, CPU info, and computer name.
        """
        info: dict[str, Any] = {}

        if not self._pyobjc_available:
            # Fallback to system commands
            try:
                # OS version
                result = subprocess.run(
                    ["sw_vers", "-productVersion"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                info["os_version"] = result.stdout.strip()
            except Exception:
                info["os_version"] = "unknown"

            try:
                # Computer name
                result = subprocess.run(
                    ["scutil", "--get", "ComputerName"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                info["computer_name"] = result.stdout.strip()
            except Exception:
                info["computer_name"] = "unknown"

            try:
                # CPU
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                info["cpu"] = result.stdout.strip()
            except Exception:
                info["cpu"] = "unknown"

            try:
                # RAM (in GB)
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                ram_bytes = int(result.stdout.strip())
                info["ram_gb"] = round(ram_bytes / (1024**3), 1)
            except Exception:
                info["ram_gb"] = 0

            return info

        try:
            # OS version
            process_info = NSProcessInfo.processInfo()
            info["os_version"] = str(process_info.operatingSystemVersionString())

            # Computer name
            host = Foundation.NSHost.currentHost()
            info["computer_name"] = str(host.localizedName())

            # Hardware info via sysctl
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                info["cpu"] = result.stdout.strip()
            except Exception:
                info["cpu"] = "unknown"

            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                ram_bytes = int(result.stdout.strip())
                info["ram_gb"] = round(ram_bytes / (1024**3), 1)
            except Exception:
                info["ram_gb"] = 0

        except Exception as e:
            logger.error(f"Error getting system info: {e}")

        return info

    async def set_display_brightness(self, level: float) -> bool:
        """
        Set display brightness.

        Args:
            level: Brightness level between 0.0 and 1.0.

        Returns:
            True if successful.
        """
        # This requires screen events API which needs additional permissions
        # We'll use the brightness command-line tool
        try:
            # Map 0-1 to 0-100
            brightness_level = int(max(0, min(100, level * 100)))

            result = subprocess.run(
                ["brightness", str(brightness_level)],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Error setting brightness: {e}")
            return False

    async def toggle_do_not_disturb(self, enabled: bool) -> bool:
        """
        Toggle Do Not Disturb mode.

        Args:
            enabled: True to enable DND, False to disable.

        Returns:
            True if successful.
        """
        # DND control has changed across macOS versions
        # We'll try multiple approaches

        try:
            # For macOS 13+ (Ventura and later)
            if enabled:
                script = """
                tell application "System Events"
                    tell appearance preferences
                        set focus status to true
                    end tell
                end tell
                """
            else:
                script = """
                tell application "System Events"
                    tell appearance preferences
                        set focus status to false
                    end tell
                end tell
                """

            from .applescript import get_applescript_runner

            runner = get_applescript_runner()
            await runner.run(script)
            return True

        except Exception as e:
            logger.error(f"Error toggling DND: {e}")
            return False

    async def get_app_bundle_id(self, app_name: str) -> str | None:
        """
        Get the bundle ID for an application by name.

        Args:
            app_name: Name of the application.

        Returns:
            Bundle ID string or None if not found.
        """
        if not self._pyobjc_available:
            # Try to find using mdfind
            try:
                result = subprocess.run(
                    ["mdfind", f"kMDItemDisplayName == '{app_name}'c"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                paths = result.stdout.strip().split("\n")
                if paths and paths[0]:
                    # Extract bundle ID from .app path
                    app_path = paths[0]
                    if ".app" in app_path:
                        plist_path = f"{app_path}/Contents/Info.plist"
                        try:
                            result = subprocess.run(
                                [
                                    "/usr/libexec/PlistBuddy",
                                    "-c",
                                    "Print :CFBundleIdentifier",
                                    plist_path,
                                ],
                                capture_output=True,
                                text=True,
                                check=True,
                            )
                            return result.stdout.strip()
                        except Exception:
                            pass
            except Exception:
                pass

            return None

        try:
            # Use NSWorkspace to find app
            app_url = self._workspace.URLForApplicationWithBundleIdentifier_(app_name)
            if app_url:
                bundle = self._workspace.bundleForURL_(app_url)
                return bundle.bundleIdentifier()

            return None

        except Exception as e:
            logger.error(f"Error getting bundle ID for {app_name}: {e}")
            return None


# Singleton instance
_macos_bridge: MacOSBridge | None = None


def get_macos_bridge() -> MacOSBridge:
    """
    Get or create the MacOSBridge singleton.

    Returns:
        MacOSBridge instance.
    """
    global _macos_bridge

    if _macos_bridge is None:
        _macos_bridge = MacOSBridge()

    return _macos_bridge
