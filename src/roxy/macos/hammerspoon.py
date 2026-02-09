"""Hammerspoon client for window management and automation.

Communicates with Hammerspoon's HTTP server to execute Lua code for
window management, audio control, and other system automation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class HammerspoonClient:
    """
    HTTP client for Hammerspoon automation.

    Expects Hammerspoon to be running with an HTTP server on localhost:7331.
    Provides pre-built commands for common automation tasks.
    """

    def __init__(self, host: str = "localhost", port: int = 7331) -> None:
        """
        Initialize Hammerspoon client.

        Args:
            host: Hammerspoon server host.
            port: Hammerspoon server port.
        """
        self._host = host
        self._port = port
        self._base_url = f"http://{host}:{port}"
        self._available: bool | None = None
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=5.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _check_available(self) -> bool:
        """Check if Hammerspoon is available.

        Returns:
            True if Hammerspoon server is responding.
        """
        if self._available is not None:
            return self._available

        try:
            client = await self._get_client()
            response = await client.get(f"{self._base_url}/")
            self._available = response.status_code == 200
            if not self._available:
                logger.debug(f"Hammerspoon returned status {response.status_code}")
            return self._available
        except Exception as e:
            logger.debug(f"Hammerspoon not available: {e}")
            self._available = False
            return False

    async def execute(self, lua_code: str) -> str:
        """
        Execute Lua code via Hammerspoon.

        Args:
            lua_code: Lua code to execute.

        Returns:
            Result string from Hammerspoon.

        Raises:
            RuntimeError: If Hammerspoon is not available.
            httpx.HTTPError: If HTTP request fails.
        """
        if not await self._check_available():
            raise RuntimeError("Hammerspoon is not available")

        logger.debug(f"Executing Hammerspoon Lua: {lua_code[:100]}...")

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self._base_url}/",
                data={"lua": lua_code},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            response.raise_for_status()
            result = response.text.strip()
            logger.debug(f"Hammerspoon result: {result[:100]}...")

            return result

        except httpx.HTTPError as e:
            logger.error(f"Hammerspoon HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error executing Hammerspoon code: {e}")
            raise

    # Pre-built commands for common automation tasks

    async def set_layout(self, name: str) -> bool:
        """
        Apply a predefined window layout.

        Args:
            name: Name of the layout to apply.

        Returns:
            True if successful.

        Note:
            This requires layout definitions in Hammerspoon config.
        """
        lua_code = f"""
        local layout = hs.layout.apply {{
            name = "{name}",
            -- Layout definitions should be in your Hammerspoon config
        }}
        return "Layout applied: {name}"
        """

        try:
            await self.execute(lua_code)
            return True
        except Exception as e:
            logger.error(f"Error setting layout: {e}")
            return False

    async def move_window(
        self,
        app_name: str,
        screen: int = 0,
        unit: dict[str, Any] | None = None,
    ) -> bool:
        """
        Move a window to a specific position.

        Args:
            app_name: Name of the application.
            screen: Screen index (0 for main screen).
            unit: Dict with 'x', 'y', 'w', 'h' for window position and size.

        Returns:
            True if successful.
        """
        if unit is None:
            unit = {"x": 0, "y": 0, "w": 0.5, "h": 1.0}  # Default: left half

        lua_code = f"""
        local app = hs.application.get("{app_name}")
        if not app then
            return "Application not found: {app_name}"
        end

        local window = app:mainWindow()
        if not window then
            return "No window found for: {app_name}"
        end

        local screen = hs.screen.allScreens()[{screen + 1}]
        local screenFrame = screen:frame()

        local frame = {{
            x = screenFrame.x + (screenFrame.w * {unit['x']}),
            y = screenFrame.y + (screenFrame.h * {unit['y']}),
            w = screenFrame.w * {unit['w']},
            h = screenFrame.h * {unit['h']}
        }}

        window:setFrame(frame)
        return "Window moved"
        """

        try:
            await self.execute(lua_code)
            return True
        except Exception as e:
            logger.error(f"Error moving window: {e}")
            return False

    async def set_volume(self, level: float) -> bool:
        """
        Set system volume.

        Args:
            level: Volume level between 0.0 and 1.0.

        Returns:
            True if successful.
        """
        lua_code = f"""
        local level = math.max(0, math.min(1, {level}))
        hs.audiodevice.defaultOutputDevice():setOutputVolume(level * 100)
        return "Volume set to " .. math.floor(level * 100) .. "%"
        """

        try:
            await self.execute(lua_code)
            return True
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
            return False

    async def get_volume(self) -> float:
        """
        Get current system volume.

        Returns:
            Volume level between 0.0 and 1.0, or 0.0 if error.
        """
        lua_code = """
        local volume = hs.audiodevice.defaultOutputDevice():outputVolume()
        return tostring(volume / 100)
        """

        try:
            result = await self.execute(lua_code)
            return float(result)
        except Exception as e:
            logger.error(f"Error getting volume: {e}")
            return 0.0

    async def toggle_wifi(self) -> bool:
        """
        Toggle WiFi on/off.

        Returns:
            True if successful.
        """
        lua_code = """
        local wifi = hs.wifi.currentNetwork()
        if wifi then
            hs.wifi.setPower(false)
            return "WiFi disabled"
        else
            hs.wifi.setPower(true)
            return "WiFi enabled"
        end
        """

        try:
            await self.execute(lua_code)
            return True
        except Exception as e:
            logger.error(f"Error toggling WiFi: {e}")
            return False

    async def get_wifi_status(self) -> dict[str, Any]:
        """
        Get WiFi status.

        Returns:
            Dict with 'enabled' and 'network' keys.
        """
        lua_code = """
        local enabled = hs.wifi.interfaceDetails() ~= nil
        local network = hs.wifi.currentNetwork()

        if enabled then
            if network then
                return string.format('{"enabled": true, "network": "%s"}', network)
            else
                return '{"enabled": true, "network": null}'
            end
        else
            return '{"enabled": false, "network": null}'
        end
        """

        try:
            result = await self.execute(lua_code)
            return json.loads(result)
        except Exception as e:
            logger.error(f"Error getting WiFi status: {e}")
            return {"enabled": False, "network": None}

    async def focus_window(self, app_name: str, window_index: int = 0) -> bool:
        """
        Focus a specific window of an application.

        Args:
            app_name: Name of the application.
            window_index: Window index to focus (0 for main window).

        Returns:
            True if successful.
        """
        lua_code = f"""
        local app = hs.application.get("{app_name}")
        if not app then
            return "Application not found: {app_name}"
        end

        local windows = app:allWindows()
        if not windows or #windows == 0 then
            return "No windows found for: {app_name}"
        end

        local window = windows[{window_index + 1}] or windows[1]
        window:focus()

        return "Window focused"
        """

        try:
            await self.execute(lua_code)
            return True
        except Exception as e:
            logger.error(f"Error focusing window: {e}")
            return False

    async def maximize_window(self, app_name: str) -> bool:
        """
        Maximize the main window of an application.

        Args:
            app_name: Name of the application.

        Returns:
            True if successful.
        """
        lua_code = f"""
        local app = hs.application.get("{app_name}")
        if not app then
            return "Application not found: {app_name}"
        end

        local window = app:mainWindow()
        if not window then
            return "No window found for: {app_name}"
        end

        window:maximize()
        return "Window maximized"
        """

        try:
            await self.execute(lua_code)
            return True
        except Exception as e:
            logger.error(f"Error maximizing window: {e}")
            return False

    async def get_windows_info(self) -> list[dict[str, Any]]:
        """
        Get information about all visible windows.

        Returns:
            List of dicts with window information.
        """
        lua_code = """
        local windows = hs.window.filter.default:getWindows()
        local result = {}

        for i, window in ipairs(windows) do
            local app = window:application()
            local frame = window:frame()

            table.insert(result, string.format(
                '{"index": %d, "app": "%s", "title": "%s", "x": %d, "y": %d, "w": %d, "h": %d}',
                i,
                app:name() or "unknown",
                window:title() or "unknown",
                frame.x,
                frame.y,
                frame.w,
                frame.h
            ))
        end

        return "[" .. table.concat(result, ",") .. "]"
        """

        try:
            result = await self.execute(lua_code)
            return json.loads(result)
        except Exception as e:
            logger.error(f"Error getting windows info: {e}")
            return []

    async def send_keys(self, text: str, app_name: str | None = None) -> bool:
        """
        Send keyboard input.

        Args:
            text: Text to type.
            app_name: Optional app name to target.

        Returns:
            True if successful.
        """
        # Escape the text for Lua
        text_escaped = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

        lua_code = f"""
        {f'hs.application.get("{app_name}"):activate()' if app_name else ''}
        hs.eventtap.keyStrokes("{text_escaped}")
        return "Keys sent"
        """

        try:
            await self.execute(lua_code)
            return True
        except Exception as e:
            logger.error(f"Error sending keys: {e}")
            return False

    async def show_notification(
        self, title: str, message: str, duration: float = 2.0
    ) -> bool:
        """
        Show a Hammerspoon notification.

        Args:
            title: Notification title.
            message: Notification message.
            duration: Duration in seconds.

        Returns:
            True if successful.
        """
        title_escaped = title.replace('\\', '\\\\').replace('"', '\\"')
        message_escaped = message.replace('\\', '\\\\').replace('"', '\\"')

        lua_code = f"""
        hs.notify.new({{
            title = "{title_escaped}",
            informativeText = "{message_escaped}",
            soundName = "default"
        }}):send()
        return "Notification sent"
        """

        try:
            await self.execute(lua_code)
            return True
        except Exception as e:
            logger.error(f"Error showing notification: {e}")
            return False


# Singleton instance
_hammerspoon_client: HammerspoonClient | None = None


def get_hammerspoon_client() -> HammerspoonClient:
    """
    Get or create the HammerspoonClient singleton.

    Returns:
        HammerspoonClient instance.
    """
    global _hammerspoon_client

    if _hammerspoon_client is None:
        _hammerspoon_client = HammerspoonClient()

    return _hammerspoon_client
