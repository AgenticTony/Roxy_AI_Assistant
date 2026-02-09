"""Menu bar application for Roxy using rumps.

Provides a system menu bar icon and menu for controlling Roxy,
displaying status, and accessing settings.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Try to import rumps
try:
    import rumps

    RUMPS_AVAILABLE = True
except ImportError:
    RUMPS_AVAILABLE = False
    logger.warning("rumps not available, menu bar disabled")


class RoxyMode(str, Enum):
    """Roxy operational modes."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class RoxyMenuBar:
    """
    Menu bar application for Roxy.

    Displays Roxy's status and provides menu items for control.
    """

    def __init__(self, name: str = "Roxy") -> None:
        """
        Initialize the menu bar app.

        Args:
            name: Name to display in the menu bar.
        """
        if not RUMPS_AVAILABLE:
            logger.warning("Cannot create menu bar - rumps not available")
            self._app = None
            return

        self._name = name
        self._mode = RoxyMode.IDLE
        self._current_command = ""

        # Create menu bar app
        self._app = rumps.App(
            name,
            title="☆",
            icon=None,
            template=True,
        )

        # Setup menu items
        self._setup_menu()

        # Callbacks
        self._on_listening_toggle: Callable[[], None] | None = None
        self._on_settings: Callable[[], None] | None = None
        self._on_memory_view: Callable[[], None] | None = None
        self._on_quit: Callable[[], None] | None = None

        logger.info("Menu bar app initialized")

    def _setup_menu(self) -> None:
        """Setup menu items."""
        if self._app is None:
            return

        # Status menu items (read-only)
        self._mode_item = rumps.MenuItem(f"Mode: {self._mode.value}")
        self._command_item = rumps.MenuItem("Command: (idle)")

        # Action menu items
        self._listening_item = rumps.MenuItem("Start Listening", callback=self._on_listening_clicked)
        self._settings_item = rumps.MenuItem("Settings...", callback=self._on_settings_clicked)
        self._memory_item = rumps.MenuItem("Memory View...", callback=self._on_memory_clicked)

        # Separator
        rumps.separator = rumps.MenuItem.separator()

        # Quit
        self._quit_item = rumps.MenuItem("Quit", callback=self._on_quit_clicked)

        # Set menu
        self._app.menu = [
            self._mode_item,
            self._command_item,
            rumps.separator,
            self._listening_item,
            self._settings_item,
            self._memory_item,
            rumps.separator,
            self._quit_item,
        ]

    def run(self) -> None:
        """Run the menu bar app (blocking)."""
        if self._app is None:
            logger.warning("Cannot run menu bar - rumps not available")
            return

        logger.info("Starting menu bar app")
        self._app.run()

    def start(self) -> None:
        """Start the menu bar app (non-blocking).

        Note:
            This creates a background thread for the menu bar.
        """
        if self._app is None:
            logger.warning("Cannot start menu bar - rumps not available")
            return

        logger.info("Starting menu bar app in background")
        # rumps.App.run() is blocking, so we need to run it in a thread
        # For now, just call run() - integration with voice pipeline will need proper async handling
        self.run()

    def stop(self) -> None:
        """Stop the menu bar app."""
        if self._app is None:
            return

        logger.info("Stopping menu bar app")
        self._app.quit()

    def update_status(self, mode: RoxyMode, command: str = "") -> None:
        """
        Update the displayed status.

        Args:
            mode: Current operational mode.
            command: Current command being processed (if any).
        """
        self._mode = mode
        self._current_command = command

        if self._app is None:
            return

        # Update mode display
        self._mode_item.title = f"Mode: {mode.value}"

        # Update command display
        if command:
            self._command_item.title = f"Command: {command[:50]}..."
        else:
            self._command_item.title = "Command: (idle)"

        # Update icon based on mode
        if mode == RoxyMode.LISTENING:
            self._app.title = "🎤"
        elif mode == RoxyMode.PROCESSING:
            self._app.title = "⚙️"
        elif mode == RoxyMode.SPEAKING:
            self._app.title = "💬"
        else:
            self._app.title = "☆"

    def set_listening_state(self, is_listening: bool) -> None:
        """
        Update the listening state menu item.

        Args:
            is_listening: Whether Roxy is currently listening.
        """
        if self._app is None:
            return

        if is_listening:
            self._listening_item.title = "Stop Listening"
        else:
            self._listening_item.title = "Start Listening"

    # Callback handlers

    def _on_listening_clicked(self, sender: rumps.MenuItem) -> None:
        """Handle listening toggle button click."""
        is_listening = "Stop" in sender.title
        self.set_listening_state(not is_listening)

        if self._on_listening_toggle:
            self._on_listening_toggle()

    def _on_settings_clicked(self, sender: rumps.MenuItem) -> None:
        """Handle settings menu item click."""
        if self._on_settings:
            self._on_settings()

    def _on_memory_clicked(self, sender: rumps.MenuItem) -> None:
        """Handle memory view menu item click."""
        if self._on_memory_view:
            self._on_memory_view()

    def _on_quit_clicked(self, sender: rumps.MenuItem) -> None:
        """Handle quit menu item click."""
        if self._on_quit:
            self._on_quit()

        # Also quit the menu bar app
        self.stop()

    # Callback setters

    def set_listening_toggle_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for listening toggle."""
        self._on_listening_toggle = callback

    def set_settings_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for settings menu."""
        self._on_settings = callback

    def set_memory_view_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for memory view menu."""
        self._on_memory_view = callback

    def set_quit_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for quit menu."""
        self._on_quit = callback

    # Notification helpers

    def notify(self, title: str, message: str, sound: bool = False) -> None:
        """
        Show a macOS notification.

        Args:
            title: Notification title.
            message: Notification message.
            sound: Whether to play a sound.
        """
        if self._app is None:
            # Fallback to AppleScript
            import asyncio

            async def show_fallback():
                from .applescript import get_applescript_runner

                runner = get_applescript_runner()
                await runner.send_notification(title, message)

            # Run in event loop if available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(show_fallback())
                else:
                    asyncio.run(show_fallback())
            except Exception:
                pass

            return

        # Use rumps notification
        rumps.notification(title, message, sound=sound)


# Singleton instance
_menubar_app: RoxyMenuBar | None = None


def get_menubar_app(name: str = "Roxy") -> RoxyMenuBar:
    """
    Get or create the RoxyMenuBar singleton.

    Args:
        name: Name to display in the menu bar.

    Returns:
        RoxyMenuBar instance.
    """
    global _menubar_app

    if _menubar_app is None:
        _menubar_app = RoxyMenuBar(name)

    return _menubar_app
