"""macOS integration module.

Provides bridges to macOS native APIs and system features.
"""

from roxy.macos.applescript import AppleScriptRunner, get_applescript_runner
from roxy.macos.pyobjc_bridge import MacOSBridge, get_macos_bridge
from roxy.macos.spotlight import SpotlightSearch, get_spotlight_search
from roxy.macos.hammerspoon import HammerspoonClient, get_hammerspoon_client
from roxy.macos.menubar import RoxyMenuBar, get_menubar_app, RoxyMode

__all__ = [
    "AppleScriptRunner",
    "get_applescript_runner",
    "MacOSBridge",
    "get_macos_bridge",
    "SpotlightSearch",
    "get_spotlight_search",
    "HammerspoonClient",
    "get_hammerspoon_client",
    "RoxyMenuBar",
    "get_menubar_app",
    "RoxyMode",
]
