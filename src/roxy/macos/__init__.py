"""macOS integration module.

Provides bridges to macOS native APIs and system features.
"""

from roxy.macos.applescript import (
    AppleScriptRunner,
    escape_applescript_string,
    get_applescript_runner,
)
from roxy.macos.hammerspoon import HammerspoonClient, get_hammerspoon_client
from roxy.macos.menubar import RoxyMenuBar, RoxyMode, get_menubar_app
from roxy.macos.path_validation import (
    ALLOWED_BASE_DIRS,
    add_allowed_directory,
    validate_directory_path,
    validate_file_path,
    validate_path,
)
from roxy.macos.pyobjc_bridge import MacOSBridge, get_macos_bridge
from roxy.macos.spotlight import SpotlightSearch, get_spotlight_search

__all__ = [
    "AppleScriptRunner",
    "get_applescript_runner",
    "escape_applescript_string",
    "MacOSBridge",
    "get_macos_bridge",
    "SpotlightSearch",
    "get_spotlight_search",
    "HammerspoonClient",
    "get_hammerspoon_client",
    "RoxyMenuBar",
    "get_menubar_app",
    "RoxyMode",
    "validate_path",
    "validate_file_path",
    "validate_directory_path",
    "add_allowed_directory",
    "ALLOWED_BASE_DIRS",
]
