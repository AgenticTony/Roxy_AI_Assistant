"""Path validation utilities for Roxy.

Provides security-focused path validation to prevent path traversal attacks
and ensure file operations stay within allowed directories.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Base directories that are allowed for file operations
ALLOWED_BASE_DIRS: list[Path] = [
    Path.home(),  # User's home directory
]

# Optional directories that may be allowed if they exist
OPTIONAL_ALLOWED_DIRS: list[Path] = [
    Path("/Users/Shared"),
    Path("/tmp"),
    Path("/var/tmp"),
]

for optional_dir in OPTIONAL_ALLOWED_DIRS:
    if optional_dir.exists():
        ALLOWED_BASE_DIRS.append(optional_dir)


def validate_path(path_str: str, must_exist: bool = False) -> Path:
    """Validate and resolve a file path.
    
    Prevents path traversal attacks by ensuring the resolved path
    is within allowed base directories.
    """
    try:
        path = Path(path_str).expanduser().resolve()
    except Exception as e:
        raise ValueError(f"Invalid path '{path_str}': {e}")

    is_allowed = False
    for base_dir in ALLOWED_BASE_DIRS:
        try:
            resolved_base = base_dir.resolve()
            if path == resolved_base or str(path).startswith(str(resolved_base)):
                is_allowed = True
                break
        except Exception:
            continue

    if not is_allowed:
        allowed_dirs_str = ", ".join(str(d) for d in ALLOWED_BASE_DIRS)
        raise ValueError(
            f"Path '{path_str}' is outside allowed directories: {allowed_dirs_str}"
        )

    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path '{path_str}' does not exist")

    if path.is_symlink():
        logger.warning(f"Path '{path}' is a symbolic link, resolving target")
        target = path.resolve()
        return validate_path(str(target), must_exist=must_exist)

    return path


def validate_file_path(path_str: str, must_exist: bool = False) -> Path:
    """Validate a path that should be a file."""
    path = validate_path(path_str, must_exist=must_exist)
    if must_exist and path.exists() and path.is_dir():
        raise NotADirectoryError(f"Path '{path_str}' is a directory, not a file")
    return path


def validate_directory_path(path_str: str, must_exist: bool = False) -> Path:
    """Validate a path that should be a directory."""
    path = validate_path(path_str, must_exist=must_exist)
    if must_exist and path.exists() and not path.is_dir():
        raise ValueError(f"Path '{path_str}' is not a directory")
    return path


def add_allowed_directory(directory: str) -> None:
    """Add a directory to the allowed base directories."""
    path = Path(directory).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Directory '{directory}' does not exist")
    if not path.is_dir():
        raise ValueError(f"Path '{directory}' is not a directory")
    if path not in ALLOWED_BASE_DIRS:
        ALLOWED_BASE_DIRS.append(path)
        logger.info(f"Added allowed directory: {path}")
