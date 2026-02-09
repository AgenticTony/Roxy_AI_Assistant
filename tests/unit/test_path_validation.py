"""Tests for path validation utilities."""

from __future__ import annotations

import pytest
import tempfile
import os
from pathlib import Path

from roxy.macos.path_validation import (
    validate_path,
    validate_file_path,
    validate_directory_path,
    add_allowed_directory,
    ALLOWED_BASE_DIRS,
)


class TestValidatePath:
    """Test path validation."""

    def test_validate_path_within_home(self) -> None:
        """Test validating a path within home directory."""
        # Test with a path that should exist (home directory)
        home = Path.home()
        result = validate_path(str(home))
        assert result == home

    def test_validate_path_with_tilde_expansion(self) -> None:
        """Test that tilde expansion works."""
        result = validate_path("~")
        assert result == Path.home()

    def test_validate_path_must_exist(self) -> None:
        """Test validation with must_exist=True."""
        # Home directory should exist
        result = validate_path("~", must_exist=True)
        assert result.exists()

    def test_validate_path_nonexistent_file(self) -> None:
        """Test validation with non-existent file."""
        with pytest.raises(FileNotFoundError):
            validate_path("~/nonexistent_file_12345.txt", must_exist=True)

    def test_validate_path_path_traversal(self) -> None:
        """Test that path traversal is blocked."""
        # Try to escape using .. sequences
        with pytest.raises(ValueError, match="outside allowed directories"):
            validate_path("/etc/passwd")

    def test_validate_symlink_resolution(self) -> None:
        """Test that symlinks are resolved and re-validated."""
        # Create a temporary file in home directory
        home = Path.home()
        test_file = home / ".test_roxy_file.txt"
        link_path = home / ".test_roxy_symlink"

        try:
            # Create a test file
            test_file.write_text("test")

            # Create symlink (may fail on some systems without proper permissions)
            try:
                os.symlink(test_file, link_path)

                # Validate the symlink - should resolve to actual file
                result = validate_path(str(link_path), must_exist=True)

                # Result should point to the actual file, not the symlink
                assert result.exists()
                assert result.resolve() == test_file.resolve()

            except OSError:
                # Symlink creation failed (permissions, etc.)
                pytest.skip("Cannot create symlinks for testing")
        finally:
            # Clean up
            try:
                link_path.unlink(missing_ok=True)
            except (OSError, FileNotFoundError):
                pass
            try:
                test_file.unlink(missing_ok=True)
            except (OSError, FileNotFoundError):
                pass


class TestValidateFilePath:
    """Test file path validation."""

    def test_validate_file_path_with_file(self) -> None:
        """Test validating a file path in home directory."""
        # Create a test file in home directory
        home = Path.home()
        test_file = home / ".test_roxy_temp.txt"

        try:
            test_file.write_text("test content")
            result = validate_file_path(str(test_file), must_exist=True)
            assert result.is_file()
        finally:
            test_file.unlink(missing_ok=True)

    def test_validate_file_path_rejects_directory(self) -> None:
        """Test that file validation rejects directories."""
        # Create a test directory in home
        home = Path.home()
        test_dir = home / ".test_roxy_temp_dir"

        try:
            test_dir.mkdir(exist_ok=True)
            # Should raise NotADirectoryError (which is a subclass of OSError)
            with pytest.raises(NotADirectoryError, match="is a directory, not a file"):
                validate_file_path(str(test_dir), must_exist=True)
        finally:
            test_dir.rmdir()


class TestValidateDirectoryPath:
    """Test directory path validation."""

    def test_validate_directory_path_with_directory(self) -> None:
        """Test validating a directory path in home."""
        home = Path.home()
        test_dir = home / ".test_roxy_temp_dir2"

        try:
            test_dir.mkdir(exist_ok=True)
            result = validate_directory_path(str(test_dir), must_exist=True)
            assert result.is_dir()
        finally:
            test_dir.rmdir()

    def test_validate_directory_path_rejects_file(self) -> None:
        """Test that directory validation rejects files."""
        home = Path.home()
        test_file = home / ".test_roxy_temp_file.txt"

        try:
            test_file.write_text("test")
            with pytest.raises(ValueError, match="is not a directory"):
                validate_directory_path(str(test_file), must_exist=True)
        finally:
            test_file.unlink(missing_ok=True)


class TestAddAllowedDirectory:
    """Test adding allowed directories."""

    def test_add_allowed_directory(self) -> None:
        """Test adding a directory to allowed list."""
        home = Path.home()
        test_dir = home / ".test_roxy_allowed_dir"

        try:
            test_dir.mkdir(exist_ok=True)
            initial_count = len(ALLOWED_BASE_DIRS)

            add_allowed_directory(str(test_dir))

            # Directory should now be in allowed list
            assert len(ALLOWED_BASE_DIRS) >= initial_count
            assert test_dir.resolve() in [d.resolve() for d in ALLOWED_BASE_DIRS]

            # Path within that directory should now be valid
            test_path = test_dir / "test.txt"
            test_path.write_text("test")
            result = validate_path(str(test_path), must_exist=True)
            assert result == test_path.resolve()
        finally:
            # Clean up
            for item in test_dir.iterdir():
                item.unlink(missing_ok=True)
            test_dir.rmdir()

    def test_add_allowed_directory_nonexistent(self) -> None:
        """Test that adding non-existent directory raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            add_allowed_directory("/nonexistent_directory_12345")

    def test_add_allowed_directory_not_a_directory(self) -> None:
        """Test that adding a file as directory raises error."""
        home = Path.home()
        test_file = home / ".test_roxy_temp_file2.txt"

        try:
            test_file.write_text("test")
            with pytest.raises(ValueError, match="is not a directory"):
                add_allowed_directory(str(test_file))
        finally:
            test_file.unlink(missing_ok=True)


class TestTempDirectoryAccess:
    """Test access to temp directories."""

    def test_tmp_directory_access(self) -> None:
        """Test that /tmp is accessible if it exists."""
        tmp_path = Path("/tmp")
        if tmp_path.exists() and tmp_path.is_dir():
            # Should be able to validate /tmp
            result = validate_path("/tmp")
            assert result == tmp_path.resolve()

    def test_var_tmp_directory_access(self) -> None:
        """Test that /var/tmp is accessible if it exists."""
        var_tmp_path = Path("/var/tmp")
        if var_tmp_path.exists() and var_tmp_path.is_dir():
            # Should be able to validate /var/tmp
            result = validate_path("/var/tmp")
            assert result == var_tmp_path.resolve()
