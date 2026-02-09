#!/usr/bin/env python3
"""Roxy Installation Verification Script.

This script verifies that all components of Roxy are properly installed
and configured according to the Phase 4 requirements.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any
import subprocess
import asyncio

# Colors for output
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{CYAN}{'=' * 60}{NC}")
    print(f"{CYAN}{text}{NC}")
    print(f"{CYAN}{'=' * 60}{NC}\n")


def print_check(name: str, passed: bool, details: str = "") -> None:
    """Print a check result."""
    status = f"{GREEN}✓{NC}" if passed else f"{RED}✗{NC}"
    print(f"{status} {name}")
    if details:
        print(f"  {details}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{YELLOW}⚠ {text}{NC}")


class VerificationResults:
    """Track verification results."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.warnings: list[str] = []

    def add_pass(self, name: str) -> None:
        self.passed.append(name)

    def add_fail(self, name: str, reason: str) -> None:
        self.failed.append((name, reason))

    def add_warning(self, text: str) -> None:
        self.warnings.append(text)

    def print_summary(self) -> None:
        """Print verification summary."""
        print_header("VERIFICATION SUMMARY")

        print(f"{GREEN}Passed: {len(self.passed)}{NC}")
        for name in self.passed:
            print(f"  ✓ {name}")

        if self.failed:
            print(f"\n{RED}Failed: {len(self.failed)}{NC}")
            for name, reason in self.failed:
                print(f"  ✗ {name}")
                print(f"    {reason}")

        if self.warnings:
            print(f"\n{YELLOW}Warnings: {len(self.warnings)}{NC}")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")

        print()


def verify_project_structure(results: VerificationResults) -> None:
    """Verify project structure is complete."""
    print_header("VERIFYING PROJECT STRUCTURE")

    required_dirs = [
        "src/roxy",
        "src/roxy/brain",
        "src/roxy/skills",
        "src/roxy/voice",
        "src/roxy/memory",
        "src/roxy/macos",
        "src/roxy/mcp",
        "src/roxy/config",
        "config",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/benchmarks",
        "docs",
        "scripts",
    ]

    required_files = [
        "src/roxy/__init__.py",
        "src/roxy/main.py",
        "src/roxy/config.py",
        "src/roxy/brain/__init__.py",
        "src/roxy/brain/orchestrator.py",
        "src/roxy/brain/router.py",
        "src/roxy/brain/privacy.py",
        "src/roxy/brain/llm_clients.py",
        "src/roxy/skills/__init__.py",
        "src/roxy/skills/base.py",
        "src/roxy/skills/registry.py",
        "src/roxy/memory/__init__.py",
        "src/roxy/memory/manager.py",
        "src/roxy/voice/__init__.py",
        "src/roxy/voice/pipeline.py",
        "src/roxy/macos/__init__.py",
        "config/default.yaml",
        "pyproject.toml",
        ".env.example",
        "README.md",
        "CLAUDE.md",
        "docs/architecture.md",
        "docs/skills-guide.md",
        "docs/privacy-model.md",
        "docs/performance.md",
        "scripts/install.sh",
        "scripts/setup_ollama.sh",
        "tests/conftest.py",
    ]

    project_root = Path(__file__).parent.parent

    # Check directories
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            results.add_pass(f"Directory: {dir_path}")
        else:
            results.add_fail(f"Directory: {dir_path}", f"Not found at {full_path}")

    # Check files
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            results.add_pass(f"File: {file_path}")
        else:
            results.add_fail(f"File: {file_path}", f"Not found at {full_path}")


def verify_config_files(results: VerificationResults) -> None:
    """Verify configuration files are valid."""
    print_header("VERIFYING CONFIGURATION FILES")

    project_root = Path(__file__).parent.parent

    config_files = [
        "config/default.yaml",
        "config/privacy.yaml",
        "config/mcp_servers.yaml",
    ]

    for config_file in config_files:
        config_path = project_root / config_file
        if config_path.exists():
            try:
                with open(config_path) as f:
                    content = f.read()
                    if content.strip():
                        results.add_pass(f"Config file: {config_file}")
                    else:
                        results.add_fail(f"Config file: {config_file}", "File is empty")
            except Exception as e:
                results.add_fail(f"Config file: {config_file}", f"Error reading: {e}")
        else:
            results.add_warning(f"Config file not found: {config_file} (optional)")


def verify_python_imports(results: VerificationResults) -> None:
    """Verify that core Python modules can be imported."""
    print_header("VERIFYING PYTHON IMPORTS")

    modules = [
        ("roxy", "Main package"),
        ("roxy.config", "Configuration"),
        ("roxy.brain.orchestrator", "Orchestrator"),
        ("roxy.brain.llm_clients", "LLM clients"),
        ("roxy.brain.router", "Router"),
        ("roxy.brain.privacy", "Privacy gateway"),
        ("roxy.skills.base", "Skill base"),
        ("roxy.skills.registry", "Skill registry"),
        ("roxy.memory.manager", "Memory manager"),
        ("roxy.voice.pipeline", "Voice pipeline"),
        ("roxy.macos.spotlight", "macOS Spotlight"),
    ]

    for module, description in modules:
        try:
            __import__(module)
            results.add_pass(f"Import: {description} ({module})")
        except ImportError as e:
            results.add_fail(f"Import: {description} ({module})", str(e))
        except Exception as e:
            results.add_warning(f"Import warning: {description} ({module}): {e}")


def verify_test_coverage(results: VerificationResults) -> None:
    """Verify test files exist and can be discovered."""
    print_header("VERIFYING TEST COVERAGE")

    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"

    # Count test files
    test_files = list(tests_dir.rglob("test_*.py"))

    results.add_pass(f"Test files found: {len(test_files)}")

    # Check for specific test categories
    test_categories = {
        "unit": "tests/unit",
        "integration": "tests/integration",
        "benchmarks": "tests/benchmarks",
    }

    for category, path in test_categories.items():
        category_path = tests_dir / path
        if category_path.exists():
            count = len(list(category_path.glob("test_*.py")))
            results.add_pass(f"{category.capitalize()} tests: {count} files")
        else:
            results.add_fail(f"{category.capitalize()} tests", f"Directory not found: {path}")


def verify_documentation(results: VerificationResults) -> None:
    """Verify documentation is complete."""
    print_header("VERIFYING DOCUMENTATION")

    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    required_docs = [
        ("architecture.md", "Architecture documentation"),
        ("skills-guide.md", "Skills guide"),
        ("privacy-model.md", "Privacy model"),
        ("performance.md", "Performance guide"),
    ]

    for doc_file, description in required_docs:
        doc_path = docs_dir / doc_file
        if doc_path.exists():
            with open(doc_path) as f:
                content = f.read()
                if len(content) > 1000:  # Minimum content check
                    results.add_pass(f"Documentation: {description}")
                else:
                    results.add_warning(f"Documentation: {description} seems short")
        else:
            results.add_fail(f"Documentation: {description}", f"File not found: {doc_file}")

    # Check README
    readme_path = project_root / "README.md"
    if readme_path.exists():
        with open(readme_path) as f:
            content = f.read()
            if len(content) > 500:
                results.add_pass("README.md exists and has content")
            else:
                results.add_warning("README.md seems short")
    else:
        results.add_fail("README.md", "Not found")


def verify_scripts(results: VerificationResults) -> None:
    """Verify installation and setup scripts."""
    print_header("VERIFYING SCRIPTS")

    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"

    scripts = [
        ("install.sh", "Installation script"),
        ("setup_ollama.sh", "Ollama setup script"),
    ]

    for script, description in scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            # Check if executable
            if os.access(script_path, os.X_OK):
                results.add_pass(f"Script: {description} (executable)")
            else:
                results.add_warning(f"Script: {description} (not executable)")
        else:
            results.add_fail(f"Script: {description}", f"Not found: {script_path}")


def verify_skill_registration(results: VerificationResults) -> None:
    """Verify skills can be registered."""
    print_header("VERIFYING SKILL REGISTRATION")

    try:
        from roxy.skills.registry import SkillRegistry
        from roxy.skills.system import (
            AppLauncherSkill,
            FileSearchSkill,
            WindowManagerSkill,
        )

        registry = SkillRegistry()
        registry.reset()

        # Register some skills
        registry.register(AppLauncherSkill)
        registry.register(FileSearchSkill)
        registry.register(WindowManagerSkill)

        skills_list = registry.list_skills()

        if len(skills_list) >= 3:
            results.add_pass(f"Skill registration: {len(skills_list)} skills registered")
        else:
            results.add_fail("Skill registration", f"Only {len(skills_list)} skills registered")

    except Exception as e:
        results.add_fail("Skill registration", str(e))


async def verify_orchestrator(results: VerificationResults) -> None:
    """Verify orchestrator can be initialized."""
    print_header("VERIFYING ORCHESTRATOR")

    try:
        from roxy.config import RoxyConfig
        from roxy.brain.orchestrator import RoxyOrchestrator
        from roxy.skills.registry import SkillRegistry
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create test config
        config = RoxyConfig(
            name="TestRoxy",
            version="0.1.0-test",
            data_dir="/tmp/roxy_test",
        )

        registry = SkillRegistry()
        registry.reset()

        # Mock LLM to avoid actual calls
        with patch("roxy.brain.llm_clients.AsyncOpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage.total_tokens = 10
            mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

            # Create orchestrator
            orchestrator = RoxyOrchestrator(config, skill_registry=registry)
            results.add_pass("Orchestrator instantiation")

            # Initialize
            await orchestrator.initialize()
            results.add_pass("Orchestrator initialization")

            # Test process
            response = await orchestrator.process("Hello")
            if isinstance(response, str):
                results.add_pass("Orchestrator process()")
            else:
                results.add_fail("Orchestrator process()", f"Unexpected response type: {type(response)}")

            # Get stats
            stats = await orchestrator.get_statistics()
            if "timing_stats" in stats:
                results.add_pass("Orchestrator timing stats")
            else:
                results.add_warning("Orchestrator timing stats missing")

            # Shutdown
            await orchestrator.shutdown()
            results.add_pass("Orchestrator shutdown")

    except Exception as e:
        results.add_fail("Orchestrator", str(e))


def verify_memory_system(results: VerificationResults) -> None:
    """Verify memory system components."""
    print_header("VERIFYING MEMORY SYSTEM")

    try:
        from roxy.memory.manager import MemoryManager
        from roxy.config import RoxyConfig

        config = RoxyConfig(
            name="TestRoxy",
            version="0.1.0-test",
            data_dir="/tmp/roxy_test",
        )

        # Check manager has required methods
        manager_methods = [
            "initialize",
            "get_session_context",
            "search_history",
            "remember",
            "recall",
            "shutdown",
        ]

        for method in manager_methods:
            if hasattr(MemoryManager, method):
                results.add_pass(f"MemoryManager.{method}()")
            else:
                results.add_fail(f"MemoryManager.{method}()", "Method not found")

    except Exception as e:
        results.add_fail("Memory system", str(e))


def verify_voice_components(results: VerificationResults) -> None:
    """Verify voice pipeline components."""
    print_header("VERIFYING VOICE COMPONENTS")

    voice_files = [
        ("src/roxy/voice/stt.py", "STT engine"),
        ("src/roxy/voice/tts.py", "TTS engine"),
        ("src/roxy/voice/wake_word.py", "Wake word detector"),
    ]

    project_root = Path(__file__).parent.parent

    for file_path, description in voice_files:
        full_path = project_root / file_path
        if full_path.exists():
            results.add_pass(f"Voice component: {description}")
        else:
            results.add_warning(f"Voice component: {description} not found")


def verify_macos_integration(results: VerificationResults) -> None:
    """Verify macOS integration components."""
    print_header("VERIFYING MACOS INTEGRATION")

    macos_files = [
        ("src/roxy/macos/applescript.py", "AppleScript wrapper"),
        ("src/roxy/macos/spotlight.py", "Spotlight integration"),
        ("src/roxy/macos/menubar.py", "Menu bar"),
    ]

    project_root = Path(__file__).parent.parent

    for file_path, description in macos_files:
        full_path = project_root / file_path
        if full_path.exists():
            results.add_pass(f"macOS component: {description}")
        else:
            results.add_warning(f"macOS component: {description} not found")


def verify_performance_instrumentation(results: VerificationResults) -> None:
    """Verify performance instrumentation is in place."""
    print_header("VERIFYING PERFORMANCE INSTRUMENTATION")

    try:
        from roxy.brain.orchestrator import RoxyOrchestrator

        # Check for timing tracking methods
        if hasattr(RoxyOrchestrator, "_track_timing"):
            results.add_pass("Orchestrator timing tracking")
        else:
            results.add_fail("Orchestrator timing tracking", "_track_timing method not found")

        if hasattr(RoxyOrchestrator, "get_timing_stats"):
            results.add_pass("Orchestrator timing stats")
        else:
            results.add_fail("Orchestrator timing stats", "get_timing_stats method not found")

    except Exception as e:
        results.add_fail("Performance instrumentation", str(e))


def verify_cli_interface(results: VerificationResults) -> None:
    """Verify CLI interface is complete."""
    print_header("VERIFYING CLI INTERFACE")

    try:
        import click
        from roxy.main import main

        # Check if main is a click command
        if hasattr(main, "params"):
            results.add_pass("CLI command interface")

            # Check for expected options
            param_names = {p.name for p in main.params}
            expected_options = {"voice", "server", "no_memory", "cloud", "verbose"}

            for option in expected_options:
                if option in param_names:
                    results.add_pass(f"CLI option: --{option}")
                else:
                    results.add_warning(f"CLI option: --{option} not found")
        else:
            results.add_fail("CLI command interface", "main() is not a click command")

    except Exception as e:
        results.add_fail("CLI interface", str(e))


def main() -> int:
    """Run all verification checks."""
    print_header("ROXY INSTALLATION VERIFICATION")

    results = VerificationResults()

    # Run all verifications
    verify_project_structure(results)
    verify_config_files(results)
    verify_python_imports(results)
    verify_test_coverage(results)
    verify_documentation(results)
    verify_scripts(results)
    verify_skill_registration(results)
    verify_memory_system(results)
    verify_voice_components(results)
    verify_macos_integration(results)
    verify_performance_instrumentation(results)
    verify_cli_interface(results)

    # Async verifications
    asyncio.run(verify_orchestrator(results))

    # Print summary
    results.print_summary()

    # Exit with appropriate code
    if results.failed:
        print(f"{RED}Verification FAILED{NC}")
        print(f"  Please fix the {len(results.failed)} failed check(s) above.")
        return 1
    else:
        print(f"{GREEN}Verification PASSED{NC}")
        print(f"  All {len(results.passed)} checks passed!")
        if results.warnings:
            print(f"  {len(results.warnings)} warning(s) - review above.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
