# Roxy Skills Guide

## Overview

Skills are the primary way to extend Roxy's capabilities. Every feature—from opening apps to searching the web—is implemented as a skill. This guide explains how to create, register, and manage skills.

## What is a Skill?

A skill is a self-contained module that:

1. **Handles a specific type of request** (e.g., "open Safari", "search the web")
2. **Declares its permissions** (filesystem, network, etc.)
3. **Executes asynchronously** via the `execute()` method
4. **Returns a structured result** with response text and optional data

## Skill Anatomy

### Basic Structure

```python
from roxy.skills.base import RoxySkill, SkillContext, SkillResult, Permission

class MySkill(RoxySkill):
    """A brief description of what this skill does."""

    # Required metadata
    name: str = "my_skill"                      # Unique identifier
    description: str = "Does something useful"  # Human-readable description
    triggers: list[str] = ["do something",      # Example phrases that activate
                          "perform action"]     # this skill

    # Permission declarations
    permissions: list[Permission] = [Permission.NETWORK]

    # Cloud LLM requirement
    requires_cloud: bool = False  # Set to True if skill needs cloud LLM

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute the skill logic.

        Args:
            context: Contains user input, parameters, memory, config, etc.

        Returns:
            SkillResult with success status, response text, and optional data
        """
        # Your skill logic here

        return SkillResult(
            success=True,
            response_text="Task completed successfully",
            data={"key": "value"},  # Optional structured data
        )
```

### SkillContext

The `context` parameter provides everything your skill needs:

```python
@dataclass
class SkillContext:
    user_input: str                   # Raw user input
    intent: str                       # Classified intent
    parameters: dict[str, Any]        # Extracted parameters
    memory: MemoryManager             # Access to memory system
    config: RoxyConfig                # Full configuration
    conversation_history: list[dict]  # Conversation so far
```

### SkillResult

Return a `SkillResult` to tell Roxy what happened:

```python
@dataclass
class SkillResult:
    success: bool                     # Whether execution succeeded
    response_text: str                # What Roxy should say back
    data: dict[str, Any] | None      # Optional structured data
    speak: bool = True                # Whether to speak the response
    follow_up: str | None = None      # Suggested follow-up question
```

## Permission Types

Skills must declare the permissions they require:

| Permission       | Description                         | Example Skills              |
|------------------|-------------------------------------|-----------------------------|
| `FILESYSTEM`     | Read/write local files              | FileSearch, Notes          |
| `NETWORK`        | Make network requests               | WebSearch, Browse          |
| `SHELL`          | Execute shell commands              | SystemInfo, GitOps         |
| `MICROPHONE`     | Access microphone                   | VoicePipeline              |
| `NOTIFICATIONS`  | Send system notifications           | Reminders                  |
| `APPLESCRIPT`    | Execute AppleScript                 | AppLauncher, Clipboard     |
| `CLOUD_LLM`      | Use cloud LLM (with privacy gateway)| FlightSearch               |

## Creating Your First Skill

### Example: Simple macOS App Launcher

```python
import subprocess
from roxy.skills.base import RoxySkill, SkillContext, SkillResult, Permission

class AppLauncherSkill(RoxySkill):
    """Launch macOS applications by name."""

    name = "app_launcher"
    description = "Opens applications on macOS"
    triggers = ["open", "launch", "start"]
    permissions = [Permission.SHELL]

    async def execute(self, context: SkillContext) -> SkillResult:
        # Get the app name from user input
        words = context.user_input.split()
        if len(words) > 1:
            app_name = words[-1]  # Last word is usually the app name
        else:
            return SkillResult(
                success=False,
                response_text="Which app would you like me to open?",
            )

        try:
            # Use macOS 'open' command
            result = subprocess.run(
                ["open", "-a", app_name],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return SkillResult(
                    success=True,
                    response_text=f"Opening {app_name}...",
                )
            else:
                return SkillResult(
                    success=False,
                    response_text=f"Couldn't open {app_name}. Is it installed?",
                )

        except Exception as e:
            return SkillResult(
                success=False,
                response_text=f"Error opening {app_name}: {e}",
            )
```

## Registering Skills

### Automatic Discovery

Skills in the `roxy/skills/` directory are automatically discovered:

```
roxy/skills/
├── system/
│   ├── app_launcher.py    # Auto-discovered
│   ├── file_search.py     # Auto-discovered
│   └── ...
├── web/
│   ├── search.py          # Auto-discovered
│   └── ...
└── productivity/
    └── calendar.py        # Auto-discovered
```

### Manual Registration

Register skills explicitly in `main.py`:

```python
from roxy.skills.registry import SkillRegistry
from roxy.skills.system import AppLauncherSkill, FileSearchSkill

registry = SkillRegistry()
registry.register(AppLauncherSkill)
registry.register(FileSearchSkill)
```

## Advanced Features

### Memory Integration

Skills can store and retrieve information:

```python
async def execute(self, context: SkillContext) -> SkillResult:
    # Store in memory
    await context.memory.remember("user_preference", "dark_mode")

    # Recall from memory
    preferences = await context.memory.recall("user preference")

    return SkillResult(success=True, response_text="Done!")
```

### Error Handling

Always handle errors gracefully:

```python
async def execute(self, context: SkillContext) -> SkillResult:
    try:
        # Your skill logic
        result = await self._do_work(context)
        return SkillResult(success=True, response_text="Success!")

    except PermissionError as e:
        return SkillResult(
            success=False,
            response_text=f"I don't have permission to do that: {e}",
        )
    except Exception as e:
        logger.error(f"Skill error: {e}")
        return SkillResult(
            success=False,
            response_text="Something went wrong. Please try again.",
        )
```

## Testing Skills

### Unit Test Example

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

from roxy.skills.system import AppLauncherSkill

@pytest.mark.asyncio
async def test_app_launcher():
    """Test the app launcher skill."""
    skill = AppLauncherSkill()

    # Mock context
    context = SkillContext(
        user_input="open Safari",
        intent="open_app",
        parameters={"app": "Safari"},
        memory=MagicMock(),
        config=MagicMock(),
        conversation_history=[],
    )

    # Execute
    result = await skill.execute(context)

    # Verify
    assert result.success is True
    assert "Safari" in result.response_text
```

## Best Practices

1. **Be Declarative:** Clearly state what your skill does in the description
2. **Use Permissions:** Declare all required permissions upfront
3. **Handle Errors:** Never let exceptions propagate from `execute()`
4. **Provide Feedback:** Always return meaningful response text
5. **Use Data:** Return structured data for complex results
6. **Respect Privacy:** Redact sensitive information before cloud calls
7. **Be Idempotent:** Calling a skill twice should be safe
8. **Document Triggers:** Include common phrases users might say

## Examples Repository

See the `roxy/skills/` directory for more examples:

- `system/`: macOS integration skills
- `web/`: Web search and browsing skills
- `productivity/`: Calendar, email, notes skills
- `dev/`: Developer workflow skills

## Further Reading

- [Architecture Documentation](architecture.md)
- [Privacy Model](privacy-model.md)
- [Performance Guide](performance.md)
