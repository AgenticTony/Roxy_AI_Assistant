# CLAUDE.md — Roxy: Local-First AI Assistant for macOS

## Project Identity

**Roxy** is a voice-controlled, privacy-first AI assistant for macOS built by Anthony Foran. She runs primarily on-device using local LLMs, with optional cloud escalation for complex tasks. Roxy can control the system, search the web, manage files, remember conversations, and perform tasks like finding cheap flights — all via voice or text.

**Owner:** Anthony Foran (Malmö, Sweden)
**License:** MIT
**Language:** Python 3.12+
**Target:** macOS on Apple Silicon (M1+, 16GB+ RAM)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    VOICE LAYER                          │
│  OpenWakeWord → Talon (commands) / whisper.cpp (dictation) │
│  Kokoro TTS via MLX-Audio (speech output)               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   BRAIN (Orchestrator)                   │
│  Agno framework — single-agent with tool access          │
│  Intent classification → Skill dispatch                  │
│  Confidence-based routing (local vs cloud)                │
└──┬───────────┬───────────┬───────────┬──────────────────┘
   │           │           │           │
┌──▼──┐  ┌────▼────┐  ┌───▼───┐  ┌───▼────┐
│Memory│  │ Skills  │  │  Web  │  │ macOS  │
│System│  │ Engine  │  │Access │  │ Hooks  │
└─────┘  └─────────┘  └───────┘  └────────┘
```

### Core Design Principles

1. **Local-first:** 80%+ of requests handled by Ollama (Qwen3 8B) with zero data leaving the device
2. **Privacy-gated cloud:** Cloud LLM calls (ChatGLM/Z.ai) require PII redaction and user awareness
3. **Plugin architecture:** Every capability is a Skill with declared permissions
4. **MCP-native:** External integrations use Model Context Protocol where possible
5. **Voice-native:** Designed for voice-first interaction, keyboard optional

---

## Project Structure

```
~/roxy/
├── CLAUDE.md                    # This file — project spec
├── pyproject.toml               # Project config and dependencies
├── .env                         # API keys (NEVER commit)
├── .env.example                 # Template for .env
├── src/
│   ├── roxy/
│   │   ├── __init__.py
│   │   ├── main.py              # Entry point — starts all services
│   │   ├── config.py            # Configuration management
│   │   │
│   │   ├── brain/               # Core orchestrator
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py  # Agno agent setup, main loop
│   │   │   ├── router.py        # Confidence-based local/cloud routing
│   │   │   ├── privacy.py       # PII detection and redaction gateway
│   │   │   └── llm_clients.py   # Ollama + ChatGLM client wrappers
│   │   │
│   │   ├── voice/               # Voice pipeline
│   │   │   ├── __init__.py
│   │   │   ├── stt.py           # whisper.cpp integration
│   │   │   ├── tts.py           # Kokoro via MLX-Audio
│   │   │   ├── wake_word.py     # OpenWakeWord "Hey Roxy" detection
│   │   │   └── talon_bridge.py  # Talon Voice command bridge
│   │   │
│   │   ├── memory/              # Three-tier memory system
│   │   │   ├── __init__.py
│   │   │   ├── session.py       # Tier 1: In-memory session context
│   │   │   ├── history.py       # Tier 2: SQLite + sqlite-vec conversation store
│   │   │   ├── longterm.py      # Tier 3: Mem0 persistent memory
│   │   │   └── manager.py       # Unified memory interface
│   │   │
│   │   ├── skills/              # Plugin/skill system
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # Abstract RoxySkill base class
│   │   │   ├── registry.py      # Skill discovery and registration
│   │   │   ├── permissions.py   # Permission declarations and checks
│   │   │   │
│   │   │   ├── system/          # macOS system skills
│   │   │   │   ├── app_launcher.py
│   │   │   │   ├── file_search.py
│   │   │   │   ├── window_manager.py
│   │   │   │   ├── system_info.py
│   │   │   │   ├── clipboard.py
│   │   │   │   └── shortcuts.py
│   │   │   │
│   │   │   ├── web/             # Web access skills
│   │   │   │   ├── search.py
│   │   │   │   ├── browse.py
│   │   │   │   └── flights.py
│   │   │   │
│   │   │   ├── productivity/    # Productivity skills
│   │   │   │   ├── calendar.py
│   │   │   │   ├── email.py
│   │   │   │   ├── notes.py
│   │   │   │   └── reminders.py
│   │   │   │
│   │   │   └── dev/             # Developer skills
│   │   │       ├── git_ops.py
│   │   │       ├── project_manager.py
│   │   │       └── claude_code.py
│   │   │
│   │   ├── macos/               # macOS integration layer
│   │   │   ├── __init__.py
│   │   │   ├── applescript.py   # osascript wrapper
│   │   │   ├── pyobjc_bridge.py # PyObjC native framework access
│   │   │   ├── hammerspoon.py   # Hammerspoon HTTP client
│   │   │   ├── spotlight.py     # mdfind / NSMetadataQuery wrapper
│   │   │   └── menubar.py       # rumps menu bar application
│   │   │
│   │   └── mcp/                 # MCP server configs and custom servers
│   │       ├── __init__.py
│   │       ├── servers.py       # MCP server manager
│   │       └── custom/          # Custom MCP servers for Roxy
│   │           └── roxy_system.py
│   │
│   └── scripts/                 # CLI utilities and setup scripts
│       ├── setup_ollama.sh      # Download and configure Ollama models
│       ├── setup_searxng.sh     # Docker-compose for SearXNG
│       └── install.sh           # Full installation script
│
├── talon/                       # Talon Voice scripts
│   ├── roxy.talon               # Main "Roxy" activation
│   ├── apps.talon               # App control commands
│   ├── system.talon             # System commands
│   └── dev.talon                # Dev workflow commands
│
├── config/                      # Configuration files
│   ├── default.yaml             # Default settings
│   ├── skills.yaml              # Skill enable/disable
│   └── privacy.yaml             # Privacy rules and PII patterns
│
├── data/                        # Local data (gitignored)
│   ├── memory.db                # SQLite conversation + vector store
│   ├── mem0/                    # Mem0 persistent memory data
│   └── chromadb/                # ChromaDB vector collections
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
└── docs/
    ├── architecture.md
    ├── skills-guide.md
    └── privacy-model.md
```

---

## Key Interfaces and Contracts

### RoxySkill Base Class

Every capability MUST be a skill. No raw functionality in the orchestrator.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

class Permission(Enum):
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    SHELL = "shell"
    MICROPHONE = "microphone"
    NOTIFICATIONS = "notifications"
    APPLESCRIPT = "applescript"
    CLOUD_LLM = "cloud_llm"

@dataclass
class SkillContext:
    """Passed to every skill execution."""
    user_input: str
    intent: str
    parameters: dict[str, Any]
    memory: "MemoryManager"
    config: "RoxyConfig"
    conversation_history: list[dict]

@dataclass
class SkillResult:
    """Returned by every skill execution."""
    success: bool
    response_text: str           # What Roxy says back
    data: dict[str, Any] | None = None  # Structured data if applicable
    speak: bool = True           # Whether to speak the response
    follow_up: str | None = None # Suggested follow-up question

class RoxySkill(ABC):
    """Base class for all Roxy skills."""
    name: str
    description: str
    triggers: list[str]          # Example phrases that activate this skill
    permissions: list[Permission]
    requires_cloud: bool = False

    @abstractmethod
    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute the skill. Must be implemented by all skills."""
        ...

    def can_handle(self, intent: str, parameters: dict) -> float:
        """Return confidence 0.0-1.0 that this skill handles the intent."""
        ...
```

### Memory Manager Interface

```python
class MemoryManager:
    """Unified interface to three-tier memory."""

    async def get_session_context(self) -> list[dict]:
        """Tier 1: Current conversation messages."""

    async def search_history(self, query: str, limit: int = 5) -> list[dict]:
        """Tier 2: Semantic search over conversation history."""

    async def remember(self, key: str, value: str) -> None:
        """Tier 3: Store a long-term fact via Mem0."""

    async def recall(self, query: str) -> list[str]:
        """Tier 3: Retrieve relevant long-term memories."""

    async def get_user_preferences(self) -> dict:
        """Tier 3: Get stored user preferences."""
```

### Confidence Router Interface

```python
class ConfidenceRouter:
    """Routes requests between local and cloud LLMs."""

    async def route(self, request: str, context: SkillContext) -> LLMResponse:
        """
        1. Send to local LLM (Qwen3 8B via Ollama)
        2. If confidence >= threshold (0.7 default) → return local response
        3. If confidence < threshold → run through privacy gateway → cloud LLM
        4. Return response with source metadata
        """
```

---

## Technology Stack

| Component | Tool | Version | Install |
|-----------|------|---------|---------|
| Python | CPython | 3.12+ | brew install python@3.12 |
| Package mgr | uv | latest | brew install uv |
| Local LLM runtime | Ollama | 0.9+ | brew install ollama |
| Local model (main) | Qwen3 8B Q4_K_M | — | ollama pull qwen3:8b |
| Local model (fast) | Qwen3 0.6B | — | ollama pull qwen3:0.6b |
| Cloud LLM | ChatGLM / Z.ai API | GLM-4.7 | API key in .env |
| Agent framework | Agno | latest | uv add agno |
| Memory (persistent) | Mem0 | latest | uv add mem0ai |
| Vector DB | ChromaDB | latest | uv add chromadb |
| SQLite vectors | sqlite-vec | latest | uv add sqlite-vec |
| RAG framework | LlamaIndex | latest | uv add llama-index |
| TTS | Kokoro via MLX-Audio | latest | uv add mlx-audio |
| STT | faster-whisper | latest | uv add faster-whisper |
| Wake word | OpenWakeWord | latest | uv add openwakeword |
| Web search | Brave Search API | — | MCP server |
| Browser automation | Browser-Use | latest | uv add browser-use |
| macOS bridge | PyObjC | 12+ | uv add pyobjc |
| Menu bar | rumps | latest | uv add rumps |
| Automation | Hammerspoon | latest | brew install hammerspoon |
| Voice commands | Talon Voice | latest | talonvoice.com |
| MCP SDK | mcp | latest | uv add mcp |

---

## Configuration

### Environment Variables (.env)

```bash
# LLM Providers
OLLAMA_HOST=http://localhost:11434
ZAI_API_KEY=your-zhipu-api-key
ZAI_BASE_URL=https://api.z.ai/api/paas/v4

# Cloud fallback (optional — OpenRouter for multi-provider)
OPENROUTER_API_KEY=optional-key

# Web Search
BRAVE_SEARCH_API_KEY=your-brave-key

# Privacy
ROXY_CLOUD_CONSENT_MODE=ask    # ask | always | never
ROXY_PII_REDACTION=true

# Voice
ROXY_WAKE_WORD=hey_roxy
ROXY_TTS_VOICE=af_heart         # Kokoro voice ID
ROXY_TTS_SPEED=1.1

# General
ROXY_LOG_LEVEL=INFO
ROXY_DATA_DIR=~/roxy/data
```

### Default Config (config/default.yaml)

```yaml
roxy:
  name: "Roxy"
  version: "0.1.0"

llm:
  local:
    model: "qwen3:8b"
    router_model: "qwen3:0.6b"
    temperature: 0.7
    max_tokens: 2048
  cloud:
    provider: "zai"              # zai | openrouter
    model: "glm-4.7"
    confidence_threshold: 0.7    # Below this → escalate to cloud

memory:
  session_max_messages: 50
  history_db: "data/memory.db"
  chromadb_path: "data/chromadb"
  mem0_config:
    llm_provider: "ollama"
    llm_model: "qwen3:8b"
    embedder_provider: "ollama"
    embedder_model: "nomic-embed-text"

voice:
  stt_model: "base.en"          # whisper model size
  tts_engine: "kokoro"
  wake_word_sensitivity: 0.6
  speak_responses: true

privacy:
  redact_patterns:
    - email
    - phone
    - ssn
    - credit_card
    - address
  cloud_consent: "ask"           # ask | always | never
  log_cloud_requests: true
```

---

## Coding Standards

### General Rules

- **Type hints everywhere.** Every function signature must have complete type annotations.
- **Async by default.** All skill execution and I/O operations use `async/await`.
- **Docstrings on all public functions.** Google style.
- **No global state.** Pass dependencies via constructors or context objects.
- **Errors as values where possible.** Use `SkillResult(success=False)` not raw exceptions for expected failures. Reserve exceptions for truly exceptional conditions.

### Naming Conventions

- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Skills: class name ends with `Skill` (e.g., `FlightSearchSkill`)
- MCP servers: class name ends with `Server` (e.g., `RoxySystemServer`)

### Import Order

1. Standard library
2. Third-party packages
3. Local imports (relative within package)

Blank line between each group. Use `from __future__ import annotations` in every file.

### Testing

- **pytest** for all tests
- **pytest-asyncio** for async tests
- Minimum one test per skill's `execute()` method
- Mock external services (Ollama, APIs) in unit tests
- Integration tests can hit local Ollama but NEVER cloud APIs
- Target: 80%+ coverage on `brain/`, `memory/`, `skills/`

### Logging

```python
import logging
logger = logging.getLogger(__name__)

# Use structured logging
logger.info("Skill executed", extra={"skill": self.name, "duration_ms": elapsed})
```

---

## Privacy Model

### Three Rules

1. **Default local.** Never send data to cloud unless explicitly required by the task AND confidence routing determines local can't handle it.
2. **Redact before sending.** The privacy gateway MUST run PII detection on all cloud-bound requests. Detected PII is replaced with placeholders.
3. **User controls everything.** `ROXY_CLOUD_CONSENT_MODE=ask` means Roxy announces cloud usage before making the call. User can set to `always` (auto-approve) or `never` (block all cloud).

### What Roxy NEVER sends to cloud

- File contents (unless user explicitly asks "summarize this file using the cloud model")
- Passwords, API keys, tokens
- Health or financial information
- Memory/conversation history

### Logging

- All cloud requests are logged to `data/cloud_requests.log` with timestamp, redacted prompt, provider, and response summary
- User can review and delete at any time

---

## MCP Servers

### Pre-configured (Anthony already has)

- **Brave Search** — Web search
- **Firecrawl** — Web scraping
- **GitHub** — Repository operations
- **Dart** — Task management

### To Add

- **Filesystem** (Anthropic reference server) — Local file read/write/search
- **Google Calendar** (`@cocal/google-calendar-mcp`) — Calendar management
- **Kiwi.com Flights** — Flight search
- **Context7** — Up-to-date dev documentation

### Custom MCP Server Pattern

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("roxy-macos")

@mcp.tool()
async def open_application(name: str) -> str:
    """Open a macOS application by name."""
    result = subprocess.run(["open", "-a", name], capture_output=True)
    return f"Opened {name}" if result.returncode == 0 else f"Failed: {result.stderr}"

@mcp.tool()
async def spotlight_search(query: str, limit: int = 10) -> list[str]:
    """Search files using macOS Spotlight."""
    result = subprocess.run(["mdfind", "-limit", str(limit), query], capture_output=True, text=True)
    return result.stdout.strip().split("\n")
```

---

## Build Phases

### Phase 1: Foundation (Single Claude Code session)

- [ ] Initialize project with `uv init`
- [ ] Create this directory structure
- [ ] Set up pyproject.toml with all dependencies
- [ ] Create config loading (`config.py`)
- [ ] Create `.env.example`
- [ ] Verify Ollama runs with Qwen3 8B

### Phase 2: Core Brain + Memory (Agent Team — 2 teammates)

**brain-architect:**
- [ ] Implement `orchestrator.py` with Agno agent
- [ ] Implement `router.py` confidence-based routing
- [ ] Implement `privacy.py` PII gateway
- [ ] Implement `llm_clients.py` (Ollama + Z.ai wrappers)
- [ ] Create basic text-mode REPL for testing

**memory-builder:**
- [ ] Implement `session.py` (in-memory context)
- [ ] Implement `history.py` (SQLite + sqlite-vec)
- [ ] Implement `longterm.py` (Mem0 + Ollama)
- [ ] Implement `manager.py` (unified interface)
- [ ] Write memory integration tests

### Phase 3: Capabilities (Agent Team — 3 teammates)

**voice-engineer:**
- [ ] Implement `wake_word.py` (OpenWakeWord)
- [ ] Implement `stt.py` (faster-whisper)
- [ ] Implement `tts.py` (Kokoro/MLX-Audio)
- [ ] Implement `talon_bridge.py`
- [ ] End-to-end voice test: wake → listen → process → speak

**macos-integrator:**
- [ ] Implement `applescript.py` wrapper library
- [ ] Implement `pyobjc_bridge.py` (NSWorkspace, Spotlight)
- [ ] Implement `hammerspoon.py` HTTP client
- [ ] Implement `menubar.py` (rumps status bar)
- [ ] Create system skills: app_launcher, file_search, window_manager, clipboard
- [ ] Implement `shortcuts.py` (macOS Shortcuts CLI bridge)

**web-connector:**
- [ ] Configure Brave Search MCP
- [ ] Set up SearXNG via Docker
- [ ] Implement Browser-Use integration for interactive browsing
- [ ] Implement Kiwi.com flight search skill
- [ ] Create privacy gateway for all outbound requests
- [ ] Create web search skill, browse skill, flights skill

### Phase 4: Integration + Polish (Single session)

- [ ] Wire voice pipeline → brain → skills → voice output
- [ ] Add Talon Voice command scripts
- [ ] Create installation script (`scripts/install.sh`)
- [ ] End-to-end testing of all voice commands
- [ ] Performance profiling (target: <2s response time for local queries)
- [ ] Documentation

---

## Voice Commands (Target)

| Command | Skill | Action |
|---------|-------|--------|
| "Hey Roxy" | wake_word | Activate listening |
| "Open [app name]" | app_launcher | Launch application |
| "Search for [query]" | web_search | Brave Search + summarize |
| "Find me flights to [destination]" | flights | Kiwi.com search |
| "What's on my calendar today?" | calendar | Read today's events |
| "Remember that [fact]" | memory | Store in Mem0 |
| "What do you know about [topic]?" | memory + search | Recall + web fallback |
| "Open the file about [topic]" | file_search | Spotlight → open |
| "Start development on [project]" | dev_workflow | Cursor → Terminal → Claude Code |
| "What's running?" | system_info | List active processes |
| "Set up my coding layout" | window_manager | Hammerspoon workspace |
| "Read my latest emails" | email | Mail.app via AppleScript |
| "How are you Roxy?" | conversation | Personality response |

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Wake word detection | <200ms |
| Intent classification (Qwen3 0.6B) | <500ms |
| Local LLM response (Qwen3 8B) | <2s for short responses |
| Cloud LLM response (GLM-4.7) | <5s including privacy gateway |
| TTS generation (Kokoro) | <300ms |
| Memory search (sqlite-vec) | <100ms |
| Skill execution (system commands) | <500ms |
| End-to-end voice → voice (local) | <4s |

---

## Agent Team Instructions

When working as part of an Agent Team:

1. **Stay in your lane.** Only modify files in your designated module. If you need an interface from another module, define it in your module and document the contract — the other teammate will implement it.
2. **Interfaces first.** Before implementing, define the public interface (function signatures, dataclasses) and get plan approval.
3. **Use the skill base class.** Every new capability MUST extend `RoxySkill`. No exceptions.
4. **Test as you go.** Write tests alongside implementation, not after.
5. **Log your decisions.** Add comments explaining WHY, not WHAT. The code shows what; comments should explain architectural choices.
6. **Don't install system packages** without documenting them in the install script.
7. **Respect the privacy model.** Never hardcode API calls to external services without going through the privacy gateway.
