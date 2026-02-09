# Roxy Architecture

## Overview

Roxy is a local-first AI assistant for macOS designed with privacy, extensibility, and voice interaction as core principles. This document explains the system architecture, key components, and design decisions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACES                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │   REPL   │  │   Voice  │  │      Menu Bar        │  │
│  │  (Text)  │  │ Pipeline │  │    (Background)      │  │
│  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘  │
└───────┼─────────────┼───────────────────┼──────────────┘
        │             │                   │
        └─────────────┴───────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   ORCHESTRATOR                           │
│  • Intent classification                                │
│  • Skill dispatch & routing                             │
│  • Confidence-based local/cloud decision                │
│  • Conversation management                              │
└───┬─────────┬─────────┬─────────┬─────────┬────────────┘
    │         │         │         │         │
┌───▼───┐ ┌──▼──┐  ┌───▼───┐  ┌──▼───┐  ┌───▼────┐
│Memory │ │Skills│  │Privacy│  │  LLM │  │   MCP  │
│System │ │Registry│Gateway│ │Clients│  │Servers │
└───────┘ └─────┘  └───────┘  └──────┘  └────────┘
    │         │         │         │         │
┌───▼───────▼─────────▼─────────▼─────────▼──────────┐
│                 INFRASTRUCTURE                        │
│  • Ollama (Local LLM)                                 │
│  • SQLite + sqlite-vec (History)                      │
│  • Mem0 (Long-term Memory)                            │
│  • macOS APIs (AppleScript, PyObjC, Spotlight)        │
│  • Browser Automation (Playwright)                    │
└──────────────────────────────────────────────────────┘
```

## Core Components

### 1. Orchestrator (`brain/orchestrator.py`)

The orchestrator is the central coordinator that routes user input through the system.

**Responsibilities:**
- Receive user input from any interface (text, voice)
- Classify intent and route to appropriate skill
- Manage conversation history
- Coordinate between local and cloud LLMs
- Track performance metrics

**Key Methods:**
- `process(user_input: str) -> str`: Main entry point
- `dispatch_to_skill(intent, context) -> SkillResult`: Skill routing
- `get_statistics() -> dict`: Performance and usage metrics

### 2. Skill System (`skills/`)

Every Roxy capability is implemented as a skill. This plugin architecture allows easy extension.

**Base Class:** `RoxySkill` (`skills/base.py`)

```python
class RoxySkill(ABC):
    name: str                          # Unique identifier
    description: str                   # What this skill does
    triggers: list[str]                # Phrases that activate this skill
    permissions: list[Permission]      # Required permissions
    requires_cloud: bool = False       # Whether cloud LLM is needed

    @abstractmethod
    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute the skill logic."""
```

**Skill Registry** (`skills/registry.py`):
- Discovers and registers all skills
- Finds best-matching skill for a given input
- Manages skill lifecycle

### 3. Memory System (`memory/`)

Three-tier memory architecture for different persistence and retrieval needs:

**Tier 1: Session Memory** (`memory/session.py`)
- In-memory conversation context
- Automatically managed during a session
- Cleared on shutdown

**Tier 2: History** (`memory/history.py`)
- SQLite database with sqlite-vec for semantic search
- Stores conversation history with embeddings
- Enables "what did we talk about?" queries

**Tier 3: Long-term** (`memory/longterm.py`)
- Mem0 integration for persistent facts
- Stores user preferences, facts, and learned information
- Survives across sessions

**Memory Manager** (`memory/manager.py`):
- Unified interface to all three tiers
- Automatic routing based on query type
- Smart fallback (session → history → long-term)

### 4. Privacy Gateway (`brain/privacy.py`)

Protects user data when cloud LLM is used.

**Features:**
- PII detection (email, phone, SSN, credit card, address)
- Automatic redaction with placeholders
- Configurable consent modes (never/ask/always)
- Audit logging of all cloud requests

**Flow:**
```
User Input → [Local LLM Check] → [Confidence Score]
                     ↓
           High confidence? → Local response
                     ↓
           Low confidence? → [Privacy Gateway]
                                ↓
                     [PII Redaction] → Cloud LLM
                                ↓
                    [Log & Return Response]
```

### 5. LLM Clients (`brain/llm_clients.py`)

Unified interface for different LLM providers.

**Local Client:** `OllamaClient`
- Connects to local Ollama server
- Uses Qwen3 8B for main responses
- Uses Qwen3 0.6B for fast confidence scoring

**Cloud Client:** `CloudLLMClient`
- Supports Z.ai (ChatGLM), OpenRouter
- Automatic retry on rate limits
- Connection pooling for performance

**Confidence Scorer:** `ConfidenceScorer`
- Uses small local model to assess if request can be handled locally
- Returns score 0.0-1.0
- Threshold determines local vs cloud routing

### 6. Voice Pipeline (`voice/`)

Complete speech interaction pipeline.

**Components:**
- **Wake Word** (`wake_word.py`): OpenWakeWord for "Hey Roxy" detection
- **STT** (`stt.py`): faster-whisper for speech-to-text
- **TTS** (`tts.py`): Kokoro via MLX-Audio for text-to-speech
- **Pipeline** (`pipeline.py`): Coordinates all components

**Flow:**
```
Audio Input → [Wake Word Detection] → [Listening Mode]
                     ↓
              [Speech Capture] → [STT] → [Text]
                                            ↓
                                  [Orchestrator Processing]
                                            ↓
                                      [Response Text]
                                            ↓
                                          [TTS] → [Audio Output]
```

### 7. macOS Integration (`macos/`)

Bridge to macOS system capabilities.

**Components:**
- **AppleScript** (`applescript.py`): Run osascript commands
- **PyObjC Bridge** (`pyobjc_bridge.py`): Native framework access
- **Spotlight** (`spotlight.py`): File search via mdfind
- **Hammerspoon** (`hammerspoon.py`): Window management
- **Menu Bar** (`menubar.py`): Background service UI

### 8. MCP Integration (`mcp/`)

Model Context Protocol for external service integration.

**Features:**
- Server manager for lifecycle
- Auto-discovery of configured servers
- Tool forwarding to LLM

**Pre-configured Servers:**
- Brave Search (web search)
- Filesystem (local file operations)
- GitHub (repository operations)

## Design Decisions

### Why Local-First?

1. **Privacy:** Sensitive data never leaves the device
2. **Latency:** Local models respond in <2s vs 5-10s for cloud
3. **Cost:** No API fees for 80%+ of queries
4. **Reliability:** Works offline, no rate limits

### Why Agno Framework?

1. **Single-Agent Architecture:** Simpler than multi-agent for personal assistant
2. **Tool Access:** Native support for function calling
3. **Memory Integration:** Built-in conversation history
4. **Provider Support:** Easy switching between LLM providers

### Why Three-Tier Memory?

1. **Session:** Fast, temporary context (current conversation)
2. **History:** Searchable past conversations with semantic search
3. **Long-term:** Persistent facts and preferences

Each tier has different characteristics:

| Tier       | Storage   | Search      | Scope       | Persistence |
|------------|-----------|-------------|-------------|-------------|
| Session    | In-memory | None        | Current     | Session     |
| History    | SQLite    | Vector      | All time    | Permanent   |
| Long-term  | Mem0      | Vector      | All time    | Permanent   |

### Why Skill-Based Architecture?

1. **Modularity:** Each capability is independent
2. **Testability:** Skills can be unit tested in isolation
3. **Permissions:** Explicit declaration of required access
4. **Extensibility:** Third parties can add skills

### Why Privacy Gateway?

1. **Default Local:** Most queries never touch cloud
2. **User Control:** Explicit consent for cloud usage
3. **Transparency:** All cloud requests logged
4. **Redaction:** PII automatically removed from cloud requests

## Performance Targets

| Operation                    | Target  | Measurement Point        |
|------------------------------|---------|--------------------------|
| Wake word detection          | <200ms  | Audio → Detection        |
| Intent classification        | <500ms  | Input → Skill selected   |
| Local LLM response           | <2s     | Input → Response text    |
| Cloud LLM response           | <5s     | Input → Response text    |
| TTS generation               | <300ms  | Text → Audio             |
| End-to-end voice (local)     | <4s     | Wake → Speak             |
| Memory search                | <100ms  | Query → Results          |

## Security Considerations

### Local Model Security

- Models run in user space, no privileged access
- Model files validated on load
- No network access for local inference

### Cloud Security

- All cloud requests go through privacy gateway
- API keys stored in environment variables only
- Request/response logging for audit
- Consent required before cloud usage

### macOS Permissions

- Skills declare required permissions
- System prompts for authorization when needed
- No hardcoded credentials

## Extension Points

### Adding New Skills

1. Create class extending `RoxySkill`
2. Implement `execute()` method
3. Add to skill registry (or auto-discovery)
4. Declare permissions

### Adding New LLM Providers

1. Implement `LLMClient` protocol
2. Add to configuration
3. Update `CloudLLMClient` if needed

### Adding New MCP Servers

1. Add server config to `config/mcp_servers.yaml`
2. Server auto-starts if enabled
3. Tools available to orchestrator

## Future Enhancements

### Planned

- [ ] Multi-modal support (images, documents)
- [ ] Skill marketplace
- [ ] Mobile app companion
- [ ] Custom model fine-tuning
- [ ] Distributed memory across devices

### Under Consideration

- [ ] Multi-user support
- [ ] Cloud backup encryption
- [ ] Local model quantization
- [ ] GPU acceleration for inference

## References

- [Agno Framework](https://github.com/emmeth/agno)
- [Ollama](https://ollama.com)
- [Mem0](https://mem0.ai)
- [OpenWakeWord](https://github.com/dscripka/openWakeWord)
- [Kokoro TTS](https://github.com/remsky/Kokoro-FastAPI)
