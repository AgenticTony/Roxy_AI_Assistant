<p align="center">
  <h1 align="center">🎙️ Roxy</h1>
  <p align="center"><strong>Privacy-first voice AI assistant for macOS</strong></p>
  <p align="center">
    Talk to your Mac. Control your apps. Search the web. Remember everything.<br>
    All running locally on your hardware. No cloud required.
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-what-can-roxy-do">Features</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-build-your-own-skill">Build a Skill</a> •
    <a href="#-contributing">Contributing</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+">
    <img src="https://img.shields.io/badge/platform-macOS-lightgrey.svg" alt="macOS">
    <img src="https://img.shields.io/badge/chip-Apple%20Silicon-orange.svg" alt="Apple Silicon">
    <img src="https://img.shields.io/badge/license-AGPL--3.0-green.svg" alt="AGPL-3.0">
    <img src="https://img.shields.io/badge/LLM-local--first-purple.svg" alt="Local-first">
  </p>
</p>

---

## Why Roxy?

Every AI assistant sends your data to the cloud. Your emails, your calendar, your voice, your files — all shipped off to someone else's server.

**Roxy doesn't.** She runs a language model directly on your Mac. 80% of requests never leave your machine. When she does need cloud help for complex reasoning, she strips out your personal information first and asks your permission.

She's not a chatbot in a window. She's a **voice-controlled system assistant** that opens your apps, searches your files, reads your email, finds cheap flights, manages your calendar, remembers your conversations, and controls your desktop — all by voice.

And she gets smarter the more you use her.

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/AgenticTony/Roxy_AI_Assistant.git
cd roxy
bash scripts/install.sh

# Start talking
uv run roxy --voice
```

Or start in text mode first:

```bash
uv run roxy
```

**Requirements:** macOS 13+ (Ventura), Apple Silicon (M1/M2/M3/M4), 16GB+ RAM, Python 3.12+

---

## 🎤 What Can Roxy Do?

### Control Your Mac

| Say this | Roxy does this |
|----------|----------------|
| "Hey Roxy, open Cursor" | Launches the app |
| "Coding layout" | Arranges windows via Hammerspoon |
| "Find the PDF about taxes" | Spotlight search → opens the file |
| "What's running?" | Lists active apps, CPU, disk, battery |
| "Run my morning shortcut" | Triggers macOS Shortcuts |
| "Copy that" | Manages clipboard with history |

### Search & Browse

| Say this | Roxy does this |
|----------|----------------|
| "Search for AI news" | Brave Search (private) → summarises results |
| "Find flights to London" | Searches, compares, returns top 5 with prices |
| "Read that article" | Headless browser → extracts → summarises locally |

### Manage Your Day

| Say this | Roxy does this |
|----------|----------------|
| "What's on my calendar?" | Reads today's events from Calendar.app |
| "Any new emails?" | Checks Mail.app, summarises unread |
| "Create a note about the meeting" | Creates in Notes.app |
| "Remind me to call Maria at 3" | Sets reminder in Reminders.app |

### Remember Things

| Say this | Roxy does this |
|----------|----------------|
| "Remember my dog's name is Bella" | Stores in long-term memory |
| "What's my dog's name?" | Recalls from memory: "Bella" |
| "What did we talk about yesterday?" | Searches conversation history |

### Developer Tools

| Say this | Roxy does this |
|----------|----------------|
| "Git status" | Runs git commands in your project |
| "Start development" | Opens IDE + terminal + Claude Code |
| "Generate a commit message" | Analyses your diff, suggests message |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VOICE LAYER                            │
│  OpenWakeWord/Porcupine → whisper.cpp STT → Kokoro TTS     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    BRAIN (Orchestrator)                      │
│  Agno framework → Intent classification → Skill dispatch    │
│  Confidence-based routing: local (80%) vs cloud (20%)       │
│  Privacy gateway: PII redaction before any cloud call       │
└──┬──────────┬────────────┬──────────────┬───────────────────┘
   │          │            │              │
┌──▼──┐  ┌───▼────┐  ┌────▼─────┐  ┌────▼────┐
│Mem  │  │Skills  │  │ Web     │  │ macOS  │
│3-tier│  │16 built│  │ Access  │  │ Hooks  │
│memory│  │-in     │  │ Private │  │ Deep   │
└─────┘  └────────┘  └──────────┘  └────────┘
```

### Local-First, Cloud-When-Needed

Roxy uses **confidence-based routing**. Every request goes to the local LLM first (Qwen3 8B via Ollama). If the local model is confident in its response, that's what you get — instant, private, free.

If confidence is low, Roxy can escalate to a cloud LLM. But first:

1. The **privacy gateway** scans for personal information (emails, phone numbers, names, addresses, ID numbers)
2. Detected PII is **replaced with placeholders** before the request leaves your machine
3. Depending on your settings, Roxy either **asks permission** or handles it automatically
4. All cloud requests are **logged** so you can audit exactly what was sent

You control the behaviour:

```yaml
# config/default.yaml
privacy:
  cloud_consent: "ask"    # ask | always | never
```

Set it to `never` and Roxy runs 100% offline.

### Three-Tier Memory

Roxy remembers things across conversations:

- **Session memory** — current conversation context (in-memory)
- **Conversation history** — past conversations with semantic search (SQLite + sqlite-vec)
- **Long-term memory** — persistent facts about you, extracted automatically via Mem0 (ChromaDB)

Ask her something you told her last week — she'll remember.

### Skill Plugin System

Every capability is a **Skill** — a Python class with declared permissions, trigger phrases, and an `execute()` method. Adding a new capability is one file:

```python
class WeatherSkill(RoxySkill):
    name = "weather"
    description = "Get current weather for a location"
    triggers = ["weather in", "what's the weather", "is it raining"]
    permissions = [Permission.NETWORK]

    async def execute(self, context: SkillContext) -> SkillResult:
        # Your implementation here
        return SkillResult(success=True, response_text="It's 3°C in Malmö")
```

Drop it in `src/roxy/skills/`, restart Roxy, and she can do it.

---

## 📦 Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Local LLM | [Ollama](https://ollama.com) + Qwen3 8B | Fast inference on Apple Silicon, function calling support |
| Agent Framework | [Agno](https://github.com/agno-agi/agno) | Model-agnostic, 5000× faster instantiation than LangGraph |
| Memory | [Mem0](https://github.com/mem0ai/mem0) + ChromaDB + SQLite | Three-tier: session → history → long-term |
| Voice (STT) | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Real-time transcription with Metal acceleration |
| Voice (TTS) | [Kokoro](https://github.com/lucasnewman/mlx-audio) via MLX-Audio | Natural speech, <300ms generation |
| Wake Word | Porcupine / OpenWakeWord / Whisper fallback | Three backends, auto-selects best available |
| Web Search | [Brave Search API](https://brave.com/search/api/) | Private, no tracking, 2000 free queries/month |
| Browser | [Browser-Use](https://github.com/browser-use/browser-use) | AI-native browsing for flight search, form filling |
| macOS | [PyObjC](https://pyobjc.readthedocs.io/) + AppleScript | Deep system integration — apps, windows, Spotlight |
| MCP | [Model Context Protocol](https://modelcontextprotocol.io/) | Extensible tool integration, 5800+ available servers |
| Menu Bar | [rumps](https://github.com/jaredks/rumps) | Native macOS menu bar presence |
| Cloud LLM | [ChatGLM / Z.ai](https://z.ai) (optional) | Fallback for complex reasoning, PII-stripped |

**Monthly cost: $0** — everything runs locally. Cloud LLM is optional ($3–6/month if enabled).

---

## 🔧 Configuration

Roxy loads config from three sources (in priority order):

1. **Environment variables** with `ROXY_` prefix
2. **`.env` file** in the project root
3. **`config/default.yaml`** for defaults

Key settings:

```yaml
# Local LLM
llm:
  local:
    model: "qwen3:8b"           # Your Ollama model
  cloud:
    confidence_threshold: 0.7    # Below this → escalate to cloud

# Voice
voice:
  stt_model: "base.en"          # Whisper model size
  wake_word:
    sensitivity: 0.6            # 0.0-1.0, higher = more sensitive

# Privacy
privacy:
  cloud_consent: "ask"          # ask | always | never
  redact_patterns:
    - email
    - phone
    - ssn
    - credit_card
```

### API Keys (Optional)

Store securely in macOS Keychain:

```bash
# Web search (free: https://brave.com/search/api/)
uv run keyring set roxy brave_search_api_key "your-key"

# Cloud LLM fallback (optional)
uv run keyring set roxy zai_api_key "your-key"

# Wake word — Porcupine (free: https://console.picovoice.ai)
uv run keyring set roxy porcupine_access_key "your-key"
```

Or use a `.env` file if you prefer. Roxy works without any API keys — she just runs fully local.

---

## 🧩 Build Your Own Skill

Creating a new skill takes about 5 minutes. Here's a complete example:

```python
"""Pomodoro timer skill for Roxy."""
from __future__ import annotations

import asyncio
from roxy.skills.base import RoxySkill, SkillContext, SkillResult, Permission


class PomodoroSkill(RoxySkill):
    name = "pomodoro"
    description = "Start a focus timer with Do Not Disturb"
    triggers = [
        "start a focus session",
        "pomodoro",
        "focus mode",
        "start timer",
    ]
    permissions = [Permission.APPLESCRIPT, Permission.NOTIFICATIONS]

    async def execute(self, context: SkillContext) -> SkillResult:
        duration = context.parameters.get("minutes", 25)

        # Turn on Do Not Disturb
        from roxy.macos.applescript import AppleScriptRunner
        runner = AppleScriptRunner()
        await runner.run('do shell script "shortcuts run \\"Focus On\\""')

        # Schedule the end notification
        asyncio.get_event_loop().call_later(
            duration * 60,
            lambda: asyncio.ensure_future(self._timer_done(runner)),
        )

        return SkillResult(
            success=True,
            response_text=f"Focus mode on. I'll let you know in {duration} minutes.",
        )

    async def _timer_done(self, runner: AppleScriptRunner) -> None:
        await runner.send_notification("Pomodoro Complete", "Time for a break!")
        await runner.run('do shell script "shortcuts run \\"Focus Off\\""')

    def can_handle(self, intent: str, parameters: dict) -> float:
        focus_words = ["focus", "pomodoro", "timer", "concentrate"]
        return 0.9 if any(w in intent.lower() for w in focus_words) else 0.0
```

Save it to `src/roxy/skills/productivity/pomodoro.py`, restart Roxy, and say "Hey Roxy, start a focus session."

For the full guide: **[docs/skills-guide.md](docs/skills-guide.md)**

---

## 🖥️ Running Roxy

Three modes:

```bash
# Text mode — type commands in the terminal
uv run roxy

# Voice mode — wake word activated, fully hands-free
uv run roxy --voice

# Server mode — background service with menu bar icon
uv run roxy --server
```

Additional commands:

```bash
uv run roxy --health          # System health check
uv run roxy --stats           # Performance analytics
uv run roxy --verbose         # Debug logging
```

### Auto-Start on Login

```bash
# Create LaunchAgent (starts Roxy when you log in)
cp docs/com.roxy.assistant.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.roxy.assistant.plist
```

### Uninstall

```bash
bash scripts/uninstall.sh
```

Clean removal of everything — project files, data, models, LaunchAgent, Talon scripts. Offers a data backup before deleting. Asks before removing shared tools like Ollama.

---

## 🔒 Privacy Model

Roxy was built with a simple principle: **your data is yours.**

### What stays on your machine (always)

- All voice recordings and transcriptions
- Email and calendar content
- Notes and reminders
- File contents
- Conversation history and memories
- Passwords, API keys, tokens

### What can leave your machine (with your permission)

- Web search queries (via Brave Search — no tracking)
- Complex LLM requests (PII-stripped, consent-gated)

### What Roxy never does

- Send data to any analytics service
- Phone home to any server
- Store data anywhere except your local drive
- Share information between users (there's only one: you)

Audit everything: `cat ~/roxy/data/cloud_requests.log`

For the full privacy model: **[docs/privacy-model.md](docs/privacy-model.md)**

---

## 🧠 Self-Improvement (Roadmap)

Roxy is designed to get smarter over time:

- **Self-tuning** — adjusts confidence routing based on what works and what doesn't
- **Self-healing** — detects broken components and auto-recovers
- **Prompt optimisation** — rewrites her own system prompts based on real performance data
- **Skill generation** — when she can't do something, she tries to write a skill for it
- **Workflow composition** — detects repeated command sequences and suggests automations

The evolution system builds on a feedback loop that tracks every interaction and infers success from your responses — no thumbs-up buttons needed.

---

## 📋 Project Structure

```
~/roxy/
├── src/roxy/
│   ├── brain/              # Orchestrator, router, privacy gateway, LLM clients
│   ├── voice/              # Wake word, STT, TTS, voice pipeline
│   ├── memory/             # Session, history (SQLite), long-term (Mem0)
│   ├── skills/             # 16 built-in skills across 4 categories
│   │   ├── system/         #   App launcher, file search, window manager, ...
│   │   ├── web/            #   Web search, browser, flight search
│   │   ├── productivity/   #   Calendar, email, notes, reminders
│   │   └── dev/            #   Git ops, Claude Code integration
│   ├── macos/              # AppleScript, PyObjC, Spotlight, Hammerspoon, menu bar
│   └── mcp/                # MCP server manager + custom servers
├── tests/                  # Unit, integration, security, benchmark tests
├── config/                 # YAML configuration files
├── talon/                  # Talon Voice command scripts
├── docs/                   # Architecture, skills guide, privacy model
└── scripts/                # Install, uninstall, setup scripts
```

**17,750 lines** of async, type-hinted Python with Google-style docstrings.

---

## 🤝 Contributing

Roxy is open to contributions. The easiest way to start:

### Add a Skill

1. Read **[docs/skills-guide.md](docs/skills-guide.md)**
2. Create a new file in `src/roxy/skills/`
3. Extend `RoxySkill`, implement `execute()` and `can_handle()`
4. Add tests in `tests/unit/`
5. Submit a PR

### Skill Ideas We'd Love

- 🌤️ Weather (OpenWeatherMap or Open-Meteo)
- 🎵 Spotify / Apple Music control
- 🏠 Home Assistant / HomeKit integration
- 🔐 1Password / Bitwarden lookup
- 🌍 Translation (local model)
- 📊 CSV / spreadsheet analysis
- 📸 Screenshot + vision model ("what's on my screen?")
- 📰 RSS feed monitoring
- 💰 Expense tracking
- ⏱️ Pomodoro / focus timer

### Other Contributions

- **Bug reports** — file an issue with steps to reproduce
- **Documentation** — improvements to guides, examples, translations
- **MCP servers** — new integrations via Model Context Protocol
- **Voice models** — better wake word models, TTS voices
- **Performance** — profiling, optimisation, benchmarks

### Development Setup

```bash
git clone https://github.com/anthonyforan/roxy.git
cd roxy
uv sync --all-extras
uv run pre-commit install
uv run pytest tests/ -v
```

---

## 🗺️ Roadmap

### v0.1 — Foundation ✅
- [x] Local LLM via Ollama with confidence-based routing
- [x] Voice pipeline: wake word → STT → TTS
- [x] 16 built-in skills (system, web, productivity, dev)
- [x] Three-tier memory (session → history → long-term)
- [x] macOS deep integration (AppleScript, PyObjC, Spotlight, Shortcuts)
- [x] Privacy gateway with PII redaction
- [x] MCP server support
- [x] Menu bar application

### v0.2 — Evolution
- [ ] Self-improvement system (feedback loop, self-tuning, self-healing)
- [ ] Prompt self-optimisation
- [ ] Workflow composition (detect and automate repeated patterns)
- [ ] Document RAG (ingest your files, answer questions about them)

### v0.3 — Perception
- [ ] Vision (screenshot + local vision model)
- [ ] Proactive scheduling (Roxy suggests actions based on your patterns)
- [ ] MCP server auto-discovery
- [ ] Multi-language voice support

### Future
- [ ] Skill auto-generation (Roxy writes her own skills)
- [ ] Cross-device sync (encrypted, user-controlled)
- [ ] Linux port

---

## ⚡ Performance

Measured on M1 Pro, 16GB RAM:

| Component | Target | Measured |
|-----------|--------|----------|
| Wake word detection | <200ms | ~150ms |
| Intent classification | <500ms | ~300ms |
| Local LLM response | <2s | ~1.2s |
| Cloud LLM response | <5s | ~3.5s |
| Memory search | <100ms | ~50ms |
| TTS generation | <300ms | ~280ms |
| **End-to-end voice → voice** | **<4s** | **~3.2s** |

---

## 📝 How Roxy Was Built

Roxy was designed and built using **Claude Code Agent Teams** — an experimental feature that coordinates multiple AI agents working in parallel. The 17,750-line codebase was created across 5 build phases:

| Phase | Mode | What Was Built |
|-------|------|----------------|
| 1 | Single session | Project structure, config, Ollama verification |
| 2 | 2-agent team | Brain orchestrator + three-tier memory system |
| 3 | 3-agent team | Voice pipeline + macOS integration + web skills |
| 4 | Single session | Integration, testing, documentation, polish |
| 5 | 3-agent team | Self-improvement system (evolution module) |

The full architecture was designed in a single conversation, then executed by teams of specialised AI agents — each responsible for a distinct subsystem with defined interfaces and boundaries. The project demonstrates what's possible when AI agents are given clear specifications and allowed to work in parallel.

---

## 🙏 Acknowledgements

Roxy is built on the shoulders of incredible open source projects:

[Ollama](https://ollama.com) • [Agno](https://github.com/agno-agi/agno) • [Mem0](https://github.com/mem0ai/mem0) • [ChromaDB](https://github.com/chroma-core/chroma) • [faster-whisper](https://github.com/SYSTRAN/faster-whisper) • [MLX-Audio](https://github.com/lucasnewman/mlx-audio) • [Browser-Use](https://github.com/browser-use/browser-use) • [PyObjC](https://pyobjc.readthedocs.io/) • [Brave Search](https://brave.com/search/api/) • [Model Context Protocol](https://modelcontextprotocol.io/) • [rumps](https://github.com/jaredks/rumps) • [Hammerspoon](https://www.hammerspoon.org/)

---

## 📄 License

AGPL-3.0. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with 🎙️ by <a href="https://github.com/anthonyforan">Anthony Foran</a></strong><br>
  <em>From casino floor manager to AI developer — Roxy is proof that career transitions<br>
  don't have age limits and the best tools are the ones you build yourself.</em>
</p>
