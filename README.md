<div align="center">

  <!-- Title -->
  <h1>
    <img src="docs/images/roxy-logo.png" alt="Roxy Logo" width="120" height="120"><br>
    Roxy
  </h1>

  <!-- Tagline -->
  <p>
    <em>Your privacy-first, voice-controlled AI assistant for macOS</em>
  </p>

  <!-- Badges -->
  <p>
    <a href="https://pypi.org/project/roxy/">
      <img src="https://img.shields.io/pypi/v/roxy?style=flat-square" alt="PyPI Version">
    </a>
    <a href="https://python.org">
      <img src="https://img.shields.io/badge/python-3.12+-blue.svg?style=flat-square" alt="Python Version">
    </a>
    <a href="https://github.com/anthonyforan/roxy/actions">
      <img src="https://img.shields.io/github/actions/workflow/status/anthonyforan/roxy/ci.yml?style=flat-square" alt="CI Status">
    </a>
    <a href="https://coveralls.io/github/anthonyforan/roxy">
      <img src="https://img.shields.io/coveralls/github/anthonyforan/roxy?style=flat-square" alt="Coverage">
    </a>
    <a href="https://github.com/anthonyforan/roxy/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg?style=flat-square" alt="License: AGPL-3.0">
    </a>
    <a href="https://github.com/anthonyforan/roxy/stargazers">
      <img src="https://img.shields.io/github/stars/anthonyforan/roxy?style=flat-square" alt="Stars">
    </a>
  </p>

  <!-- Quick Links -->
  <p>
    <a href="#-features">Features</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-documentation">Documentation</a> •
    <a href="#-contributing">Contributing</a> •
    <a href="#-license">License</a>
  </p>
</div>

---

## 📖 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Screenshots](#-screenshots)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Development](#-development)
- [Contributing](#-contributing)
- [Documentation](#-documentation)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)
- [Roadmap](#-roadmap)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 🌟 About

**Roxy** is a privacy-first, voice-controlled AI assistant designed specifically for macOS. Unlike other AI assistants that send everything to the cloud, Roxy processes **80%+ of requests locally** on your Mac using state-of-the-art local LLMs.

### Why Roxy?

- 🔒 **Privacy-First**: Your conversations, files, and data stay on your Mac
- 🏠 **Local-First**: Fast responses without internet connectivity
- 🎙️ **Voice-Native**: Designed for voice interaction from the ground up
- 🔌 **Extensible**: Plugin-based skill system to add custom capabilities
- 🧠 **Smart**: Three-tier memory system that learns about you over time
- 🍎 **macOS Native**: Deep integration with macOS APIs and features

### Who is Roxy For?

- **Privacy-conscious users** who want AI assistance without data collection
- **Developers** who want a customizable, local AI assistant
- **macOS power users** looking for voice automation
- **Researchers** interested in local LLM applications

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🏠 Local-First Processing
- 80%+ of requests processed on-device
- Uses Ollama with Qwen3 8B for fast, private responses
- Works completely offline
- No API fees for local processing

</td>
<td width="50%">

### 🔊 Voice-Controlled
- Wake word detection with OpenWakeWord
- Speech-to-text with faster-whisper
- Text-to-speech with Kokoro via MLX-Audio
- Natural conversation flow

</td>
</tr>
<tr>
<td width="50%">

### 🎯 Skill-Based Architecture
- Modular plugin system
- 30+ built-in skills
- Easy to create custom skills
- Permission-based access control

</td>
<td width="50%">

### 🔒 Privacy-Gated Cloud
- PII detection and redaction
- User consent before cloud usage
- Audit logging of all cloud requests
- Configurable privacy modes

</td>
</tr>
<tr>
<td width="50%">

### 🧠 Smart Memory
- Session context tracking
- Semantic conversation history search
- Long-term fact storage with Mem0
- Learns your preferences

</td>
<td width="50%">

### 🖥️ macOS Integration
- Launch applications
- Search files with Spotlight
- Window management
- System information
- AppleScript execution

</td>
</tr>
</table>

---

## 🖼️ Screenshots

<div align="center">
  <img src="docs/images/roxy-repl.png" alt="Roxy REPL" width="800">
  <p><em>Text mode REPL with syntax highlighting</em></p>
</div>

<div align="center">
  <img src="docs/images/roxy-voice.png" alt="Roxy Voice Mode" width="800">
  <p><em>Voice mode with wake word detection</em></p>
</div>

---

## 📋 Prerequisites

### Required

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| macOS Version | 13.0 (Ventura) | 14.0 (Sonoma) or later |
| Hardware | Intel Mac | Apple Silicon M1+ |
| RAM | 8GB | 16GB+ |
| Python | 3.12 | 3.12+ |
| Disk Space | 20GB | 30GB+ |

### Optional

- **Docker**: For SearXNG local search
- **Talon Voice**: For advanced voice commands
- **Hammerspoon**: For window management
- **API Keys**: For cloud LLM fallback and web search

---

## 🚀 Installation

### Option 1: Automated Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/anthonyforan/roxy.git
cd roxy

# Run the installer
bash scripts/install.sh
```

The installer will:
✓ Check prerequisites (macOS, Python, uv)
✓ Install Ollama and required models
✓ Set up Python environment
✓ Configure environment variables
✓ Prompt for optional API keys
✓ Set up optional components (SearXNG, Talon)

### Option 2: Manual Installation

<details>
<summary>Click to expand manual installation steps</summary>

#### Step 1: Install Dependencies

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Ollama
brew install ollama

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.12+
brew install python@3.12
```

#### Step 2: Set Up Ollama Models

```bash
# Start Ollama
ollama serve &

# Pull required models
ollama pull qwen3:8b
ollama pull qwen3:0.6b
ollama pull nomic-embed-text
```

#### Step 3: Install Roxy

```bash
# Clone the repository
git clone https://github.com/anthonyforan/roxy.git
cd roxy

# Install Python dependencies
uv sync

# Install Playwright browsers
uv run playwright install chromium
```

#### Step 4: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Add API keys for cloud LLM (optional) and web search (optional)
```

</details>

### Option 3: Docker Installation

<details>
<summary>Click to expand Docker installation</summary>

```bash
# Build Docker image
docker build -t roxy .

# Run Roxy container
docker run -it --rm \
  -v ~/roxy/data:/app/data \
  --device /dev/snd \
  roxy
```
</details>

---

## ⚡ Quick Start

### Starting Roxy

```bash
# Text mode (interactive REPL)
uv run roxy

# Voice mode (wake word + speech I/O)
uv run roxy --voice

# Server mode (background service with menu bar)
uv run roxy --server

# Enable verbose logging
uv run roxy --verbose

# Override cloud consent mode
uv run roxy --cloud never
```

### First Steps

Once Roxy starts, try these commands:

```
# Voice commands (say these aloud)
"Hey Roxy, open Safari"
"Hey Roxy, what time is it?"
"Hey Roxy, search for Python tutorials"

# REPL commands (type these)
/help          # Show all commands
/stats         # Show performance statistics
/config        # Show configuration
/memory        # Search memories
/quit          # Exit
```

---

## 📖 Usage

### Voice Commands

Roxy responds to natural voice commands. Here are some examples:

| Command | Action | Skill |
|---------|--------|-------|
| "Hey Roxy, open Safari" | Launch an application | AppLauncherSkill |
| "Hey Roxy, search for AI news" | Search the web | WebSearchSkill |
| "Hey Roxy, find files about my project" | Search local files | FileSearchSkill |
| "Hey Roxy, what's running?" | Show system processes | SystemInfoSkill |
| "Hey Roxy, set up coding layout" | Arrange windows | WindowManagerSkill |
| "Hey Roxy, remember I like dark mode" | Store preference | MemorySkill |
| "Hey Roxy, what's my email address?" | Recall from memory | MemorySkill |
| "Hey Roxy, create a note" | Create a note | NotesSkill |
| "Hey Roxy, check my calendar" | Show calendar events | CalendarSkill |

### REPL Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/stats` | Display routing and usage statistics |
| `/config` | Show current configuration |
| `/memory [query]` | Search and display memories |
| `/quit` or `/exit` | Exit the REPL |

### Creating Custom Skills

```python
from roxy.skills.base import RoxySkill, SkillContext, SkillResult, Permission

class MyCustomSkill(RoxySkill):
    """A custom skill for my specific need."""

    name = "my_custom_skill"
    description = "Does something specific"
    triggers = ["do my thing", "custom action"]
    permissions = [Permission.NETWORK]

    async def execute(self, context: SkillContext) -> SkillResult:
        # Your skill logic here
        result = await self._do_work(context)

        return SkillResult(
            success=True,
            response_text=f"Completed: {result}",
        )
```

See [docs/skills-guide.md](docs/skills-guide.md) for detailed skill development guide.

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Providers
OLLAMA_HOST=http://localhost:11434

# Cloud LLM (optional - for complex queries)
ZAI_API_KEY=your-zhipu-api-key
ZAI_BASE_URL=https://api.z.ai/api/paas/v4

# Web Search (optional)
BRAVE_SEARCH_API_KEY=your-brave-key

# Privacy Settings
ROXY_CLOUD_CONSENT_MODE=ask    # ask | always | never
ROXY_PII_REDACTION=true

# Voice Settings
ROXY_WAKE_WORD=hey_roxy
ROXY_TTS_VOICE=af_heart
ROXY_TTS_SPEED=1.1

# General Settings
ROXY_LOG_LEVEL=INFO
ROXY_DATA_DIR=$HOME/roxy/data
```

### Configuration File

Edit `config/default.yaml` for advanced settings:

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
    confidence_threshold: 0.7    # Below this → cloud

memory:
  session_max_messages: 50
  history_db: "data/memory.db"
  chromadb_path: "data/chromadb"
  mem0_config:
    llm_provider: "ollama"
    llm_model: "qwen3:8b"

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

## 🛠️ Development

### Project Structure

```
roxy/
├── src/roxy/              # Source code
│   ├── brain/            # Orchestrator, LLM clients, routing
│   ├── skills/           # Skill system and implementations
│   │   ├── system/       # macOS skills
│   │   ├── web/          # Web search and browsing
│   │   ├── productivity/ # Calendar, email, notes
│   │   └── dev/          # Developer tools
│   ├── voice/            # STT, TTS, wake word
│   ├── memory/           # Three-tier memory system
│   ├── macos/            # macOS integration
│   └── mcp/              # MCP server management
├── config/               # Configuration files
├── scripts/              # Setup and utility scripts
├── tests/                # Unit and integration tests
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── docs/                 # Documentation
├── talon/                # Talon Voice scripts
└── data/                 # Local data (gitignored)
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/anthonyforan/roxy.git
cd roxy

# Install development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=roxy --cov-report=html
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/benchmarks/

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/integration/test_orchestrator_flow.py

# Run with coverage report
uv run pytest --cov=roxy --cov-report=html
open htmlcov/index.html
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Check linting
uv run ruff check src/ tests/

# Type checking
uv run mypy src/

# Run pre-commit hooks manually
uv run pre-commit run --all-files
```

### Building for Distribution

```bash
# Build Python package
uv build

# Create distribution
uv run pyinstaller roxy.spec

# Notarize for macOS (requires Apple Developer account)
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" dist/Roxy.app
```

---

## 🤝 Contributing

We love contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Quick Contribution Guide

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/roxy.git`
3. **Create** a branch: `git checkout -b feature/your-feature-name`
4. **Make** your changes
5. **Test** thoroughly: `uv run pytest`
6. **Commit** your changes: `git commit -m "Add your feature"`
7. **Push** to your branch: `git push origin feature/your-feature-name`
8. **Create** a Pull Request on GitHub

### Contribution Areas

We're looking for help with:

- 🎯 **New Skills**: Create skills for your favorite tools
- 🐛 **Bug Fixes**: Help squash bugs
- 📚 **Documentation**: Improve guides and docs
- 🧪 **Tests**: Increase test coverage
- 🌍 **Internationalization**: Add language support
- 🎨 **UI/UX**: Improve the interface

### Coding Standards

- Use `black` for formatting
- Follow PEP 8 style guide
- Add type hints to all functions
- Write docstrings for all public APIs
- Add tests for new features
- Keep commits atomic and well-described

---

## 📚 Documentation

- [Architecture](docs/architecture.md) - System design and architecture
- [Skills Guide](docs/skills-guide.md) - Creating custom skills
- [Privacy Model](docs/privacy-model.md) - Privacy and security
- [Performance Guide](docs/performance.md) - Performance optimization

---

## ⚡ Performance

### Benchmarks

| Operation | Target | Typical |
|-----------|--------|---------|
| Wake word detection | <200ms | ~150ms |
| Intent classification | <500ms | ~300ms |
| Local LLM response | <2s | ~1.2s |
| Cloud LLM response | <5s | ~3.5s |
| Memory search | <100ms | ~50ms |
| Skill execution | <500ms | ~150ms |
| End-to-end voice (local) | <4s | ~2.5s |

### Optimization Tips

1. **Use Apple Silicon** - M1/M2/M3 for best performance
2. **Increase RAM** - 16GB+ recommended for smooth operation
3. **Use SSD** - For faster memory and model loading
4. **Close unnecessary apps** - To free up resources
5. **Adjust model sizes** - Use smaller models for faster responses

---

## 🔧 Troubleshooting

### Common Issues

#### "Ollama not responding"

```bash
# Check if Ollama is running
pgrep ollama

# Start Ollama
ollama serve

# Check models are installed
ollama list
```

#### "Module not found" errors

```bash
# Reinstall dependencies
uv sync --reinstall

# Try installing in editable mode
uv pip install -e .
```

#### Voice mode not working

```bash
# Check microphone permissions
# System Settings > Privacy & Security > Microphone

# Test microphone input
uv run python -c "import pyaudio; p = pyaudio.PyAudio(); print('OK')"
```

#### Slow response times

- Check Ollama is using GPU: `ollama show qwen3:8b`
- Reduce model size: Use `qwen3:0.6b` for faster responses
- Close other applications to free up RAM
- Check CPU usage in Activity Monitor

### Getting Help

- 📖 Check the [documentation](docs/)
- 🔍 [Search existing issues](https://github.com/anthonyforan/roxy/issues)
- 💬 [Start a discussion](https://github.com/anthonyforan/roxy/discussions)
- 🐛 [Report a bug](https://github.com/anthonyforan/roxy/issues/new?template=bug_report.md)
- 💡 [Request a feature](https://github.com/anthonyforan/roxy/issues/new?template=feature_request.md)

---

## ❓ FAQ

<details>
<summary><b>Is Roxy really free and open source?</b></summary>

Yes! Roxy is licensed under the MIT License, which means it's completely free to use, modify, and distribute. The local LLMs run on your machine without any subscription or API fees.
</details>

<details>
<summary><b>Do I need an internet connection?</b></summary>

No! Roxy works completely offline for 80%+ of requests. An internet connection is only needed for:
- Initial model download
- Web search functionality
- Cloud LLM fallback for complex queries
- Optional features like flight search
</details>

<details>
<summary><b>What data does Roxy send to the cloud?</b></summary>

Roxy only sends data to the cloud when:
1. You explicitly enable cloud mode, AND
2. The request requires capabilities beyond local models, AND
3. You give consent (if in "ask" mode)

All cloud requests go through a privacy gateway that redacts PII (emails, phone numbers, addresses, etc.) before sending. You can review all cloud requests in `data/cloud_requests.log`.
</details>

<details>
<summary><b>Can I use Roxy on an Intel Mac?</b></summary>

Yes, but performance will be slower than on Apple Silicon. Intel Macs may take 2-3x longer for LLM responses. We recommend M1 or later for the best experience.
</details>

<details>
<summary><b>How much RAM do I need?</b></summary>

- **8GB**: Minimum, but may be slow
- **16GB**: Recommended for smooth operation
- **32GB+**: Optimal for larger models and multitasking
</details>

<details>
<summary><b>Can I change the wake word?</b></summary>

Yes! Edit the `ROXY_WAKE_WORD` environment variable in your `.env` file. You'll need to train a custom wake word model using OpenWakeWord.
</details>

<details>
<summary><b>How do I uninstall Roxy?</b></summary>

Run the uninstall script:
```bash
bash scripts/uninstall.sh
```

This will remove Roxy, its data, and optionally the Ollama models.
</details>

---

## 🗺️ Roadmap

### Version 0.2 (Current)
- [x] Basic orchestrator and skill system
- [x] Local LLM integration
- [x] Three-tier memory system
- [x] Privacy gateway
- [x] macOS integration skills
- [x] Web search and browsing
- [x] Voice pipeline
- [ ] Comprehensive testing
- [ ] Documentation

### Version 0.3 (Next)
- [ ] Productivity skills (calendar, email, notes)
- [ ] Developer skills (git, project management)
- [ ] Improved memory with better fact extraction
- [ ] Multi-language support
- [ ] Custom wake word training
- [ ] Performance optimization

### Version 0.4 (Future)
- [ ] Multi-modal support (images, documents)
- [ ] Custom model fine-tuning
- [ ] Skill marketplace
- [ ] Mobile app companion (iOS)
- [ ] Distributed memory across devices
- [ ] Advanced web automation

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** - see the [LICENSE](LICENSE) file for details.

### What AGPL-3.0 Means

This is a **copyleft license** that ensures:

- ✅ Freedom to use, modify, and distribute the software
- ✅ Source code must be made available when the software is distributed
- ✅ Modifications must be shared under the same license
- ✅ **Network use provision**: If you run this software as a network service (SaaS, web API, etc.), you must provide source code to users

### Why AGPL-3.0?

The AGPL-3.0 license was chosen to:
- Keep the software truly free and open source
- Prevent companies from using Roxy in proprietary products without sharing improvements
- Ensure that any SaaS/service offerings of Roxy contribute back to the community
- Protect users' freedom to study, modify, and improve the software

For more information about AGPL-3.0, visit: https://www.gnu.org/licenses/agpl-3.0.html

---

## 🙏 Acknowledgments

Roxy wouldn't be possible without these amazing open source projects:

### Core Technologies

- [**Agno**](https://github.com/emmeth/agno) - Agent framework for orchestration
- [**Ollama**](https://ollama.com) - Local LLM runtime
- [**Qwen**](https://qwenlm.github.io/) - Local LLM models
- [**Mem0**](https://mem0.ai) - Long-term memory system
- [**ChromaDB**](https://www.trychroma.com/) - Vector database

### Voice & Audio

- [**OpenWakeWord**](https://github.com/dscripka/openWakeWord) - Wake word detection
- [**faster-whisper**](https://github.com/SYSTRAN/faster-whisper) - Speech-to-text
- [**Kokoro**](https://github.com/remsky/Kokoro-FastAPI) - Text-to-speech
- [**MLX-Audio**](https://github.com/ml-explore/mlx-audio) - Audio processing on Apple Silicon

### Web & Automation

- [**Playwright**](https://playwright.dev) - Browser automation
- [**Brave Search**](https://search.brave.com) - Privacy-focused search API
- [**SearXNG**](https://searxng.org) - Local metasearch engine

### macOS Integration

- [**PyObjC**](https://pyobjc.readthedocs.io/) - Python to Objective-C bridge
- [**rumps**](https://github.com/jaredks/rumps) - macOS menu bar
- [**Hammerspoon**](https://www.hammerspoon.org/) - Window automation

### Development Tools

- [**uv**](https://github.com/astral-sh/uv) - Fast Python package manager
- [**pytest**](https://pytest.org/) - Testing framework
- [**Click**](https://click.palletsprojects.com/) - CLI framework
- [**Rich**](https://rich.readthedocs.io/) - Terminal formatting

---

## 💬 Contact & Community

### Get Help

- 📖 [Documentation](docs/)
- 🐛 [Report Issues](https://github.com/anthonyforan/roxy/issues)
- 💬 [Discussions](https://github.com/anthonyforan/roxy/discussions)
- 📧 Email: [anthony@foran.io](mailto:anthony@foran.io)

### Connect

- **GitHub**: [@anthonyforan](https://github.com/anthonyforan)
- **Location**: Malmö, Sweden
- **Website**: [https://roxy.ai](https://roxy.ai) (coming soon)

### Star History

<a href="https://github.com/anthonyforan/roxy">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=anthonyforan/roxy&type=timeline&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=anthonyforan/roxy&type=timeline" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=anthonyforan/roxy&type=timeline" />
  </picture>
</a>

---

<div align="center">

**Built with ❤️ on macOS**

*Your privacy-respecting AI companion*

[⬆ Back to Top](#-roxy-)

</div>
