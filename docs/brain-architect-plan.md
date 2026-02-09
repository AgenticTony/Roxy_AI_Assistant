# Brain Architect Implementation Plan

**Author:** brain-architect agent
**Date:** 2026-02-08
**Status:** Planning Phase - Awaiting Approval

---

## Overview

This plan details the implementation of Roxy's core orchestration layer (the "brain"). The brain is responsible for:
1. Loading and managing configuration
2. Interfacing with local and cloud LLMs
3. Protecting user privacy via PII detection and redaction
4. Routing requests between local and cloud based on confidence
5. Orchestrating skill execution using the Agno framework

---

## File Structure

### Files to Create

```
src/roxy/
├── config.py                      # Configuration management (pydantic-settings)
│
├── brain/
│   ├── __init__.py               # Public API exports
│   ├── llm_clients.py            # OllamaClient, CloudLLMClient, LLMClient protocol
│   ├── privacy.py                # PrivacyGateway with PII redaction/restoration
│   ├── router.py                 # ConfidenceRouter for local/cloud routing
│   └── orchestrator.py           # RoxyOrchestrator main class
│
├── main.py                        # Entry point with text REPL
└── skills/
    └── base.py                    # RoxySkill base class (required by orchestrator)

tests/
├── conftest.py                    # Shared pytest fixtures
├── unit/
│   ├── test_config.py            # Config loading tests
│   ├── test_llm_clients.py       # LLM client tests
│   ├── test_privacy.py           # Privacy gateway tests
│   └── test_router.py            # Router tests
└── integration/
    └── test_orchestrator.py      # End-to-end orchestrator tests
```

---

## Key Interfaces and Dataclasses

### 1. Configuration (`config.py`)

```python
from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class LocalLLMConfig(BaseModel):
    """Local LLM (Ollama) configuration."""
    model: str = "qwen3:8b"
    router_model: str = "qwen3:0.6b"
    temperature: float = 0.7
    max_tokens: int = 2048
    host: str = "http://localhost:11434"

class CloudLLMConfig(BaseModel):
    """Cloud LLM configuration."""
    provider: Literal["zai", "openrouter"] = "zai"
    model: str = "glm-4.7"
    confidence_threshold: float = 0.7
    api_key: str = ""
    base_url: str = ""

class PrivacyConfig(BaseModel):
    """Privacy and consent configuration."""
    redact_patterns: list[str] = Field(default_factory=lambda: ["email", "phone", "ssn", "credit_card", "address"])
    cloud_consent: Literal["ask", "always", "never"] = "ask"
    log_cloud_requests: bool = True
    pii_redaction_enabled: bool = True

class RoxyConfig(BaseSettings):
    """Roxy's main configuration, loaded from .env and YAML."""
    model_config = SettingsConfigDict(
        env_prefix="ROXY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # From .env
    data_dir: Path = Field(default=Path.home() / "roxy" / "data")
    log_level: str = "INFO"

    # From YAML - merged in
    llm_local: LocalLLMConfig = Field(default_factory=LocalLLMConfig)
    llm_cloud: CloudLLMConfig
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)

    @classmethod
    def load(cls, yaml_path: Path | None = None) -> "RoxyConfig":
        """Load config from YAML file and merge with .env."""
        # Implementation loads YAML, merges with env vars
```

### 2. LLM Clients (`llm_clients.py`)

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Literal
from openai import AsyncOpenAI
import httpx

@dataclass
class LLMResponse:
    """Response from any LLM client."""
    content: str
    model: str
    provider: Literal["local", "cloud"]
    confidence: float | None = None
    tokens_used: int | None = None
    latency_ms: int | None = None

class LLMClient(Protocol):
    """Protocol for LLM clients."""

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        ...

class OllamaClient:
    """Client for local Ollama LLM."""
    def __init__(self, host: str, model: str) -> None:
        # Uses AsyncOpenAI with base_url=host

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Direct Ollama API call

class CloudLLMClient:
    """Client for cloud LLM (Z.ai/OpenRouter)."""
    def __init__(self, provider: str, api_key: str, model: str, base_url: str) -> None:
        # Uses AsyncOpenAI with provider-specific config

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Cloud API call with retry logic
```

### 3. Privacy Gateway (`privacy.py`)

```python
from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PIIMatch:
    """A detected PII instance."""
    pattern_name: str
    original: str
    start: int
    end: int
    placeholder: str

@dataclass
class RedactionResult:
    """Result of PII redaction."""
    redacted_text: str
    pii_matches: list[PIIMatch] = field(default_factory=list)
    was_redacted: bool = False

class PrivacyGateway:
    """
    Gateway that enforces Roxy's privacy model.

    Responsibilities:
    1. Detect PII in text using regex patterns
    2. Redact PII with placeholders
    3. Restore PII in responses
    4. Log all cloud requests
    5. Handle user consent for cloud access
    """

    def __init__(
        self,
        redact_patterns: list[str],
        consent_mode: Literal["ask", "always", "never"],
        log_path: Path,
    ) -> None:
        self.consent_mode = consent_mode
        self.log_path = log_path
        self._patterns = self._compile_patterns(redact_patterns)

    async def can_use_cloud(self) -> tuple[bool, str | None]:
        """Check if cloud LLM can be used based on consent mode."""
        # Returns (allowed, message)

    def redact(self, text: str) -> RedactionResult:
        """Detect and redact PII from text."""
        # Returns redacted text and list of replacements

    def restore(self, text: str, pii_matches: list[PIIMatch]) -> str:
        """Restore PII placeholders with original values."""
        # Inverse of redact()

    async def log_cloud_request(
        self,
        original_prompt: str,
        redacted_prompt: str,
        provider: str,
        response_summary: str,
    ) -> None:
        """Log cloud request to file for audit."""

    def _compile_patterns(self, pattern_names: list[str]) -> dict[str, re.Pattern]:
        """Compile regex patterns for PII detection."""
        # Patterns for: email, phone, ssn, credit_card, address
```

### 4. Confidence Router (`router.py`)

```python
from __future__ import annotations
import logging
from dataclasses import dataclass

from .llm_clients import LLMClient, LLMResponse, OllamaClient
from .privacy import PrivacyGateway, RedactionResult

logger = logging.getLogger(__name__)

class ConfidenceRouter:
    """
    Routes requests between local and cloud LLMs.

    Algorithm:
    1. Send request to local LLM with smaller model for confidence scoring
    2. If confidence >= threshold, use full local model for response
    3. If confidence < threshold:
       - Check privacy consent
       - Redact PII
       - Send to cloud LLM
       - Restore PII in response
       - Log the cloud request
    """

    def __init__(
        self,
        local_client: OllamaClient,
        cloud_client: LLMClient,
        privacy: PrivacyGateway,
        confidence_threshold: float = 0.7,
    ) -> None:
        self.local_client = local_client
        self.cloud_client = cloud_client
        self.privacy = privacy
        self.threshold = confidence_threshold

    async def route(self, request: str) -> LLMResponse:
        """Route request to appropriate LLM based on confidence."""
        # Implementation follows algorithm above

    async def _assess_confidence(self, request: str) -> float:
        """Assess local LLM's confidence on handling the request."""
        # Uses smaller router model (qwen3:0.6b)
        # Prompts model to rate confidence 0.0-1.0
```

### 5. Orchestrator (`orchestrator.py`)

```python
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any
import asyncio

from agno import Agent
from agno.tools import Toolkit

from ..config import RoxyConfig
from .router import ConfidenceRouter, LLMResponse

logger = logging.getLogger(__name__)

@dataclass
class SkillContext:
    """Context passed to every skill execution."""
    user_input: str
    intent: str
    parameters: dict[str, Any]
    memory: "MemoryManager"  # Protocol, will be real implementation later
    config: RoxyConfig
    conversation_history: list[dict] = field(default_factory=list)

@dataclass
class SkillResult:
    """Result returned by every skill execution."""
    success: bool
    response_text: str
    data: dict[str, Any] | None = None
    speak: bool = True
    follow_up: str | None = None

# Memory Manager Protocol (interface contract with memory-builder)
class MemoryManager(Protocol):
    """Protocol for the three-tier memory system."""

    async def get_session_context(self) -> list[dict]:
        """Get current conversation context."""

    async def search_history(self, query: str, limit: int = 5) -> list[dict]:
        """Search conversation history."""

    async def remember(self, key: str, value: str) -> None:
        """Store a long-term fact."""

    async def recall(self, query: str) -> list[str]:
        """Retrieve relevant long-term memories."""

    async def get_user_preferences(self) -> dict:
        """Get stored user preferences."""

# Stub implementation until memory-builder finishes
class StubMemoryManager:
    """In-memory stub for testing before real MemoryManager is available."""
    async def get_session_context(self) -> list[dict]:
        return []

    async def search_history(self, query: str, limit: int = 5) -> list[dict]:
        return []

    async def remember(self, key: str, value: str) -> None:
        pass

    async def recall(self, query: str) -> list[str]:
        return []

    async def get_user_preferences(self) -> dict:
        return {}

class RoxyOrchestrator:
    """
    Main orchestrator for Roxy's brain.

    Uses Agno framework to manage agent, tools, and conversation flow.
    """

    def __init__(self, config: RoxyConfig) -> None:
        self.config = config
        self.router = ConfidenceRouter(...)
        self.memory: MemoryManager = StubMemoryManager()  # Will be replaced
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """Create Agno agent with appropriate tools."""
        # Agno setup with tools registered from skills

    async def process(self, user_input: str) -> str:
        """
        Main processing pipeline:
        1. Get conversation context from memory
        2. Route to appropriate LLM
        3. Extract intent and parameters
        4. Dispatch to skill
        5. Store interaction in memory
        6. Return response
        """
```

### 6. Skill Base Class (`skills/base.py`)

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Any

class Permission(Enum):
    """Permissions that skills can declare."""
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    SHELL = "shell"
    MICROPHONE = "microphone"
    NOTIFICATIONS = "notifications"
    APPLESCRIPT = "applescript"
    CLOUD_LLM = "cloud_llm"

# Import from orchestrator to avoid duplication
from ..brain.orchestrator import SkillContext, SkillResult

class RoxySkill(ABC):
    """Base class for all Roxy skills."""

    name: str
    description: str
    triggers: list[str]
    permissions: list[Permission]
    requires_cloud: bool = False

    @abstractmethod
    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute the skill."""
        ...

    def can_handle(self, intent: str, parameters: dict) -> float:
        """
        Return confidence 0.0-1.0 that this skill handles the intent.
        Override for custom intent matching logic.
        """
        # Default: simple keyword matching on triggers
```

---

## Integration Points with memory-builder

### Contract

The brain-architect will define the `MemoryManager` protocol in `orchestrator.py`. The memory-builder teammate will implement this interface.

**Key requirements:**
1. MemoryManager MUST implement all async methods defined in the protocol
2. All methods MUST be async
3. MemoryManager will be injected into SkillContext for every skill call
4. Brain creates MemoryManager instance at startup
5. Brain passes MemoryManager to each skill via SkillContext

### Handoff Plan

1. **brain-architect completes:**
   - `StubMemoryManager` implementation
   - `MemoryManager` protocol definition
   - Orchestrator creates StubMemoryManager by default
   - Switch to real MemoryManager when ready (via config flag)

2. **memory-builder implements:**
   - Real `MemoryManager` class in `memory/manager.py`
   - All tier implementations (session, history, longterm)
   - Integration tests

3. **Integration:**
   - When memory-builder is done, brain-architect updates `RoxyOrchestrator.__init__()` to use real MemoryManager
   - Remove StubMemoryManager

---

## Privacy Model Compliance

### Enforcement Points

1. **PrivacyGateway.redact()** runs before ALL cloud requests
2. **PrivacyGateway.can_use_cloud()** enforces consent mode
3. **PrivacyGateway.log_cloud_request()** logs EVERY cloud call
4. **Skills NEVER call cloud APIs directly** - must go through router

### Cloud Request Flow

```
User Input
    ↓
ConfidenceRouter.route()
    ↓
[If confidence < threshold]
    ↓
PrivacyGateway.can_use_cloud()
    ↓
[If consent granted]
    ↓
PrivacyGateway.redact() → redacted text
    ↓
CloudLLMClient.generate(redacted_text)
    ↓
PrivacyGateway.restore(response)
    ↓
PrivacyGateway.log_cloud_request()
    ↓
Return response
```

### What Never Gets Sent to Cloud

1. File contents (checked in router before cloud escalation)
2. Passwords, API keys (matched by PII patterns)
3. Full conversation history (only last N messages, redacted)
4. Memory database contents (memory queries stay local)

---

## Implementation Order

### Phase 1: Configuration and LLM Clients (Day 1)

1. **config.py** - Configuration loading with pydantic-settings
2. **llm_clients.py** - LLMClient protocol, OllamaClient, CloudLLMClient
3. **skills/base.py** - RoxySkill base class and dataclasses
4. **tests/unit/test_config.py** - Config loading tests

### Phase 2: Privacy Gateway (Day 2)

1. **privacy.py** - PrivacyGateway with PII redaction
2. **tests/unit/test_privacy.py** - PII detection and redaction tests
3. **PII Patterns:**
   - Email: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`
   - Phone: `\b\d{3}[-.]?\d{3}[-.]?\d{4}\b`
   - SSN: `\b\d{3}-\d{2}-\d{4}\b`
   - Credit Card: `\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b`
   - Address: Complex pattern for street + city/state/zip

### Phase 3: Router and Orchestrator (Day 3-4)

1. **router.py** - ConfidenceRouter with local/cloud routing
2. **orchestrator.py** - RoxyOrchestrator with Agno agent
3. **tests/unit/test_router.py** - Router logic tests
4. **tests/integration/test_orchestrator.py** - End-to-end tests

### Phase 4: REPL and Integration (Day 5)

1. **main.py** - Text-mode REPL for testing
2. **tests/conftest.py** - Shared fixtures (mock clients, config)
3. Integration testing with real Ollama (if available)

---

## Testing Strategy

### Unit Tests

- **test_config.py:** Test YAML loading, .env merging, validation
- **test_privacy.py:** Test each PII pattern, redaction/restoration, consent modes
- **test_router.py:** Test confidence threshold logic, routing decisions
- **test_llm_clients.py:** Mock API calls, error handling, response parsing

### Integration Tests

- **test_orchestrator.py:** Full pipeline with mocked LLMs
- Test skill dispatch with mock skills
- Test memory integration with stub

### Test Fixtures (conftest.py)

```python
import pytest
from pathlib import Path

@pytest.fixture
def mock_config(tmp_path: Path) -> RoxyConfig:
    """Return test config with temporary data directory."""

@pytest.fixture
def mock_ollama_client():
    """Return mocked OllamaClient."""

@pytest.fixture
def mock_cloud_client():
    """Return mocked CloudLLMClient."""

@pytest.fixture
def sample_skill():
    """Return a simple test skill for orchestrator tests."""
```

---

## Dependencies to Add

None - all dependencies are already in pyproject.toml from Phase 1.

**Key dependencies already included:**
- `agno>=1.0.0` - Agent framework
- `openai>=1.50.0` - LLM client (used for both Ollama and cloud)
- `httpx>=0.27.0` - Async HTTP
- `pydantic>=2.7.0` - Settings validation
- `pydantic-settings>=2.3.0` - .env loading
- `pyyaml>=6.0` - YAML config

---

## Performance Considerations

1. **Confidence assessment** uses smaller model (qwen3:0.6b) for speed
2. **PII detection** is regex-based (fast) vs LLM-based (slow)
3. **Async throughout** - no blocking calls in the hot path
4. **Connection pooling** via httpx for cloud API calls
5. **Caching** - consider caching frequently-used PII patterns

---

## Open Questions

1. **Agno API:** Exact Agno framework API for tool/skill registration needs verification
2. **Confidence scoring:** Best approach to extract confidence score from LLM response
3. **Memory storage:** Should orchestrator cache conversation context between turns?

---

## Success Criteria

- [ ] All unit tests pass with >80% coverage on brain/ module
- [ ] Integration test demonstrates full pipeline: input → router → LLM → response
- [ ] Privacy gateway correctly detects and redacts all PII patterns
- [ ] Router correctly makes local vs cloud decisions based on confidence
- [ ] REPL allows interactive testing of the brain
- [ ] Code follows all coding standards from CLAUDE.md
- [ ] All type hints complete, all public functions have docstrings
