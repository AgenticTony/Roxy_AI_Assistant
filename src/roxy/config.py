"""Roxy configuration management.

Loads configuration from .env files and YAML config files, merges them,
and provides validated settings via pydantic models.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class CloudProvider(str, Enum):
    """Supported cloud LLM providers."""
    ZAI = "zai"
    OPENROUTER = "openrouter"


class ConsentMode(str, Enum):
    """User consent modes for cloud LLM access."""
    ASK = "ask"
    ALWAYS = "always"
    NEVER = "never"


class LocalLLMConfig(BaseModel):
    """Configuration for local LLM (Ollama)."""

    model: str = "qwen3:8b"
    router_model: str = "qwen3:0.6b"
    temperature: float = 0.7
    max_tokens: int = 2048
    host: str = "http://localhost:11434"

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature is in valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Ensure max_tokens is positive."""
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class CloudLLMConfig(BaseModel):
    """Configuration for cloud LLM provider."""

    provider: CloudProvider = CloudProvider.ZAI
    model: str = "glm-4.7"
    confidence_threshold: float = 0.7
    api_key: str = ""
    base_url: str = ""

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        """Ensure confidence_threshold is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def set_default_base_url(self) -> "CloudLLMConfig":
        """Set default base URL based on provider."""
        if not self.base_url:
            if self.provider == CloudProvider.ZAI:
                self.base_url = "https://api.z.ai/api/paas/v4"
            elif self.provider == CloudProvider.OPENROUTER:
                self.base_url = "https://openrouter.ai/api/v1"
        return self


class PrivacyConfig(BaseModel):
    """Configuration for privacy and consent management."""

    redact_patterns: list[str] = Field(
        default_factory=lambda: ["email", "phone", "ssn", "credit_card", "address"]
    )
    cloud_consent: ConsentMode = ConsentMode.ASK
    log_cloud_requests: bool = True
    pii_redaction_enabled: bool = True


class MemoryConfig(BaseModel):
    """Configuration for memory systems."""

    session_max_messages: int = 50
    history_db: str = "data/memory.db"
    chromadb_path: str = "data/chromadb"

    mem0_llm_provider: str = "ollama"
    mem0_llm_model: str = "qwen3:8b"
    mem0_embedder_provider: str = "ollama"
    mem0_embedder_model: str = "nomic-embed-text"

    @field_validator("session_max_messages")
    @classmethod
    def validate_session_max_messages(cls, v: int) -> int:
        """Ensure session_max_messages is positive."""
        if v <= 0:
            raise ValueError("session_max_messages must be positive")
        return v


class WakeWordConfig(BaseModel):
    """Configuration for wake word detection."""

    backend: str = "auto"  # auto | porcupine | openwakeword | whisper
    wake_phrase: str = "hey roxy"
    sensitivity: float = 0.6
    porcupine_access_key: str = ""
    model_name: str = "hey_roxy"  # For OpenWakeWord backend

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Ensure backend is valid."""
        valid_backends = {"auto", "porcupine", "openwakeword", "whisper"}
        if v not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")
        return v

    @field_validator("sensitivity")
    @classmethod
    def validate_sensitivity(cls, v: float) -> float:
        """Ensure sensitivity is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("sensitivity must be between 0.0 and 1.0")
        return v


class VoiceConfig(BaseModel):
    """Configuration for voice systems."""

    stt_model: str = "base.en"
    tts_engine: str = "kokoro"
    wake_word_sensitivity: float = 0.6
    speak_responses: bool = True
    tts_voice: str = "af_heart"
    tts_speed: float = 1.1
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)

    @field_validator("wake_word_sensitivity")
    @classmethod
    def validate_sensitivity(cls, v: float) -> float:
        """Ensure sensitivity is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("wake_word_sensitivity must be between 0.0 and 1.0")
        return v

    @field_validator("tts_speed")
    @classmethod
    def validate_tts_speed(cls, v: float) -> float:
        """Ensure TTS speed is positive."""
        if v <= 0:
            raise ValueError("tts_speed must be positive")
        return v


class RoxyConfig(BaseSettings):
    """
    Roxy's main configuration.

    Loads from:
    1. .env file (via pydantic-settings)
    2. YAML config files (via load() classmethod)
    3. Environment variables with ROXY_ prefix

    Environment variables override YAML values.
    """

    model_config = SettingsConfigDict(
        env_prefix="ROXY_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Core settings (from .env)
    data_dir: str = "~/roxy/data"
    log_level: str = "INFO"
    name: str = "Roxy"
    version: str = "0.1.0"

    # LLM configurations (from YAML)
    llm_local: LocalLLMConfig = Field(default_factory=LocalLLMConfig)
    llm_cloud: CloudLLMConfig = Field(default_factory=CloudLLMConfig)

    # System configurations
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log_level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @model_validator(mode="after")
    def expand_data_dir(self) -> "RoxyConfig":
        """Expand user home directory in data_dir."""
        self.data_dir = str(Path(self.data_dir).expanduser())
        return self

    @classmethod
    def load(
        cls,
        yaml_path: Path | str | None = None,
        env_file: str | None = ".env",
    ) -> "RoxyConfig":
        """
        Load configuration from YAML and environment.

        Args:
            yaml_path: Path to YAML config file. If None, searches default locations.
            env_file: Path to .env file.

        Returns:
            Validated RoxyConfig instance.
        """
        # Find YAML config file
        yaml_file = cls._find_yaml_config(yaml_path)

        # Load YAML content
        yaml_data: dict[str, Any] = {}
        if yaml_file and yaml_file.exists():
            yaml_data = cls._load_yaml_file(yaml_file)
            logger.debug(f"Loaded config from {yaml_file}")

        # Handle the structure from default.yaml:
        # The YAML has top-level keys like 'roxy', 'llm', 'memory', etc.
        # We need to merge the 'roxy' section and flatten other sections

        # Start with roxy section if present
        if "roxy" in yaml_data:
            merged_data = dict(yaml_data["roxy"])
            # Remove roxy key and merge the rest
            yaml_data_without_roxy = {k: v for k, v in yaml_data.items() if k != "roxy"}
            merged_data.update(yaml_data_without_roxy)
            yaml_data = merged_data

        # Flatten nested YAML data for pydantic-settings
        flat_data = cls._flatten_dict(yaml_data)

        # Reconstruct nested structure for BaseModel fields
        nested_data = cls._reconstruct_nested(flat_data)

        # Set environment file if provided
        if env_file and Path(env_file).exists():
            os.environ.setdefault("ROXY_ENV_FILE", env_file)

        # Create config instance (pydantic-settings will load from .env)
        config = cls(**nested_data)

        # Ensure data directory exists
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)

        return config

    @classmethod
    def _find_yaml_config(cls, yaml_path: Path | str | None) -> Path | None:
        """Find the YAML config file to load."""
        if yaml_path:
            return Path(yaml_path)

        # Search in default locations
        default_locations = [
            Path("config/default.yaml"),
            Path("config/default.yml"),
            Path.home() / ".roxy" / "config.yaml",
            Path("/etc/roxy/config.yaml"),
        ]

        for location in default_locations:
            if location.exists():
                return location

        return None

    @classmethod
    def _load_yaml_file(cls, path: Path) -> dict[str, Any]:
        """Load YAML file and return parsed data."""
        try:
            with path.open("r") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse YAML file {path}: {e}")
            return {}

    @classmethod
    def _flatten_dict(cls, data: dict[str, Any], parent_key: str = "", sep: str = "_") -> dict[str, Any]:
        """
        Flatten nested dictionary for pydantic-settings.

        This function converts nested dicts to flat keys with underscores,
        which is what pydantic-settings expects when using env_nested_delimiter.

        Example:
            {"llm": {"local": {"model": "qwen3:8b"}}}
            -> {"llm_local_model": "qwen3:8b"}

        However, for nested BaseModel fields, we need to preserve the nested structure
        and only flatten leaf values.

        Args:
            data: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items: list[tuple[str, Any]] = []

        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                # Recursively flatten nested dicts
                items.extend(cls._flatten_dict(value, new_key, sep).items())
            else:
                items.append((new_key, value))

        return dict(items)

    @classmethod
    def _reconstruct_nested(cls, flat_data: dict[str, Any], sep: str = "_") -> dict[str, Any]:
        """
        Reconstruct nested structure from flattened keys.

        This is the inverse of _flatten_dict, converting flat keys back
        to nested dicts for pydantic BaseModel construction.

        For fields that are BaseModels with simple scalar fields, we don't
        need to reconstruct - we just pass the flat key-value pairs directly.

        Example:
            {"llm_local_model": "qwen3:8b"}
            -> {"llm_local": {"model": "qwen3:8b"}}
        """
        # Fields in RoxyConfig that are BaseModel types
        # We need to reconstruct nested dicts for these
        nested_model_fields = {
            "llm_local",  # LocalLLMConfig
            "llm_cloud",  # CloudLLMConfig
            "privacy",    # PrivacyConfig
            "memory",     # MemoryConfig
            "voice",      # VoiceConfig
        }

        # Fields that are simple sub-fields within nested models
        # For privacy config, these are direct fields, not nested
        privacy_fields = {
            "redact_patterns",
            "cloud_consent",
            "log_cloud_requests",
            "pii_redaction_enabled",
        }

        result: dict[str, Any] = {}

        for flat_key, value in flat_data.items():
            # Check if this is a privacy config field
            if flat_key.startswith("privacy_"):
                field_name = flat_key[len("privacy_"):]
                if "privacy" not in result:
                    result["privacy"] = {}
                result["privacy"][field_name] = value
                continue

            # Check if this is an llm_local field
            if flat_key.startswith("llm_local_"):
                field_name = flat_key[len("llm_local_"):]
                if "llm_local" not in result:
                    result["llm_local"] = {}
                result["llm_local"][field_name] = value
                continue

            # Check if this is an llm_cloud field
            if flat_key.startswith("llm_cloud_"):
                field_name = flat_key[len("llm_cloud_"):]
                if "llm_cloud" not in result:
                    result["llm_cloud"] = {}
                result["llm_cloud"][field_name] = value
                continue

            # Check if this is a memory field
            if flat_key.startswith("memory_"):
                field_name = flat_key[len("memory_"):]
                if "memory" not in result:
                    result["memory"] = {}
                # Handle mem0_config nested fields
                if field_name.startswith("mem0_"):
                    if "mem0_config" not in result["memory"]:
                        result["memory"]["mem0_config"] = {}
                    sub_field = field_name[len("mem0_"):]
                    result["memory"]["mem0_config"][sub_field] = value
                else:
                    result["memory"][field_name] = value
                continue

            # Check if this is a voice field
            if flat_key.startswith("voice_"):
                field_name = flat_key[len("voice_"):]
                if "voice" not in result:
                    result["voice"] = {}
                result["voice"][field_name] = value
                continue

            # Top-level field
            result[flat_key] = value

        return result

    def get_log_config(self) -> dict[str, Any]:
        """Get logging configuration dict for use with logging.config."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.log_level,
                    "formatter": "standard",
                },
            },
            "root": {
                "level": self.log_level,
                "handlers": ["console"],
            },
        }
