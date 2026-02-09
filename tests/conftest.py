"""Shared pytest fixtures for Roxy tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from roxy.config import RoxyConfig, LocalLLMConfig, CloudLLMConfig, PrivacyConfig


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test data."""
    return tmp_path


@pytest.fixture
def mock_config(temp_dir: Path) -> RoxyConfig:
    """Return test configuration with temporary data directory."""
    config = RoxyConfig(
        name="TestRoxy",
        version="0.1.0-test",
        data_dir=str(temp_dir / "data"),
        log_level="DEBUG",
        llm_local=LocalLLMConfig(
            model="qwen3:8b",
            router_model="qwen3:0.6b",
            temperature=0.7,
            max_tokens=2048,
            host="http://localhost:11434",
        ),
        llm_cloud=CloudLLMConfig(
            provider="zai",
            model="glm-4.7",
            confidence_threshold=0.7,
            api_key="test-api-key",
            base_url="https://api.test.com",
        ),
        privacy=PrivacyConfig(
            redact_patterns=["email", "phone", "ssn"],
            cloud_consent="always",
            log_cloud_requests=True,
            pii_redaction_enabled=True,
        ),
    )

    # Ensure data directory exists
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)

    return config


@pytest.fixture
def mock_ollama_response() -> dict:
    """Return a mock Ollama API response."""
    return {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen3:8b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_cloud_response() -> dict:
    """Return a mock cloud API response."""
    return {
        "id": "cloud-test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "glm-4.7",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a cloud test response.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        },
    }


@pytest.fixture
def mock_httpx_client() -> MagicMock:
    """Return a mock httpx.AsyncClient."""
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.aclose = AsyncMock()
    return mock_client


@pytest.fixture
def mock_openai_client(mock_ollama_response: dict) -> MagicMock:
    """Return a mock OpenAI client."""
    mock_client = MagicMock()

    # Mock chat completions
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = mock_ollama_response["choices"][0]["message"]["content"]
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.total_tokens = mock_ollama_response["usage"]["total_tokens"]

    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    return mock_client


@pytest.fixture
def event_loop() -> asyncio.AbstractEventLoop:
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_ollama_client(mock_config: RoxyConfig) -> AsyncGenerator:
    """Yield a mock OllamaClient for testing."""
    from roxy.brain.llm_clients import OllamaClient

    # Create real client but with mocked HTTP
    client = OllamaClient(
        host=mock_config.llm_local.host,
        model=mock_config.llm_local.model,
    )

    # Mock the internal client
    with patch.object(client, "_client") as mock_openai:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 30

        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        yield client

    # Cleanup
    await client.close()


@pytest.fixture
def sample_user_input() -> str:
    """Return sample user input for testing."""
    return "What's the weather like today?"


@pytest.fixture
def sample_pii_text() -> str:
    """Return sample text containing PII."""
    return (
        "My email is john.doe@example.com and my phone number is 555-123-4567. "
        "My SSN is 123-45-6789 and I live at 123 Main St, Springfield, IL 62701."
    )


# Async test marker
pytest_asyncio = pytest.mark.asyncio
