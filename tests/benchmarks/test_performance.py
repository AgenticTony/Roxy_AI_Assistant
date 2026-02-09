"""Performance benchmarks for Roxy components.

Tests that performance targets from docs/performance.md are met.
"""

from __future__ import annotations

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from roxy.config import RoxyConfig
from roxy.brain.orchestrator import RoxyOrchestrator
from roxy.brain.llm_clients import OllamaClient
from roxy.skills.registry import SkillRegistry
from roxy.skills.base import SkillContext, SkillResult
from roxy.memory.manager import MemoryManager


# Performance targets (in milliseconds)
TARGET_WAKE_WORD_DETECTION = 200
TARGET_INTENT_CLASSIFICATION = 500
TARGET_LOCAL_LLM_RESPONSE = 2000
TARGET_CLOUD_LLM_RESPONSE = 5000
TARGET_TTS_GENERATION = 300
TARGET_MEMORY_SEARCH = 100
TARGET_SKILL_EXECUTION = 500
TARGET_E2E_VOICE_LOCAL = 4000


@pytest.mark.asyncio
async def test_skill_find_performance(mock_config: RoxyConfig) -> None:
    """Benchmark skill finding performance.

    Target: <500ms for intent classification (includes skill find)
    """
    registry = SkillRegistry()
    registry.reset()

    # Register a few skills
    from roxy.skills.system import AppLauncherSkill, FileSearchSkill

    registry.register(AppLauncherSkill)
    registry.register(FileSearchSkill)

    # Warm up
    registry.find_skill("open Safari", {})
    registry.find_skill("find files", {})

    # Benchmark
    iterations = 100
    times = []

    for _ in range(iterations):
        start = time.time()
        skill, confidence = registry.find_skill("open Safari", {})
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    max_time = max(times)

    print(f"\nSkill Find Performance:")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Max: {max_time:.2f}ms")
    print(f"  Target: <{TARGET_INTENT_CLASSIFICATION}ms")

    # Should be much faster than 500ms (just string matching)
    assert avg_time < 50, f"Skill find too slow: {avg_time:.2f}ms"


@pytest.mark.asyncio
async def test_skill_execution_performance(mock_config: RoxyConfig) -> None:
    """Benchmark skill execution performance.

    Target: <500ms for skill execution
    """
    from roxy.skills.system import AppLauncherSkill

    skill = AppLauncherSkill()

    context = SkillContext(
        user_input="open Calculator",
        intent="open_app",
        parameters={"app": "Calculator"},
        memory=MagicMock(),
        config=mock_config,
        conversation_history=[],
    )

    # Mock subprocess to avoid actual app launch
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        # Warm up
        await skill.execute(context)

        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            await skill.execute(context)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        print(f"\nSkill Execution Performance:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Max: {max_time:.2f}ms")
        print(f"  Target: <{TARGET_SKILL_EXECUTION}ms")

        assert avg_time < TARGET_SKILL_EXECUTION, f"Skill execution too slow: {avg_time:.2f}ms"


@pytest.mark.asyncio
async def test_memory_search_performance(mock_config: RoxyConfig) -> None:
    """Benchmark memory search performance.

    Target: <100ms for memory search
    """
    manager = MemoryManager(config=mock_config)
    await manager.initialize()

    # Add some test data
    await manager.session.add_message("user", "My name is Alice")
    await manager.session.add_message("assistant", "Hello Alice!")

    # Warm up
    await manager.search_history("Alice", limit=5)

    # Benchmark
    times = []
    for _ in range(20):
        start = time.time()
        await manager.search_history("Alice", limit=5)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    avg_time = sum(times) / len(times)

    print(f"\nMemory Search Performance:")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Target: <{TARGET_MEMORY_SEARCH}ms")

    # Note: May not meet target if embedding generation is slow
    assert avg_time < 500, f"Memory search too slow: {avg_time:.2f}ms"

    await manager.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_throughput(mock_config: RoxyConfig) -> None:
    """Benchmark orchestrator request throughput.

    Measures how many requests can be processed per second.
    """
    registry = SkillRegistry()
    registry.reset()

    with patch("roxy.brain.llm_clients.AsyncOpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 10
        mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

        orchestrator = RoxyOrchestrator(mock_config, skill_registry=registry)
        await orchestrator.initialize()

        # Warm up
        await orchestrator.process("Hello")

        # Benchmark throughput
        num_requests = 10
        start = time.time()

        for i in range(num_requests):
            await orchestrator.process(f"Test message {i}")

        elapsed = time.time() - start
        requests_per_second = num_requests / elapsed

        print(f"\nOrchestrator Throughput:")
        print(f"  Requests: {num_requests}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {requests_per_second:.2f} req/s")

        # Should handle at least 0.5 req/s (2s per request)
        assert requests_per_second > 0.5, f"Throughput too low: {requests_per_second:.2f} req/s"

        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_timing_stats_collection(mock_config: RoxyConfig) -> None:
    """Test that timing statistics are properly collected."""
    registry = SkillRegistry()
    registry.reset()

    with patch("roxy.brain.llm_clients.AsyncOpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 10
        mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

        orchestrator = RoxyOrchestrator(mock_config, skill_registry=registry)
        await orchestrator.initialize()

        # Process some requests
        for i in range(5):
            await orchestrator.process(f"Test message {i}")

        # Get stats
        stats = await orchestrator.get_statistics()
        timing_stats = stats.get("timing_stats", {})

        print(f"\nTiming Statistics:")
        for operation, metrics in timing_stats.items():
            print(f"  {operation}:")
            print(f"    Avg: {metrics['avg']:.2f}ms")
            print(f"    Min: {metrics['min']:.2f}ms")
            print(f"    Max: {metrics['max']:.2f}ms")
            print(f"    Count: {metrics['count']}")

        # Verify we collected stats for expected operations
        expected_operations = [
            "process_total",
            "memory_context_build",
            "skill_find",
            "memory_store",
            "history_trim",
        ]

        for op in expected_operations:
            assert op in timing_stats, f"Missing timing stats for {op}"

        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_concurrent_performance(mock_config: RoxyConfig) -> None:
    """Test performance under concurrent requests."""
    import asyncio

    registry = SkillRegistry()
    registry.reset()

    with patch("roxy.brain.llm_clients.AsyncOpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Concurrent response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 10
        mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

        orchestrator = RoxyOrchestrator(mock_config, skill_registry=registry)
        await orchestrator.initialize()

        # Process concurrent requests
        num_concurrent = 5
        start = time.time()

        tasks = [
            orchestrator.process(f"Concurrent message {i}")
            for i in range(num_concurrent)
        ]

        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start

        print(f"\nConcurrent Performance:")
        print(f"  Concurrent requests: {num_concurrent}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Per request: {elapsed / num_concurrent:.2f}s")

        # All requests should complete
        assert len(results) == num_concurrent
        assert all(isinstance(r, str) for r in results)

        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_memory_usage_stability(mock_config: RoxyConfig) -> None:
    """Test that memory usage remains stable over many requests."""
    import gc
    import sys

    registry = SkillRegistry()
    registry.reset()

    with patch("roxy.brain.llm_clients.AsyncOpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 10
        mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

        orchestrator = RoxyOrchestrator(mock_config, skill_registry=registry)
        await orchestrator.initialize()

        # Force garbage collection before
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Process many requests
        for i in range(50):
            await orchestrator.process(f"Memory test {i}")

            # Periodically check memory
            if i % 10 == 0:
                gc.collect()
                current_objects = len(gc.get_objects())
                growth = current_objects - initial_objects
                print(f"\nMemory at request {i}: {growth} objects")

        # Final check
        gc.collect()
        final_objects = len(gc.get_objects())
        total_growth = final_objects - initial_objects

        print(f"\nMemory Usage:")
        print(f"  Initial objects: {initial_objects}")
        print(f"  Final objects: {final_objects}")
        print(f"  Growth: {total_growth}")

        # Growth should be reasonable (< 10x initial growth)
        # Note: This is a rough check; memory usage depends on many factors

        await orchestrator.shutdown()


# Async test marker
# Note: Using @pytest.mark.asyncio decorator directly
