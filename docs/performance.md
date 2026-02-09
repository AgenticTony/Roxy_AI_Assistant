# Roxy Performance Guide

## Overview

This document covers Roxy's performance characteristics, profiling tools, and optimization strategies. Roxy is designed for responsive, local-first operation with targeted performance metrics for each component.

## Performance Targets

| Operation                    | Target | Measurement Point          |
|------------------------------|--------|-----------------------------|
| Wake word detection          | <200ms | Audio → Detection           |
| Intent classification        | <500ms | Input → Skill selected      |
| Local LLM response           | <2s    | Input → Response text       |
| Cloud LLM response           | <5s    | Input → Response text       |
| TTS generation               | <300ms | Text → Audio                |
| Memory search                | <100ms | Query → Results             |
| Skill execution              | <500ms | Skill dispatch → Result     |
| End-to-end voice (local)     | <4s    | Wake → Speak                |

## Architecture Performance

### Local vs Cloud Routing

Roxy uses a confidence-based routing system:

```
Input → [Intent Classification: <500ms]
         ↓
       [Confidence Score: <100ms]
         ↓
    High confidence (≥0.7)?
         ↓ Yes      ↓ No
    [Local LLM: <2s]  [Privacy Gateway: <50ms]
                            ↓
                      [Cloud LLM: <5s]
```

**Local routing:**
- Typical response: 1-2 seconds
- No external dependencies
- Works offline

**Cloud routing:**
- Typical response: 3-5 seconds (includes privacy gateway)
- Requires internet connection
- More capable for complex queries

## Built-in Profiling

### Timing Statistics

Roxy automatically tracks timing for key operations:

```python
# Access timing stats via REPL
/stats

# Example output:
Timing Statistics:
┌─────────────────────┬────────┬────────┬────────┬────────┐
│ Operation           │ Avg    │ Min    │ Max    │ Count  │
├─────────────────────┼────────┼────────┼────────┼────────┤
│ process_total       │ 1250ms │ 800ms  │ 3000ms │ 45     │
│ skill_execution     │ 150ms  │ 50ms   │ 800ms  │ 30     │
│ llm_local_generate  │ 900ms  │ 600ms  │ 2000ms │ 12     │
│ llm_cloud_generate  │ 3500ms │ 2500ms │ 6000ms │ 3      │
└─────────────────────┴────────┴────────┴────────┴────────┘
```

### Code Examples

**Access timing stats programmatically:**

```python
from roxy.brain.orchestrator import RoxyOrchestrator

orchestrator = RoxyOrchestrator(config)
await orchestrator.initialize()

# Process some requests
await orchestrator.process("Hello")

# Get timing statistics
stats = await orchestrator.get_statistics()
timing_stats = stats["timing_stats"]

for operation, metrics in timing_stats.items():
    print(f"{operation}: avg {metrics['avg']:.0f}ms")
```

**Track custom timing in skills:**

```python
import time
from roxy.skills.base import RoxySkill, SkillContext, SkillResult

class MySkill(RoxySkill):
    async def execute(self, context: SkillContext) -> SkillResult:
        start = time.time()

        # Your skill logic here
        result = await self._do_work()

        elapsed_ms = (time.time() - start) * 1000

        # Log timing (this will be picked up by logging)
        logger.info(f"{self.name} executed in {elapsed_ms:.0f}ms")

        return result
```

## Performance Profiling Tools

### Using Python Profiler

Profile Roxy's performance using Python's built-in profiler:

```bash
# Profile a single Roxy session
uv run python -m cProfile -o roxy.prof -m roxy

# Analyze the profile
uv run python -c "
import pstats
p = pstats.Stats('roxy.prof')
p.sort_stats('cumulative')
p.print_stats(20)
"
```

### Memory Profiling

Profile memory usage:

```bash
# Install memory profiler
uv add memory-profiler

# Run with memory profiling
uv run python -m memory_profiler -m roxy
```

### Flame Graphs

Generate flame graphs for visual performance analysis:

```bash
# Install flamegraph
pip install flamegraph

# Generate flame graph
flamegraph -o roxy_flamegraph.svg uv run python -m roxy
```

## Component Performance

### Ollama (Local LLM)

**Model sizes and performance:**

| Model          | Size (Q4) | RAM   | Speed    |
|----------------|-----------|-------|----------|
| Qwen3 0.6B     | ~400MB    | 1GB   | <100ms   |
| Qwen3 8B       | ~5GB      | 8GB   | <2s      |
| Qwen3 14B      | ~9GB      | 16GB  | <4s      |

**Optimization tips:**

1. **Use appropriate model size:**
   - 0.6B for intent classification (fast)
   - 8B for general queries (balanced)
   - 14B+ for complex reasoning (slower)

2. **Enable GPU acceleration (if available):**
   ```bash
   # Set Ollama to use GPU (Metal on Apple Silicon)
   export OLLAMA_GPU=nvidia  # or 'metal' for Apple Silicon
   ```

3. **Cache models:**
   - Models are cached after first load
   - Keep Ollama running between sessions

### Voice Pipeline

**Wake Word Detection:**

```python
# Performance monitoring
from roxy.voice.wake_word import WakeWordDetector

detector = WakeWordDetector(sensitivity=0.6)

# Benchmark detection
import time
start = time.time()
detected = detector.detect(audio_data)
elapsed = (time.time() - start) * 1000
print(f"Detection in {elapsed:.0f}ms")
```

**Speech-to-Text (STT):**

```python
# Benchmark different model sizes
models = ["tiny.en", "base.en", "small.en", "medium.en"]

for model in models:
    stt = STTEngine(model_name=model)
    start = time.time()
    text = await stt.transcribe(audio_data)
    elapsed = (time.time() - start) * 1000
    print(f"{model}: {elapsed:.0f}ms - {text}")
```

**Text-to-Speech (TTS):**

```python
# Benchmark TTS generation
from roxy.voice.tts import TTSEngine

tts = TTSEngine(voice_id="af_heart", speed=1.1)

start = time.time()
audio = await tts.synthesize("Hello, this is a test.")
elapsed = (time.time() - start) * 1000
print(f"TTS generated in {elapsed:.0f}ms")
```

### Memory System

**Benchmark memory operations:**

```python
# Session memory (fastest)
start = time.time()
await memory.session.add_message("user", "Test")
context = await memory.get_session_context()
print(f"Session: {(time.time() - start) * 1000:.0f}ms")

# History search (moderate)
start = time.time()
results = await memory.search_history("test query", limit=5)
print(f"History: {(time.time() - start) * 1000:.0f}ms")

# Long-term memory (slower but cached)
start = time.time()
memories = await memory.recall("user preference")
print(f"Long-term: {(time.time() - start) * 1000:.0f}ms")
```

### Skills

**Profile skill execution:**

```python
import time
from functools import wraps

def timed(skill_method):
    @wraps(skill_method)
    async def wrapper(self, context):
        start = time.time()
        try:
            result = await skill_method(self, context)
            return result
        finally:
            elapsed = (time.time() - start) * 1000
            logger.info(f"{self.name} executed in {elapsed:.0f}ms")
    return wrapper

class TimedSkill(RoxySkill):
    @timed
    async def execute(self, context):
        # Your skill logic
        pass
```

## Optimization Strategies

### Reduce Cloud Usage

1. **Increase confidence threshold:**
   ```yaml
   llm:
     cloud:
       confidence_threshold: 0.8  # Default 0.7
   ```

2. **Optimize skill triggers:**
   - Use specific, unique trigger phrases
   - Avoid generic triggers that match everything

3. **Cache common responses:**
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=100)
   async def get_cached_response(query: str) -> str:
       # Check if we have a cached response
       pass
   ```

### Improve Local LLM Speed

1. **Use quantized models:**
   - Q4_K_M is the sweet spot for 8B models
   - Smaller quantization = faster but less accurate

2. **Adjust token limits:**
   ```yaml
   llm:
     local:
       max_tokens: 1024  # Default 2048, half is faster
   ```

3. **Lower temperature:**
   ```yaml
   llm:
     local:
       temperature: 0.5  # Lower = faster sampling
   ```

### Optimize Memory System

1. **Limit history search:**
   ```python
   # Reduce search results
   results = await memory.search_history(query, limit=3)  # Default 5
   ```

2. **Use session memory when possible:**
   - Session memory is fastest (in-memory)
   - Use for temporary conversation context

3. **Batch memory operations:**
   ```python
   # Batch store facts
   await asyncio.gather(
       memory.remember("fact1", "value1"),
       memory.remember("fact2", "value2"),
       memory.remember("fact3", "value3"),
   )
   ```

### Voice Optimization

1. **Use smaller STT model:**
   ```python
   stt = STTEngine(model_name="tiny.en")  # Fast but less accurate
   ```

2. **Adjust TTS speed:**
   ```python
   tts = TTSEngine(voice_id="af_heart", speed=1.3)  # Faster speech
   ```

3. **Increase wake word sensitivity:**
   ```python
   detector = WakeWordDetector(sensitivity=0.7)  # More sensitive = faster detection
   ```

## Performance Monitoring

### Logging

Enable performance logging:

```bash
# Enable debug logging
export ROXY_LOG_LEVEL=DEBUG

# Run Roxy
uv run roxy
```

**Log format:**
```
2024-01-15 10:23:45 [DEBUG] roxy.brain.orchestrator: Processed in 1250ms
2024-01-15 10:23:45 [DEBUG] roxy.brain.llm_clients: Local LLM response: model=qwen3:8b, tokens=45, latency=900ms
```

### Metrics Export

Export metrics for external monitoring:

```python
# Export to Prometheus format (future)
from roxy.monitoring import PrometheusExporter

exporter = PrometheusExporter(port=9090)
exporter.start()
```

### Performance Dashboard

View real-time performance:

```bash
# Start monitoring dashboard (future)
uv run roxy --monitor
```

## Benchmarking

### Run Benchmarks

```bash
# Run performance benchmarks
uv run pytest tests/benchmarks/ -v

# With detailed output
uv run pytest tests/benchmarks/ -v --tb=short
```

### Custom Benchmarks

Create custom benchmarks:

```python
import pytest
import time
import asyncio

@pytest.mark.asyncio
async def benchmark_skill_execution():
    """Benchmark skill execution time."""
    from roxy.skills.system import AppLauncherSkill
    from roxy.skills.base import SkillContext

    skill = AppLauncherSkill()
    context = SkillContext(
        user_input="open Safari",
        intent="open_app",
        parameters={"app": "Safari"},
        memory=MagicMock(),
        config=MagicMock(),
        conversation_history=[],
    )

    # Warm up
    await skill.execute(context)

    # Benchmark
    times = []
    for _ in range(10):
        start = time.time()
        await skill.execute(context)
        times.append((time.time() - start) * 1000)

    avg = sum(times) / len(times)
    print(f"Average: {avg:.0f}ms")
    assert avg < 500  # Should be under 500ms
```

## Performance Checklist

Use this checklist to verify performance:

- [ ] Local LLM responses under 2s
- [ ] Cloud LLM responses under 5s
- [ ] Wake word detection under 200ms
- [ ] Intent classification under 500ms
- [ ] Memory search under 100ms
- [ ] Skill execution under 500ms
- [ ] TTS generation under 300ms
- [ ] End-to-end voice (local) under 4s
- [ ] CPU usage under 80% during normal operation
- [ ] RAM usage appropriate for model size
- [ ] No memory leaks during extended sessions
- [ ] Logs show timing information

## Troubleshooting

### Slow Local LLM

**Symptoms:** Local responses take >5s

**Solutions:**
1. Check Ollama is running: `pgrep ollama`
2. Verify model is loaded: `ollama list`
3. Check RAM availability: `vm_stat`
4. Try smaller model: `ollama pull qwen3:0.6b`

### Slow Wake Word Detection

**Symptoms:** Wake word takes >500ms

**Solutions:**
1. Check audio device is working
2. Adjust sensitivity: `WAKE_WORD_SENSITIVITY=0.7`
3. Verify OpenWakeWord model is loaded

### High Memory Usage

**Symptoms:** RAM usage grows over time

**Solutions:**
1. Check for memory leaks: profile with `memory_profiler`
2. Clear conversation history: `/memory clear`
3. Restart Roxy periodically
4. Reduce session max messages in config

### Stuttering Audio

**Symptoms:** TTS audio stutters or delays

**Solutions:**
1. Reduce TTS speed: `ROXY_TTS_SPEED=1.0`
2. Check audio device
3. Use simpler voice model
4. Pre-generate common responses

## Performance Tuning Guide

### For Low-End Macs (M1, 8GB RAM)

```yaml
llm:
  local:
    model: "qwen3:0.6b"  # Use smaller model
    max_tokens: 512      # Reduce output

memory:
  session_max_messages: 20  # Reduce context

voice:
  stt_model: "tiny.en"     # Smallest STT model
  tts_speed: 1.2           # Faster speech
```

### For High-End Macs (M2/M3, 32GB+ RAM)

```yaml
llm:
  local:
    model: "qwen3:14b"     # Larger model
    max_tokens: 4096       # Longer responses

memory:
  session_max_messages: 100  # More context

voice:
  stt_model: "medium.en"   # Better accuracy
  tts_speed: 1.0           # Normal speed
```

## Further Reading

- [Architecture Documentation](architecture.md)
- [Skills Guide](skills-guide.md)
- [Privacy Model](privacy-model.md)
- [Ollama Performance](https://ollama.com/blog/performance)
