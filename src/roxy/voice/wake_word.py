"""Wake word detection with multi-backend support.

Supports Porcupine (default), OpenWakeWord (Python <=3.11), and Whisper fallback.
Auto-selects the best available backend at runtime.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger(__name__)


class WakeWordBackend(ABC):
    """Abstract backend for wake word detection."""

    @abstractmethod
    async def start(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Start listening for wake word.

        Args:
            callback: Async function to call when wake word is detected.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Stop listening for wake word."""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the backend is currently running."""


class PorcupineBackend(WakeWordBackend):
    """Wake word detection using Picovoice Porcupine (default).

    Features:
    - Python 3.11+ compatible
    - Runs natively on macOS Apple Silicon
    - Free tier with custom wake words
    - Low CPU usage
    """

    def __init__(self, access_key: str | None = None, sensitivity: float = 0.6):
        """Initialize Porcupine backend.

        Args:
            access_key: Picovoice API key (get free from console.picovoice.ai).
            sensitivity: Detection threshold 0.0-1.0 (default 0.6).
        """
        try:
            import pvporcupine
            self._pvporcupine = pvporcupine
        except ImportError as e:
            raise RuntimeError(
                "pvporcupine not installed. Install with: uv add pvporcupine"
            ) from e

        self._access_key = access_key
        self._sensitivity = sensitivity
        self._running = False
        self._handle = None
        self._stream = None

        logger.debug(f"PorcupineBackend initialized (sensitivity={sensitivity})")

    async def start(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Start listening for wake word using Porcupine.

        Uses built-in "porcupine" keyword (always available) or custom keyword
        from Picovoice console for "hey roxy".
        """
        import sounddevice as sd
        import numpy as np

        # Create Porcupine instance
        # Use "porcupine" built-in keyword as default (always available)
        # For custom "hey roxy", get a .ppn file from console.picovoice.ai
        keyword_paths = None
        if self._access_key:
            # Try to load custom keyword if access key is provided
            # In production, you would use a custom .ppn file for "hey roxy"
            keyword_paths = [self._pvporcupine.KEYWORD_PATHS[self._pvporcupine.Keyword.PORCUPINE]]

        self._handle = self._pvporcupine.create(
            access_key=self._access_key,
            keywords=[self._pvporcupine.Keyword.PORCUPINE],
            sensitivities=[self._sensitivity],
        )

        self._running = True
        frame_length = self._handle.frame_length
        sample_rate = self._handle.sample_rate

        def audio_callback(indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
            """Audio stream callback - processes audio and detects wake word."""
            if not self._running:
                return

            # Convert to int16 for Porcupine
            pcm = (indata[:, 0] * 32767).astype(np.int16)

            # Process audio in chunks matching Porcupine's frame length
            for i in range(0, len(pcm) - frame_length + 1, frame_length):
                result = self._handle.process(pcm[i : i + frame_length])
                if result >= 0:
                    # Wake word detected!
                    logger.info("Porcupine: Wake word detected!")
                    asyncio.get_event_loop().call_soon_threadsafe(
                        lambda: asyncio.ensure_future(callback())
                    )
                    # Add brief pause to prevent multiple triggers
                    return

        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_length * 4,
            callback=audio_callback,
        )
        self._stream.start()
        logger.info("Porcupine wake word detection started")

    async def stop(self) -> None:
        """Stop Porcupine wake word detection."""
        self._running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._handle is not None:
            self._handle.delete()
            self._handle = None

        logger.info("Porcupine wake word detection stopped")

    @property
    def is_running(self) -> bool:
        return self._running


class OpenWakeWordBackend(WakeWordBackend):
    """Wake word detection using OpenWakeWord (requires Python <=3.11).

    Features:
    - Free and open source
    - Custom model training
    - Good accuracy
    - Higher CPU usage than Porcupine
    """

    def __init__(self, model_name: str = "hey_roxy", sensitivity: float = 0.6):
        """Initialize OpenWakeWord backend.

        Args:
            model_name: Name of the wake word model (default "hey_roxy").
            sensitivity: Detection threshold 0.0-1.0 (default 0.6).
        """
        try:
            import openwakeword
            self._oww = openwakeword
        except ImportError as e:
            raise RuntimeError(
                "openwakeword not installed. Requires Python <=3.11. "
                "Install with: uv add openwakeword --python 3.11"
            ) from e

        self._model_name = model_name
        self._sensitivity = sensitivity
        self._running = False
        self._model = None
        self._stream = None

        logger.debug(f"OpenWakeWordBackend initialized (model={model_name}, sensitivity={sensitivity})")

    async def start(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Start listening for wake word using OpenWakeWord."""
        import sounddevice as sd
        import numpy as np
        from pathlib import Path

        # Load model
        try:
            from openwakeword.model import Model

            model_path = Path.home() / ".openwakeword" / "models"
            model_path.mkdir(parents=True, exist_ok=True)

            self._model = Model(
                wakeword_models=[self._model_name],
                enable_speex_noise_suppression=True,
            )
            logger.info(f"OpenWakeWord model loaded for '{self._model_name}'")

        except Exception as e:
            logger.error(f"Failed to load OpenWakeWord model: {e}")
            raise

        # Audio parameters for OpenWakeWord
        SAMPLE_RATE = 16000
        CHUNK_SIZE = 1280

        self._running = True

        def audio_callback(indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
            """Audio callback for OpenWakeWord detection."""
            if not self._running:
                return

            audio_float = indata[:, 0]  # Single channel

            # Detect wake word
            predictions = self._model.predict(audio_float)

            if self._model_name in predictions:
                score = predictions[self._model_name]
                threshold = 1.0 - self._sensitivity
                if score >= threshold:
                    logger.info(f"OpenWakeWord: '{self._model_name}' detected! (score: {score:.3f})")
                    asyncio.get_event_loop().call_soon_threadsafe(
                        lambda: asyncio.ensure_future(callback())
                    )
                    return

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        )
        self._stream.start()
        logger.info("OpenWakeWord detection started")

    async def stop(self) -> None:
        """Stop OpenWakeWord detection."""
        self._running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._model = None
        logger.info("OpenWakeWord detection stopped")

    @property
    def is_running(self) -> bool:
        return self._running


class WhisperWakeWordBackend(WakeWordBackend):
    """Fallback: Use energy detection + Whisper to check for wake phrase.

    Features:
    - No extra dependencies (uses faster-whisper already in stack)
    - Works on any Python version
    - Higher CPU usage
    - Slightly slower detection
    """

    def __init__(self, wake_phrase: str = "hey roxy", sensitivity: float = 0.6):
        """Initialize Whisper fallback backend.

        Args:
            wake_phrase: Phrase to listen for (default "hey roxy").
            sensitivity: Detection threshold 0.0-1.0 (default 0.6).
        """
        self._wake_phrase = wake_phrase.lower()
        self._sensitivity = sensitivity
        self._running = False
        self._stream = None
        self._whisper = None

        logger.debug(f"WhisperWakeWordBackend initialized (phrase='{wake_phrase}', sensitivity={sensitivity})")

    async def start(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Start energy-based wake word detection with Whisper verification."""
        import sounddevice as sd
        import numpy as np

        self._running = True
        sample_rate = 16000
        chunk_duration = 2.0  # seconds
        energy_threshold = 0.01 + (1.0 - self._sensitivity) * 0.05

        def audio_callback(indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
            """Audio callback - detects speech energy and transcribes with Whisper."""
            if not self._running:
                return

            # Calculate RMS energy
            energy = np.sqrt(np.mean(indata ** 2))

            if energy > energy_threshold:
                # Someone is speaking - check if it's the wake phrase
                asyncio.get_event_loop().call_soon_threadsafe(
                    lambda: asyncio.ensure_future(
                        self._check_wake_phrase(indata.copy(), callback)
                    )
                )

        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(sample_rate * chunk_duration),
            callback=audio_callback,
        )
        self._stream.start()
        logger.info("Whisper-based wake word detection started (fallback mode)")

    async def _check_wake_phrase(self, audio_data: np.ndarray, callback: Callable[[], Awaitable[None]]) -> None:
        """Transcribe audio chunk and check for wake phrase.

        Args:
            audio_data: Audio chunk that triggered energy threshold.
            callback: Callback to invoke if wake phrase is found.
        """
        try:
            from faster_whisper import WhisperModel
            import numpy as np

            # Lazy-load Whisper model
            if self._whisper is None:
                self._whisper = WhisperModel("tiny.en", device="auto")
                logger.info("Whisper tiny model loaded for wake word detection")

            # Convert to format expected by faster-whisper
            audio_flat = audio_data.flatten().astype(np.float32)

            # Transcribe
            segments, _ = self._whisper.transcribe(audio_flat, language="en")
            text = " ".join(s.text for s in segments).lower().strip()

            logger.debug(f"Heard: '{text}'")

            if self._wake_phrase in text:
                logger.info(f"Whisper: Wake phrase '{self._wake_phrase}' detected!")
                await callback()

        except Exception as e:
            logger.debug(f"Wake phrase check failed: {e}")

    async def stop(self) -> None:
        """Stop Whisper-based detection."""
        self._running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        logger.info("Whisper-based wake word detection stopped")

    @property
    def is_running(self) -> bool:
        return self._running


class WakeWordDetector:
    """Main wake word detector - auto-selects best available backend.

    Tries backends in priority order:
    1. Porcupine (best quality, 3.11+ compatible, free tier)
    2. OpenWakeWord (free option, needs Python <=3.11)
    3. Whisper fallback (always works, higher CPU)

    Automatically falls back if a backend fails to initialize.
    """

    # Backend class references for instantiation
    _BACKENDS = [
        ("Porcupine", PorcupineBackend),
        ("OpenWakeWord", OpenWakeWordBackend),
        ("Whisper", WhisperWakeWordBackend),
    ]

    def __init__(self, config: dict | None = None):
        """Initialize wake word detector with auto backend selection.

        Args:
            config: Optional configuration dict with keys:
                - backend: "auto" | "porcupine" | "openwakeword" | "whisper"
                - wake_phrase: Wake phrase (default "hey roxy")
                - sensitivity: Detection threshold 0.0-1.0 (default 0.6)
                - porcupine_access_key: Picovoice API key (optional)
                - model_name: Model name for OpenWakeWord (default "hey_roxy")
        """
        self._config = config or {}
        self._backend: WakeWordBackend | None = None
        self._backend_name: str = "None"
        self._select_backend()

    def _select_backend(self) -> None:
        """Try backends in order of preference.

        Respects config["backend"] if specified, otherwise auto-selects.
        """
        backend_preference = self._config.get("backend", "auto")
        sensitivity = self._config.get("sensitivity", 0.6)
        wake_phrase = self._config.get("wake_phrase", "hey roxy")
        porcupine_key = self._config.get("porcupine_access_key")

        # Build list of backends to try
        backends_to_try = []

        if backend_preference == "auto":
            # Try all backends in order
            for name, backend_cls in self._BACKENDS:
                backends_to_try.append((name, backend_cls))
        else:
            # Use specified backend only
            backend_map = {name.lower(): (name, cls) for name, cls in self._BACKENDS}
            if backend_preference.lower() in backend_map:
                backends_to_try.append(backend_map[backend_preference.lower()])
            else:
                logger.warning(f"Unknown backend '{backend_preference}', trying auto-selection")
                for name, backend_cls in self._BACKENDS:
                    backends_to_try.append((name, backend_cls))

        # Try each backend until one works
        for backend_name, backend_cls in backends_to_try:
            try:
                if backend_cls == PorcupineBackend:
                    # Porcupine requires access key for custom keywords,
                    # but works with built-in keywords without one
                    self._backend = PorcupineBackend(
                        access_key=porcupine_key,
                        sensitivity=sensitivity,
                    )
                elif backend_cls == OpenWakeWordBackend:
                    self._backend = OpenWakeWordBackend(
                        model_name=self._config.get("model_name", "hey_roxy"),
                        sensitivity=sensitivity,
                    )
                elif backend_cls == WhisperWakeWordBackend:
                    self._backend = WhisperWakeWordBackend(
                        wake_phrase=wake_phrase,
                        sensitivity=sensitivity,
                    )

                self._backend_name = backend_name
                logger.info(f"Wake word backend: {backend_name}")
                return

            except (ImportError, RuntimeError) as e:
                logger.debug(f"Backend {backend_name} not available: {e}")
                continue

        # If we get here, all backends failed
        logger.error("All wake word backends failed to initialize!")
        self._backend = None
        self._backend_name = "None"

    @property
    def backend_name(self) -> str:
        """Get the name of the active backend."""
        return self._backend_name

    async def start_listening(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Start listening for the wake word.

        Args:
            callback: Async function to call when wake word is detected.

        Raises:
            RuntimeError: If no backend is available or already listening.
        """
        if self._backend is None:
            raise RuntimeError("No wake word backend available. Check configuration and dependencies.")

        if self._backend.is_running:
            raise RuntimeError("Wake word detector is already listening")

        logger.info(f"Starting wake word detection with {self._backend_name} backend...")
        await self._backend.start(callback)

    async def stop_listening(self) -> None:
        """Stop listening for the wake word."""
        if self._backend is not None:
            await self._backend.stop()

    @property
    def is_active(self) -> bool:
        """Check if the detector is currently listening.

        Returns:
            True if listening, False otherwise.
        """
        return self._backend is not None and self._backend.is_running


async def detect_wake_word(
    config: dict | None = None,
) -> WakeWordDetector:
    """
    Create a wake word detector.

    Convenience function that creates a detector.
    The detector will be started by VoicePipeline.

    Args:
        config: Optional configuration dict for the detector.

    Returns:
        WakeWordDetector instance ready to start listening.
    """
    detector = WakeWordDetector(config)
    return detector
