"""Text-to-speech using Kokoro TTS via MLX-Audio.

Generates speech from text with support for streaming output
and macOS 'say' command fallback.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Audio output constants
SAMPLE_RATE = 24000  # Kokoro sample rate
CHANNELS = 1
DTYPE = np.float32

# Kokoro voice IDs (from MLX-Audio)
DEFAULT_VOICE = "af_heart"
AVAILABLE_VOICES = [
    "af_heart",  # American female
    "af_bella",  # American female
    "af_nicole",  # American female
    "af_sarah",  # American female
    "af_sky",  # American female
    "am_michael",  # American male
    "am_adam",  # American male
    "am_echo",  # American male
    "am_eric",  # American male
    "am_liam",  # American male
    "bf_emma",  # British female
    "bf_george",  # British male
    "bm_lewis",  # British male
]


class TextToSpeech:
    """
    Text-to-speech conversion using Kokoro TTS.

    Supports both MLX-Audio (Apple Silicon optimized) and macOS
    'say' command as fallback.
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.1,
        use_fallback: bool = False,
        output_device: str | None = None,
    ) -> None:
        """Initialize the text-to-speech engine.

        Args:
            voice: Voice ID to use (default: "af_heart").
            speed: Speech speed multiplier (1.0 = normal, >1.0 = faster).
            use_fallback: Force use of macOS 'say' command instead of Kokoro.
            output_device: Optional audio output device name or index.
        """
        self.voice = voice
        self.speed = speed
        self.use_fallback = use_fallback
        self.output_device = output_device

        # State management
        self._is_playing: bool = False
        self._stop_requested: bool = False

        # Thread pool for blocking operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts_")

        # Audio stream for playback
        self._stream: sd.OutputStream | None = None

        # Check if Kokoro is available
        self._kokoro_available: bool = False
        if not use_fallback:
            self._check_kokoro_available()

        logger.debug(
            f"TextToSpeech initialized with voice='{voice}', speed={speed}, "
            f"use_fallback={use_fallback}, kokoro_available={self._kokoro_available}"
        )

    def _check_kokoro_available(self) -> None:
        """Check if MLX-Audio Kokoro is available."""
        try:
            import mlx_audio

            self._kokoro_available = True
            logger.info("MLX-Audio Kokoro TTS is available")
        except ImportError:
            logger.warning("MLX-Audio not available, will use fallback 'say' command")
            self._kokoro_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize MLX-Audio: {e}, using fallback")
            self._kokoro_available = False

    def _get_output_device(self) -> int | None:
        """Get the audio output device index.

        Returns:
            Device index or None for default device.
        """
        if self.output_device is None:
            return None

        try:
            if isinstance(self.output_device, str):
                devices = sd.query_devices()
                for idx, device in enumerate(devices):
                    if device["name"] == self.output_device and device["max_output_channels"] > 0:
                        logger.debug(f"Using audio device: {device['name']} (index {idx})")
                        return idx
                logger.warning(f"Could not find device '{self.output_device}', using default")
                return None
            return int(self.output_device)
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}. Using default device.")
            return None

    async def speak(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> None:
        """Speak the given text.

        Args:
            text: Text to speak.
            voice: Override voice for this utterance.
            speed: Override speed for this utterance.
        """
        if not text or not text.strip():
            logger.debug("Empty text, skipping TTS")
            return

        # Use overrides or defaults
        voice_to_use = voice or self.voice
        speed_to_use = speed or self.speed

        logger.info(f"Speaking: {text[:100]}{'...' if len(text) > 100 else ''}")

        self._is_playing = True
        self._stop_requested = False

        try:
            if self._kokoro_available and not self.use_fallback:
                await self._speak_kokoro(text, voice_to_use, speed_to_use)
            else:
                await self._speak_fallback(text, voice_to_use, speed_to_use)
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
        finally:
            self._is_playing = False
            self._stop_requested = False

    async def _speak_kokoro(self, text: str, voice: str, speed: float) -> None:
        """Speak using Kokoro TTS.

        Args:
            text: Text to speak.
            voice: Voice ID to use.
            speed: Speech speed multiplier.
        """
        try:
            from mlx_audio.tts import TTS

            # Generate audio in thread pool
            loop = asyncio.get_event_loop()

            def _generate_audio() -> tuple[np.ndarray, int]:
                """Generate audio using Kokoro (blocking)."""
                tts = TTS(voice=voice)
                audio_array = tts.generate(
                    text=text,
                    speed=speed,
                )
                # MLX-Audio returns float32 arrays at 24kHz
                return audio_array, SAMPLE_RATE

            # Generate audio
            audio, sample_rate = await loop.run_in_executor(
                self._executor,
                _generate_audio,
            )

            # Check if stopped during generation
            if self._stop_requested:
                return

            # Play audio
            await self._play_audio(audio, sample_rate)

        except ImportError:
            logger.warning("MLX-Audio import failed, falling back to 'say' command")
            await self._speak_fallback(text, voice, speed)
        except Exception as e:
            logger.error(f"Kokoro TTS failed: {e}, falling back to 'say' command")
            await self._speak_fallback(text, voice, speed)

    async def _speak_fallback(self, text: str, voice: str, speed: float) -> None:
        """Speak using macOS 'say' command fallback.

        Args:
            text: Text to speak.
            voice: Voice ID (mapped to macOS voice).
            speed: Speech speed (mapped to words per minute).
        """
        try:
            # Map Kokoro voice to macOS voice (simplified mapping)
            macos_voice = self._map_voice_to_macos(voice)

            # Convert speed multiplier to words per minute
            # Normal is ~170 wpm, speed 1.0 = 170, 2.0 = 340
            wpm = int(170 * speed)

            # Run 'say' command
            loop = asyncio.get_event_loop()

            def _run_say() -> None:
                """Run the say command (blocking)."""
                subprocess.run(
                    ["say", "-v", macos_voice, "-r", str(wpm), text],
                    check=True,
                    capture_output=True,
                )

            await loop.run_in_executor(self._executor, _run_say)

        except Exception as e:
            logger.error(f"Fallback TTS failed: {e}", exc_info=True)

    def _map_voice_to_macos(self, voice: str) -> str:
        """Map Kokoro voice ID to macOS voice name.

        Args:
            voice: Kokoro voice ID.

        Returns:
            macOS voice name.
        """
        # Simplified mapping - could be more sophisticated
        voice_map = {
            "af_heart": "Samantha",
            "af_bella": "Samantha",
            "af_nicole": "Samantha",
            "af_sarah": "Samantha",
            "af_sky": "Samantha",
            "am_michael": "Alex",
            "am_adam": "Alex",
            "am_echo": "Alex",
            "am_eric": "Alex",
            "am_liam": "Alex",
            "bf_emma": "Moira",
            "bf_george": "Daniel",
            "bm_lewis": "Daniel",
        }

        return voice_map.get(voice, "Samantha")

    async def _play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play audio array through sounddevice.

        Args:
            audio: Audio data as numpy array.
            sample_rate: Sample rate in Hz.
        """
        device = self._get_output_device()

        # Create audio stream
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            device=device,
        )

        try:
            self._stream.start()

            # Play audio in chunks to allow interruption
            chunk_size = 1024
            for i in range(0, len(audio), chunk_size):
                if self._stop_requested:
                    logger.debug("Playback stopped by request")
                    break

                chunk = audio[i : i + chunk_size]
                self._stream.write(chunk)

                # Small sleep to allow event loop processing
                await asyncio.sleep(0.001)

        finally:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def speak_streaming(
        self,
        text_stream: AsyncGenerator[str, None],
    ) -> None:
        """Speak streaming text as it arrives.

        Queues text chunks and speaks them in order.

        Args:
            text_stream: Async generator yielding text chunks.
        """
        buffer: list[str] = []

        async for chunk in text_stream:
            buffer.append(chunk)

            # Speak complete sentences when possible
            combined = "".join(buffer)

            # Simple sentence boundary detection
            if any(punct in combined for punct in [".", "!", "?", "\n"]):
                # Speak what we have
                await self.speak(combined)
                buffer = []

        # Speak remaining text
        if buffer:
            await self.speak("".join(buffer))

    async def stop(self) -> None:
        """Stop current speech playback."""
        if self._is_playing and not self._stop_requested:
            logger.debug("Stopping speech playback")
            self._stop_requested = True

            # Close stream if open
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.debug(f"Error closing stream: {e}")
                self._stream = None

            # Wait for playback to stop
            while self._is_playing:
                await asyncio.sleep(0.01)

    @property
    def is_playing(self) -> bool:
        """Check if currently playing speech.

        Returns:
            True if playing, False otherwise.
        """
        return self._is_playing

    @property
    def is_available(self) -> bool:
        """Check if TTS is available.

        Returns:
            True if Kokoro or fallback is available.
        """
        return self._kokoro_available or not self.use_fallback

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.stop()
        self._executor.shutdown(wait=True)
        logger.debug("TextToSpeech cleanup complete")


async def create_tts(
    voice: str = DEFAULT_VOICE,
    speed: float = 1.1,
    use_fallback: bool = False,
    output_device: str | None = None,
) -> TextToSpeech:
    """
    Create a text-to-speech engine.

    Convenience function that creates a TTS instance.

    Args:
        voice: Voice ID to use.
        speed: Speech speed multiplier.
        use_fallback: Force use of fallback.
        output_device: Optional audio output device.

    Returns:
        TextToSpeech instance.
    """
    tts = TextToSpeech(
        voice=voice,
        speed=speed,
        use_fallback=use_fallback,
        output_device=output_device,
    )

    return tts
