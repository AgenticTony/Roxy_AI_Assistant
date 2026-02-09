"""Voice pipeline orchestrator.

Manages the complete voice interaction flow:
wake word → speech-to-text → orchestrator → text-to-speech.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from roxy.config import VoiceConfig

from .stt import SpeechToText
from .tts import TextToSpeech
from .wake_word import WakeWordDetector

if TYPE_CHECKING:
    from roxy.brain.orchestrator import RoxyOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class PipelineStatus:
    """Status information for the voice pipeline."""

    is_running: bool = False
    is_listening: bool = False
    is_speaking: bool = False
    current_interaction: str | None = None
    total_interactions: int = 0
    last_wake_time: float | None = None
    error: str | None = None


class VoicePipeline:
    """
    Orchestrates the complete voice interaction pipeline.

    Manages wake word detection, speech-to-text, orchestrator interaction,
    and text-to-speech in a seamless loop.
    """

    def __init__(
        self,
        orchestrator: "RoxyOrchestrator",
        config: VoiceConfig,
    ) -> None:
        """Initialize the voice pipeline.

        Args:
            orchestrator: Roxy orchestrator for processing user input.
            config: Voice configuration settings.
        """
        self.orchestrator = orchestrator
        self.config = config

        # Initialize components
        # Build wake word config dict
        wake_word_config = {
            "backend": config.wake_word.backend,
            "wake_phrase": config.wake_word.wake_phrase,
            "sensitivity": config.wake_word.sensitivity,
            "porcupine_access_key": config.wake_word.porcupine_access_key or None,
            "model_name": config.wake_word.model_name,
        }
        self._wake_word = WakeWordDetector(config=wake_word_config)

        self._stt = SpeechToText(
            model_size=config.stt_model,
            device="cpu",
        )

        self._tts = TextToSpeech(
            voice=config.tts_voice,
            speed=config.tts_speed,
        )

        # State management
        self._status = PipelineStatus()
        self._running: bool = False
        self._processing_lock = asyncio.Lock()

        logger.info("VoicePipeline initialized")

    async def start(self) -> None:
        """Start the voice pipeline.

        Begins listening for the wake word and processing interactions.
        """
        if self._running:
            logger.warning("Voice pipeline is already running")
            return

        logger.info("Starting voice pipeline...")
        self._running = True
        self._status.is_running = True

        # Start wake word detection
        await self._wake_word.start_listening(callback=self._on_wake_word_detected)

        logger.info("Voice pipeline started, listening for wake word")

    async def stop(self) -> None:
        """Stop the voice pipeline."""
        if not self._running:
            return

        logger.info("Stopping voice pipeline...")

        self._running = False
        self._status.is_running = False

        # Stop wake word detection
        await self._wake_word.stop_listening()

        # Stop any current speech
        await self._tts.stop()

        # Cleanup components
        await self._stt.cleanup()
        await self._tts.cleanup()

        logger.info("Voice pipeline stopped")

    async def _on_wake_word_detected(self) -> None:
        """Handle wake word detection.

        Called when the wake word is detected. Starts the interaction
        processing flow.
        """
        if not self._running:
            return

        logger.info("Wake word detected, starting interaction")

        # Acquire lock to prevent overlapping interactions
        if self._processing_lock.locked():
            logger.info("Already processing interaction, ignoring wake word")
            return

        async with self._processing_lock:
            await self._process_interaction()

    async def _process_interaction(self) -> None:
        """Process a complete voice interaction.

        Flow: Listen → Transcribe → Process → Speak
        """
        import time

        self._status.last_wake_time = time.time()
        self._status.is_listening = True

        try:
            # 1. Listen for user input
            logger.debug("Listening for user input...")
            user_text = await self._stt.listen_until_silence(timeout=10.0)

            if not user_text or not user_text.strip():
                logger.info("No speech detected, returning to listening")
                self._status.is_listening = False
                return

            logger.info(f"User said: {user_text}")
            self._status.current_interaction = user_text
            self._status.is_listening = False

            # 2. Process through orchestrator
            logger.debug("Processing through orchestrator...")
            response = await self.orchestrator.process(user_text)

            logger.info(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")

            # 3. Speak response
            if self.config.speak_responses and response:
                self._status.is_speaking = True
                try:
                    await self._tts.speak(response)
                finally:
                    self._status.is_speaking = False

            # Update statistics
            self._status.total_interactions += 1
            self._status.current_interaction = None

            logger.info("Interaction complete, returning to listening")

        except Exception as e:
            logger.error(f"Error processing interaction: {e}", exc_info=True)
            self._status.error = str(e)
            self._status.is_listening = False
            self._status.is_speaking = False
            self._status.current_interaction = None

            # Speak error message if possible
            try:
                if self.config.speak_responses:
                    await self._tts.speak("I'm sorry, I encountered an error processing your request.")
            except Exception:
                pass

    async def process_text(self, text: str) -> str:
        """Process text input directly (for testing/text mode).

        Bypasses wake word and STT, directly processes through orchestrator.

        Args:
            text: User input text.

        Returns:
            Response text.
        """
        logger.debug(f"Processing text input: {text[:100]}...")

        self._status.current_interaction = text

        try:
            response = await self.orchestrator.process(text)

            if self.config.speak_responses and response:
                await self._tts.speak(response)

            self._status.current_interaction = None
            return response

        except Exception as e:
            logger.error(f"Error processing text: {e}", exc_info=True)
            self._status.current_interaction = None
            raise

    async def speak(self, text: str) -> None:
        """Speak text without processing.

        Args:
            text: Text to speak.
        """
        await self._tts.speak(text)

    async def stop_speech(self) -> None:
        """Stop current speech playback."""
        await self._tts.stop()

    @property
    def status(self) -> dict[str, Any]:
        """Get pipeline status for menu bar/monitoring.

        Returns:
            Dictionary with status information.
        """
        return {
            "running": self._status.is_running,
            "listening": self._status.is_listening,
            "speaking": self._status.is_speaking,
            "current_interaction": self._status.current_interaction,
            "total_interactions": self._status.total_interactions,
            "last_wake_time": self._status.last_wake_time,
            "error": self._status.error,
            "wake_word_active": self._wake_word.is_active,
            "tts_playing": self._tts.is_playing,
        }

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running

    @property
    def is_listening(self) -> bool:
        """Check if currently listening for user input."""
        return self._status.is_listening

    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._status.is_speaking


async def create_voice_pipeline(
    orchestrator: "RoxyOrchestrator",
    config: VoiceConfig,
) -> VoicePipeline:
    """
    Create a voice pipeline.

    Convenience function that creates a VoicePipeline instance.

    Args:
        orchestrator: Roxy orchestrator for processing.
        config: Voice configuration.

    Returns:
        VoicePipeline instance.
    """
    pipeline = VoicePipeline(
        orchestrator=orchestrator,
        config=config,
    )

    return pipeline
