"""Speech-to-text using faster-whisper.

Converts voice input to text with support for streaming transcription
and voice activity detection.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Audio recording constants
SAMPLE_RATE = 16000  # 16kHz is optimal for Whisper
CHANNELS = 1
DTYPE = np.int16

# VAD (Voice Activity Detection) settings
SILENCE_THRESHOLD = 500  # RMS energy threshold for silence
SILENCE_DURATION = 1.0  # Seconds of silence to stop recording
MIN_SPEECH_DURATION = 0.3  # Minimum speech duration to consider valid


class SpeechToText:
    """
    Speech-to-text conversion using faster-whisper.

    Supports real-time transcription with VAD, file-based transcription,
    and streaming transcription for continuous audio.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        input_device: str | None = None,
    ) -> None:
        """Initialize the speech-to-text engine.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large).
                        Add ".en" for English-only models (faster, more accurate).
            device: Device to use for inference ("cpu", "cuda", "auto").
            compute_type: Computation type ("float16", "int8", "int8_float16").
            input_device: Optional audio input device name or index.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.input_device = input_device

        # Model loaded lazily
        self._model: object | None = None

        # Thread pool for blocking operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt_")

        logger.debug(f"SpeechToText initialized with model_size='{model_size}', device='{device}'")

    def _load_model(self) -> None:
        """Load the Whisper model (called in thread pool)."""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model '{self.model_size}' on device '{self.device}'...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(Path.home() / ".cache" / "whisper"),
            )
            logger.info(f"Whisper model '{self.model_size}' loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise

    async def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded (async wrapper)."""
        if self._model is None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._load_model)

    def _get_input_device(self) -> int | None:
        """Get the audio input device index.

        Returns:
            Device index or None for default device.
        """
        if self.input_device is None:
            return None

        try:
            if isinstance(self.input_device, str):
                devices = sd.query_devices()
                for idx, device in enumerate(devices):
                    if device["name"] == self.input_device and device["max_input_channels"] > 0:
                        logger.debug(f"Using audio device: {device['name']} (index {idx})")
                        return idx
                logger.warning(f"Could not find device '{self.input_device}', using default")
                return None
            return int(self.input_device)
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}. Using default device.")
            return None

    async def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file.

        Args:
            path: Path to audio file (WAV, MP3, FLAC, etc.).

        Returns:
            Transcribed text.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If transcription fails.
        """
        audio_path = Path(path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        await self._ensure_model_loaded()

        try:
            loop = asyncio.get_event_loop()

            def _transcribe() -> tuple[list[str], float]:
                """Blocking transcription in thread pool."""
                if self._model is None:
                    raise RuntimeError("Model not loaded")

                segments, info = self._model.transcribe(
                    str(audio_path),
                    language="en",
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 500,
                        "speech_pad_ms": 300,
                    },
                )

                # Collect all segments
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)

                logger.debug(f"Transcription completed: language={info.language}, "
                           f"probability={info.language_probability:.2f}")

                return text_parts, info.duration

            text_parts, duration = await loop.run_in_executor(self._executor, _transcribe)
            text = "".join(text_parts).strip()

            logger.info(f"Transcribed {duration:.2f}s of audio: {len(text)} characters")

            return text

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to transcribe audio file: {e}") from e

    async def listen_until_silence(
        self,
        timeout: float = 10.0,
        silence_threshold: float | None = None,
        min_duration: float = MIN_SPEECH_DURATION,
    ) -> str:
        """
        Listen to microphone until silence is detected.

        Records audio from the microphone, detects speech using VAD,
        and stops when silence is detected or timeout is reached.

        Args:
            timeout: Maximum recording duration in seconds.
            silence_threshold: RMS energy threshold for silence (default auto).
            min_duration: Minimum speech duration to consider valid.

        Returns:
            Transcribed text, or empty string if no speech detected.
        """
        await self._ensure_model_loaded()

        # Auto-calculate silence threshold if not provided
        if silence_threshold is None:
            silence_threshold = SILENCE_THRESHOLD

        device = self._get_input_device()

        try:
            # Record audio
            audio_data = await self._record_audio(
                timeout=timeout,
                silence_threshold=silence_threshold,
                min_duration=min_duration,
                device=device,
            )

            if audio_data is None or len(audio_data) == 0:
                logger.info("No speech detected")
                return ""

            # Calculate audio duration
            duration = len(audio_data) / SAMPLE_RATE
            logger.info(f"Recorded {duration:.2f}s of audio")

            # Save to temporary WAV file for transcription
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            self._save_wav(tmp_path, audio_data, SAMPLE_RATE)

            try:
                # Transcribe the recording
                text = await self.transcribe_file(tmp_path)
                return text
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"listen_until_silence failed: {e}", exc_info=True)
            return ""

    async def _record_audio(
        self,
        timeout: float,
        silence_threshold: float,
        min_duration: float,
        device: int | None,
    ) -> np.ndarray | None:
        """Record audio with VAD until silence is detected.

        Args:
            timeout: Maximum recording duration.
            silence_threshold: RMS threshold for silence detection.
            min_duration: Minimum speech duration before stopping.
            device: Audio device index or None for default.

        Returns:
            Recorded audio as int16 numpy array, or None if no speech.
        """
        # Audio buffers
        audio_buffer: list[np.ndarray] = []
        silence_frames = 0
        speech_detected = False
        total_frames = 0
        max_frames = int(timeout * SAMPLE_RATE)
        silence_frames_threshold = int(SILENCE_DURATION * SAMPLE_RATE)
        min_frames = int(min_duration * SAMPLE_RATE)

        # Callback for audio stream
        def _audio_callback(indata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:
            """Process incoming audio chunks."""
            nonlocal silence_frames, speech_detected, total_frames

            if status:
                logger.warning(f"Audio callback status: {status}")

            # Convert to int16 for VAD
            audio_int16 = (indata * 32767).astype(np.int16)
            audio_buffer.append(audio_int16.copy())

            # Calculate RMS energy for VAD
            rms = float(np.sqrt(np.mean(audio_int16 ** 2)))

            # Detect speech/silence
            if rms >= silence_threshold:
                speech_detected = True
                silence_frames = 0
            elif speech_detected:
                silence_frames += frames

            total_frames += frames

        # Create and start audio stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=_audio_callback,
            device=device,
        )

        try:
            stream.start()

            # Monitor recording state
            while total_frames < max_frames:
                await asyncio.sleep(0.05)  # Check every 50ms

                # Stop if we have silence after speech
                if speech_detected and silence_frames >= silence_frames_threshold:
                    # Ensure minimum duration
                    if total_frames >= min_frames:
                        logger.debug(f"Silence detected after {total_frames / SAMPLE_RATE:.2f}s")
                        break

                # Continue listening if we haven't heard speech yet
                if not speech_detected and total_frames >= max_frames // 2:
                    logger.debug("No speech detected in first half of timeout")
                    break

        finally:
            stream.stop()
            stream.close()

        # Check if we captured any speech
        if not speech_detected or total_frames < min_frames:
            return None

        # Concatenate audio buffer
        if audio_buffer:
            return np.concatenate(audio_buffer)
        return None

    def _save_wav(self, path: str, audio: np.ndarray, sample_rate: int) -> None:
        """Save audio data to WAV file.

        Args:
            path: Output file path.
            audio: Audio data as int16 numpy array.
            sample_rate: Sample rate in Hz.
        """
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 2 bytes for int16
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
    ) -> AsyncGenerator[str, None]:
        """Transcribe streaming audio chunks.

        Yields transcription results as they become available.

        Args:
            audio_stream: Async generator yielding audio bytes.

        Yields:
            Transcribed text chunks.
        """
        await self._ensure_model_loaded()

        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Accumulate audio for transcription
        audio_buffer: list[np.ndarray] = []
        chunk_duration = 2.0  # Process in 2-second chunks
        chunk_samples = int(chunk_duration * SAMPLE_RATE)

        try:
            async for audio_bytes in audio_stream:
                # Convert bytes to numpy array
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_buffer.append(audio_chunk)

                # Process when we have enough audio
                total_samples = sum(len(chunk) for chunk in audio_buffer)

                if total_samples >= chunk_samples:
                    # Concatenate and transcribe
                    audio_data = np.concatenate(audio_buffer)
                    audio_buffer = []  # Clear buffer

                    # Save to temp file for transcription
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = tmp.name

                    try:
                        self._save_wav(tmp_path, audio_data, SAMPLE_RATE)

                        # Transcribe chunk
                        text = await self.transcribe_file(tmp_path)
                        if text:
                            yield text

                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Stream transcription error: {e}", exc_info=True)
            raise

        # Process remaining audio
        if audio_buffer:
            audio_data = np.concatenate(audio_buffer)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                self._save_wav(tmp_path, audio_data, SAMPLE_RATE)
                text = await self.transcribe_file(tmp_path)
                if text:
                    yield text
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._executor.shutdown(wait=True)
        logger.debug("SpeechToText cleanup complete")


async def create_stt(
    model_size: str = "base.en",
    device: str = "cpu",
    input_device: str | None = None,
) -> SpeechToText:
    """
    Create a speech-to-text engine.

    Convenience function that creates an STT instance.

    Args:
        model_size: Whisper model size.
        device: Device for inference.
        input_device: Optional audio input device.

    Returns:
        SpeechToText instance.
    """
    stt = SpeechToText(
        model_size=model_size,
        device=device,
        input_device=input_device,
    )

    return stt
