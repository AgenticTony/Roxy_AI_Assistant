"""Unit tests for speech-to-text."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from roxy.voice.stt import SpeechToText


class TestSpeechToText:
    """Tests for SpeechToText class."""

    def test_initialization(self) -> None:
        """Test STT initialization with default values."""
        stt = SpeechToText()
        assert stt.model_size == "base.en"
        assert stt.device == "cpu"
        assert stt.compute_type == "int8"
        assert stt.input_device is None

    def test_initialization_with_params(self) -> None:
        """Test STT initialization with custom parameters."""
        stt = SpeechToText(
            model_size="small",
            device="cuda",
            compute_type="float16",
            input_device="Built-in Microphone",
        )
        assert stt.model_size == "small"
        assert stt.device == "cuda"
        assert stt.compute_type == "float16"
        assert stt.input_device == "Built-in Microphone"

    @pytest.mark.asyncio
    async def test_ensure_model_loaded(self) -> None:
        """Test that model is loaded on demand."""
        stt = SpeechToText()

        with patch.object(stt, "_load_model", new_callable=AsyncMock) as mock_load:
            await stt._ensure_model_loaded()
            mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_model_loaded_skips_if_loaded(self) -> None:
        """Test that model loading is skipped if already loaded."""
        stt = SpeechToText()
        stt._model = MagicMock()  # Simulate already loaded

        with patch.object(stt, "_load_model", new_callable=AsyncMock) as mock_load:
            await stt._ensure_model_loaded()
            mock_load.assert_not_called()

    def test_get_input_device_with_string(self) -> None:
        """Test getting input device by name."""
        stt = SpeechToText(input_device="Test Device")

        with patch("roxy.voice.stt.sd.query_devices") as mock_query:
            mock_query.return_value = [
                {"name": "Test Device", "max_input_channels": 1, "index": 0},
            ]

            device_idx = stt._get_input_device()
            assert device_idx == 0

    def test_save_wav(self) -> None:
        """Test saving audio to WAV file."""
        stt = SpeechToText()

        import tempfile

        audio = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            stt._save_wav(tmp_path, audio, 16000)

            # Verify file exists and has content
            assert Path(tmp_path).exists()
            assert Path(tmp_path).stat().st_size > 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self) -> None:
        """Test transcribing non-existent file raises error."""
        stt = SpeechToText()

        with pytest.raises(FileNotFoundError):
            await stt.transcribe_file("/nonexistent/file.wav")

    @pytest.mark.asyncio
    async def test_transcribe_file_success(self) -> None:
        """Test successful file transcription."""
        stt = SpeechToText()

        # Mock model loading
        with patch.object(stt, "_ensure_model_loaded", new_callable=AsyncMock):
            # Create a mock model
            mock_model = MagicMock()
            mock_segments = [
                MagicMock(text="Hello ", end=1.0),
                MagicMock(text="world", end=2.0),
            ]
            mock_model.transcribe.return_value = (mock_segments, MagicMock(language="en", language_probability=0.95))
            stt._model = mock_model

            # Create a temporary audio file
            import tempfile
            import wave

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                # Create valid WAV file
                with wave.open(tmp_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

                result = await stt.transcribe_file(tmp_path)

                assert result == "Hello world"

            finally:
                Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_listen_until_silence_no_speech(self) -> None:
        """Test listen_until_silence with no speech detected."""
        stt = SpeechToText()

        with patch.object(stt, "_ensure_model_loaded", new_callable=AsyncMock):
            with patch.object(stt, "_record_audio", return_value=None) as mock_record:
                result = await stt.listen_until_silence(timeout=5.0)

                assert result == ""
                mock_record.assert_called_once()

    @pytest.mark.asyncio
    async def test_listen_until_silence_with_speech(self) -> None:
        """Test listen_until_silence with speech detected."""
        stt = SpeechToText()
        audio_data = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)

        with patch.object(stt, "_ensure_model_loaded", new_callable=AsyncMock):
            with patch.object(stt, "_record_audio", return_value=audio_data):
                with patch.object(stt, "transcribe_file", return_value="Hello world") as mock_transcribe:
                    result = await stt.listen_until_silence(timeout=5.0)

                    assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_cleanup(self) -> None:
        """Test cleanup shuts down thread pool."""
        stt = SpeechToText()

        await stt.cleanup()

        # Executor should be shut down
        assert stt._executor._shutdown is True


class TestCreateSTT:
    """Tests for create_stt convenience function."""

    @pytest.mark.asyncio
    async def test_create_stt_returns_instance(self) -> None:
        """Test that create_stt returns SpeechToText instance."""
        from roxy.voice.stt import create_stt

        stt = await create_stt()

        assert isinstance(stt, SpeechToText)
        assert stt.model_size == "base.en"

    @pytest.mark.asyncio
    async def test_create_stt_with_custom_params(self) -> None:
        """Test create_stt with custom parameters."""
        from roxy.voice.stt import create_stt

        stt = await create_stt(
            model_size="small",
            device="cuda",
        )

        assert stt.model_size == "small"
        assert stt.device == "cuda"
