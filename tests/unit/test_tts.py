"""Unit tests for text-to-speech."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from roxy.voice.tts import AVAILABLE_VOICES, DEFAULT_VOICE, TextToSpeech


class TestTextToSpeech:
    """Tests for TextToSpeech class."""

    def test_initialization(self) -> None:
        """Test TTS initialization with default values."""
        tts = TextToSpeech()
        assert tts.voice == DEFAULT_VOICE
        assert tts.speed == 1.1
        assert tts.use_fallback is False
        assert tts.output_device is None

    def test_initialization_with_params(self) -> None:
        """Test TTS initialization with custom parameters."""
        tts = TextToSpeech(
            voice="af_bella",
            speed=1.5,
            use_fallback=True,
            output_device="Built-in Output",
        )
        assert tts.voice == "af_bella"
        assert tts.speed == 1.5
        assert tts.use_fallback is True
        assert tts.output_device == "Built-in Output"

    def test_check_kokoro_available(self) -> None:
        """Test checking if Kokoro is available."""
        tts = TextToSpeech()

        with patch("roxy.voice.tts.mlx_audio") as mock_mlx:
            tts._check_kokoro_available()
            assert tts._kokoro_available is True

    def test_check_kokoro_not_available(self) -> None:
        """Test handling when Kokoro is not available."""
        tts = TextToSpeech()

        with patch("roxy.voice.tts.mlx_audio", side_effect=ImportError):
            tts._check_kokoro_available()
            assert tts._kokoro_available is False

    def test_map_voice_to_macos(self) -> None:
        """Test mapping Kokoro voices to macOS voices."""
        tts = TextToSpeech()

        # Test various mappings
        assert tts._map_voice_to_macos("af_heart") == "Samantha"
        assert tts._map_voice_to_macos("am_michael") == "Alex"
        assert tts._map_voice_to_macos("bf_emma") == "Moira"
        assert tts._map_voice_to_macos("unknown") == "Samantha"  # Default

    def test_get_output_device_with_string(self) -> None:
        """Test getting output device by name."""
        tts = TextToSpeech(output_device="Test Device")

        with patch("roxy.voice.tts.sd.query_devices") as mock_query:
            mock_query.return_value = [
                {"name": "Test Device", "max_output_channels": 2, "index": 0},
            ]

            device_idx = tts._get_output_device()
            assert device_idx == 0

    def test_get_output_device_with_index(self) -> None:
        """Test getting output device by index."""
        tts = TextToSpeech(output_device=2)

        device_idx = tts._get_output_device()
        assert device_idx == 2

    def test_get_output_device_not_found(self) -> None:
        """Test handling of non-existent device."""
        tts = TextToSpeech(output_device="Non-existent Device")

        with patch("roxy.voice.tts.sd.query_devices") as mock_query:
            mock_query.return_value = [
                {"name": "Test Device", "max_output_channels": 2, "index": 0},
            ]

            device_idx = tts._get_output_device()
            assert device_idx is None

    @pytest.mark.asyncio
    async def test_speak_empty_text(self) -> None:
        """Test speaking empty text does nothing."""
        tts = TextToSpeech()

        # Should not raise
        await tts.speak("")
        await tts.speak("   ")

    @pytest.mark.asyncio
    async def test_speak_with_fallback(self) -> None:
        """Test speaking with fallback to 'say' command."""
        tts = TextToSpeech(use_fallback=True)

        with patch("roxy.voice.tts.subprocess.run") as mock_run:
            await tts.speak("Hello world")

            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "say" in args
            assert "Hello world" in args

    @pytest.mark.asyncio
    async def test_speak_fallback_error_handling(self) -> None:
        """Test error handling in fallback TTS."""
        tts = TextToSpeech(use_fallback=True)

        with patch("roxy.voice.tts.subprocess.run", side_effect=Exception("Test error")):
            # Should not raise, just log error
            await tts.speak("Hello world")

    @pytest.mark.asyncio
    async def test_stop(self) -> None:
        """Test stopping speech playback."""
        tts = TextToSpeech()
        tts._is_playing = True
        tts._stop_requested = False

        with patch.object(tts, "_stream"):
            await tts.stop()

            assert tts._stop_requested is True
            assert not tts._is_playing

    @pytest.mark.asyncio
    async def test_is_playing_property(self) -> None:
        """Test is_playing property."""
        tts = TextToSpeech()

        assert not tts.is_playing

        tts._is_playing = True
        assert tts.is_playing

    @pytest.mark.asyncio
    async def test_is_available_property(self) -> None:
        """Test is_available property."""
        tts = TextToSpeech()

        # With Kokoro available
        tts._kokoro_available = True
        assert tts.is_available

        # With fallback
        tts._kokoro_available = False
        tts.use_fallback = True
        assert tts.is_available

    @pytest.mark.asyncio
    async def test_cleanup(self) -> None:
        """Test cleanup shuts down thread pool."""
        tts = TextToSpeech()

        await tts.cleanup()

        # Executor should be shut down
        assert tts._executor._shutdown is True


class TestCreateTTS:
    """Tests for create_tts convenience function."""

    @pytest.mark.asyncio
    async def test_create_tts_returns_instance(self) -> None:
        """Test that create_tts returns TextToSpeech instance."""
        from roxy.voice.tts import create_tts

        tts = await create_tts()

        assert isinstance(tts, TextToSpeech)
        assert tts.voice == DEFAULT_VOICE

    @pytest.mark.asyncio
    async def test_create_tts_with_custom_params(self) -> None:
        """Test create_tts with custom parameters."""
        from roxy.voice.tts import create_tts

        tts = await create_tts(
            voice="am_michael",
            speed=1.3,
            use_fallback=True,
        )

        assert tts.voice == "am_michael"
        assert tts.speed == 1.3
        assert tts.use_fallback is True


class TestAvailableVoices:
    """Tests for available voices constant."""

    def test_available_voices_contains_common_voices(self) -> None:
        """Test that AVAILABLE_VOICES contains expected voices."""
        assert "af_heart" in AVAILABLE_VOICES
        assert "am_michael" in AVAILABLE_VOICES
        assert "bf_emma" in AVAILABLE_VOICES

    def test_default_voice_is_available(self) -> None:
        """Test that DEFAULT_VOICE is in AVAILABLE_VOICES."""
        assert DEFAULT_VOICE in AVAILABLE_VOICES
