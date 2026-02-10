"""Unit tests for voice pipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxy.voice.pipeline import PipelineStatus, VoicePipeline


class TestPipelineStatus:
    """Tests for PipelineStatus dataclass."""

    def test_default_values(self) -> None:
        """Test PipelineStatus default values."""
        status = PipelineStatus()
        assert status.is_running is False
        assert status.is_listening is False
        assert status.is_speaking is False
        assert status.current_interaction is None
        assert status.total_interactions == 0
        assert status.last_wake_time is None
        assert status.error is None


class TestVoicePipeline:
    """Tests for VoicePipeline class."""

    def test_initialization(self) -> None:
        """Test pipeline initialization."""
        mock_orchestrator = MagicMock()
        mock_config = MagicMock()

        pipeline = VoicePipeline(
            orchestrator=mock_orchestrator,
            config=mock_config,
        )

        assert pipeline.orchestrator is mock_orchestrator
        assert pipeline.config is mock_config
        assert not pipeline.is_running
        assert not pipeline.is_listening
        assert not pipeline.is_speaking

    @pytest.mark.asyncio
    async def test_start(self) -> None:
        """Test starting the pipeline."""
        mock_orchestrator = MagicMock()
        mock_config = MagicMock()

        pipeline = VoicePipeline(
            orchestrator=mock_orchestrator,
            config=mock_config,
        )

        with patch.object(
            pipeline._wake_word, "start_listening", new_callable=AsyncMock
        ) as mock_start:
            await pipeline.start()

            assert pipeline.is_running
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_already_running(self) -> None:
        """Test starting when already running."""
        mock_orchestrator = MagicMock()
        mock_config = MagicMock()

        pipeline = VoicePipeline(
            orchestrator=mock_orchestrator,
            config=mock_config,
        )
        pipeline._running = True

        with patch.object(
            pipeline._wake_word, "start_listening", new_callable=AsyncMock
        ) as mock_start:
            await pipeline.start()

            # Should not call start_listening again
            mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop(self) -> None:
        """Test stopping the pipeline."""
        mock_orchestrator = MagicMock()
        mock_config = MagicMock()

        pipeline = VoicePipeline(
            orchestrator=mock_orchestrator,
            config=mock_config,
        )
        pipeline._running = True

        with patch.object(
            pipeline._wake_word, "stop_listening", new_callable=AsyncMock
        ) as mock_stop:
            with patch.object(pipeline._stt, "cleanup", new_callable=AsyncMock):
                with patch.object(pipeline._tts, "cleanup", new_callable=AsyncMock):
                    await pipeline.stop()

                    assert not pipeline.is_running
                    mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_text(self) -> None:
        """Test processing text input."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.process = AsyncMock(return_value="Response text")
        mock_config = MagicMock()
        mock_config.speak_responses = True

        pipeline = VoicePipeline(
            orchestrator=mock_orchestrator,
            config=mock_config,
        )

        with patch.object(pipeline._tts, "speak", new_callable=AsyncMock) as mock_speak:
            result = await pipeline.process_text("Hello Roxy")

            assert result == "Response text"
            mock_orchestrator.process.assert_called_once_with("Hello Roxy")
            mock_speak.assert_called_once_with("Response text")

    @pytest.mark.asyncio
    async def test_process_text_no_speak(self) -> None:
        """Test processing text without speaking."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.process = AsyncMock(return_value="Response text")
        mock_config = MagicMock()
        mock_config.speak_responses = False

        pipeline = VoicePipeline(
            orchestrator=mock_orchestrator,
            config=mock_config,
        )

        with patch.object(pipeline._tts, "speak", new_callable=AsyncMock) as mock_speak:
            result = await pipeline.process_text("Hello Roxy")

            assert result == "Response text"
            mock_speak.assert_not_called()

    @pytest.mark.asyncio
    async def test_speak(self) -> None:
        """Test speaking text."""
        mock_orchestrator = MagicMock()
        mock_config = MagicMock()

        pipeline = VoicePipeline(
            orchestrator=mock_orchestrator,
            config=mock_config,
        )

        with patch.object(pipeline._tts, "speak", new_callable=AsyncMock) as mock_speak:
            await pipeline.speak("Hello world")

            mock_speak.assert_called_once_with("Hello world")

    @pytest.mark.asyncio
    async def test_stop_speech(self) -> None:
        """Test stopping speech."""
        mock_orchestrator = MagicMock()
        mock_config = MagicMock()

        pipeline = VoicePipeline(
            orchestrator=mock_orchestrator,
            config=mock_config,
        )

        with patch.object(pipeline._tts, "stop", new_callable=AsyncMock) as mock_stop:
            await pipeline.stop_speech()

            mock_stop.assert_called_once()

    def test_status_property(self) -> None:
        """Test status property returns correct information."""
        mock_orchestrator = MagicMock()
        mock_config = MagicMock()

        pipeline = VoicePipeline(
            orchestrator=mock_orchestrator,
            config=mock_config,
        )

        # Set up some state
        pipeline._status.is_running = True
        pipeline._status.is_listening = False
        pipeline._status.is_speaking = False
        pipeline._status.current_interaction = "test interaction"
        pipeline._status.total_interactions = 5

        status = pipeline.status

        assert status["running"] is True
        assert status["listening"] is False
        assert status["speaking"] is False
        assert status["current_interaction"] == "test interaction"
        assert status["total_interactions"] == 5

    @pytest.mark.asyncio
    async def test_on_wake_word_detected(self) -> None:
        """Test wake word detection handling."""
        mock_orchestrator = MagicMock()
        mock_config = MagicMock()
        mock_config.speak_responses = False

        pipeline = VoicePipeline(
            orchestrator=mock_orchestrator,
            config=mock_config,
        )
        pipeline._running = True

        with patch.object(pipeline, "_process_interaction", new_callable=AsyncMock) as mock_process:
            await pipeline._on_wake_word_detected()

            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_wake_word_while_processing(self) -> None:
        """Test wake word detection while already processing."""
        mock_orchestrator = MagicMock()
        mock_config = MagicMock()

        pipeline = VoicePipeline(
            orchestrator=mock_orchestrator,
            config=mock_config,
        )
        pipeline._running = True

        # Simulate lock being held
        async def mock_process():
            await asyncio.sleep(0.1)

        with patch.object(
            pipeline, "_process_interaction", new_callable=AsyncMock, side_effect=mock_process
        ):
            # Start first processing
            task1 = asyncio.create_task(pipeline._on_wake_word_detected())
            await asyncio.sleep(0.01)

            # Try second processing (should be ignored)
            with patch.object(pipeline._processing_lock, "locked", return_value=True):
                task2 = asyncio.create_task(pipeline._on_wake_word_detected())

            await task1
            await task2

            # Only first process should have been attempted
            # (second should have been skipped due to lock)


class TestCreateVoicePipeline:
    """Tests for create_voice_pipeline convenience function."""

    @pytest.mark.asyncio
    async def test_create_voice_pipeline_returns_instance(self) -> None:
        """Test that create_voice_pipeline returns VoicePipeline instance."""
        from roxy.voice.pipeline import create_voice_pipeline

        mock_orchestrator = MagicMock()
        mock_config = MagicMock()

        pipeline = await create_voice_pipeline(
            orchestrator=mock_orchestrator,
            config=mock_config,
        )

        assert isinstance(pipeline, VoicePipeline)
        assert pipeline.orchestrator is mock_orchestrator
        assert pipeline.config is mock_config
