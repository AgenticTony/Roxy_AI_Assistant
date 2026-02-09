"""Roxy voice module.

Contains wake word detection, speech-to-text, text-to-speech,
voice pipeline orchestration, and Talon Voice integration.
"""

from roxy.voice.pipeline import VoicePipeline, PipelineStatus, create_voice_pipeline
from roxy.voice.stt import SpeechToText, create_stt
from roxy.voice.talon_bridge import TalonBridge, create_talon_bridge
from roxy.voice.tts import TextToSpeech, create_tts
from roxy.voice.wake_word import WakeWordDetector, detect_wake_word

__all__ = [
    # Pipeline
    "VoicePipeline",
    "PipelineStatus",
    "create_voice_pipeline",
    # Speech-to-Text
    "SpeechToText",
    "create_stt",
    # Text-to-Speech
    "TextToSpeech",
    "create_tts",
    # Wake Word
    "WakeWordDetector",
    "detect_wake_word",
    # Talon Bridge
    "TalonBridge",
    "create_talon_bridge",
]
