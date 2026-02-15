"""Vox -- local voice AI framework.

Whisper STT + Silero VAD + Kokoro TTS. No cloud required.

Basic usage::

    from vox_voice import Vox, SileroVad, WhisperStt

    vox = Vox(vad=SileroVad(), stt=WhisperStt("tiny.en"))
    for result in vox.listen():
        print(result.text)
"""

from vox_voice._vox_voice import (
    AudioOutput,
    KokoroTts,
    SileroVad,
    TranscribeResult,
    Vox,
    VoxListener,
    WhisperStt,
)

__version__ = "0.1.0"
__all__ = [
    "AudioOutput",
    "KokoroTts",
    "SileroVad",
    "TranscribeResult",
    "Vox",
    "VoxListener",
    "WhisperStt",
]
