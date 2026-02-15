"""Tests for vox-voice Python bindings.

These tests verify that the native extension loads correctly and that
the wrapper classes can be instantiated without requiring actual model
files on disk.
"""


def test_import():
    """All public classes should be importable."""
    from vox_voice import (
        AudioOutput,
        KokoroTts,
        SileroVad,
        TranscribeResult,
        Vox,
        VoxListener,
        WhisperStt,
    )

    assert Vox is not None
    assert SileroVad is not None
    assert WhisperStt is not None
    assert KokoroTts is not None
    assert AudioOutput is not None
    assert TranscribeResult is not None
    assert VoxListener is not None


def test_version():
    """Package version should be set."""
    import vox_voice

    assert vox_voice.__version__ == "0.2.0"


def test_silero_vad_default():
    """SileroVad with no args should use the default model path."""
    from vox_voice import SileroVad

    vad = SileroVad()
    assert vad is not None
    assert "silero_vad.onnx" in repr(vad)


def test_silero_vad_custom_path():
    """SileroVad should accept a custom model path."""
    from vox_voice import SileroVad

    vad = SileroVad(model_path="/tmp/custom_vad.onnx")
    assert "/tmp/custom_vad.onnx" in repr(vad)


def test_whisper_stt_default():
    """WhisperStt with no args should default to tiny.en."""
    from vox_voice import WhisperStt

    stt = WhisperStt()
    assert stt is not None
    assert "tiny.en" in repr(stt)


def test_whisper_stt_model_name():
    """WhisperStt should resolve shorthand model names."""
    from vox_voice import WhisperStt

    stt = WhisperStt("base.en")
    assert "ggml-base.en.bin" in repr(stt)


def test_whisper_stt_full_path():
    """WhisperStt should accept a full file path."""
    from vox_voice import WhisperStt

    stt = WhisperStt("/models/ggml-tiny.en.bin")
    assert "/models/ggml-tiny.en.bin" in repr(stt)


def test_kokoro_tts_default():
    """KokoroTts with no args should use default model paths."""
    from vox_voice import KokoroTts

    tts = KokoroTts()
    assert tts is not None
    assert "kokoro-v1.0.onnx" in repr(tts)
    assert "voices.bin" in repr(tts)


def test_kokoro_tts_custom_paths():
    """KokoroTts should accept custom model paths."""
    from vox_voice import KokoroTts

    tts = KokoroTts(
        model_path="/tmp/kokoro.onnx",
        voices_path="/tmp/voices.bin",
    )
    assert "/tmp/kokoro.onnx" in repr(tts)
    assert "/tmp/voices.bin" in repr(tts)


def test_vox_construction():
    """Vox should be constructable with keyword args."""
    from vox_voice import KokoroTts, SileroVad, Vox, WhisperStt

    vox = Vox(vad=SileroVad(), stt=WhisperStt())
    assert vox is not None
    assert "tts=no" in repr(vox)


def test_vox_with_tts():
    """Vox should accept an optional TTS backend."""
    from vox_voice import KokoroTts, SileroVad, Vox, WhisperStt

    vox = Vox(vad=SileroVad(), stt=WhisperStt(), tts=KokoroTts())
    assert "tts=yes" in repr(vox)
