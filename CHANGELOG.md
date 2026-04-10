# Changelog

All notable changes to Vox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-04-10

### Added
- Distil-Whisper backend for 6x faster speech recognition with minimal accuracy loss
- INT8 quantization support for 4x smaller models (Whisper, Distil-Whisper, Kokoro, Pocket)
- Model caching in server mode to eliminate cold start latency
- `vox test` command for audio I/O diagnostics and troubleshooting
- `vox benchmark` command for performance testing STT and TTS backends
- `vox config` command for interactive setup wizard
- `distil-whisper` feature flag for enabling Distil-Whisper backend
- Raspberry Pi deployment guide with recommended configurations
- Quantized model downloads (tiny.en-int8, base.en-int8, kokoro-int8, pocket-int8)

### Changed
- Bumped version to 0.5.0
- Updated README with performance optimization sections
- Expanded model table to include Distil-Whisper and quantized variants

### Performance
- Raspberry Pi 4: Whisper base.en now runs in real-time (was 3.3x RTF, now 0.8x with Distil-Whisper + INT8)
- Server cold start reduced from ~2s to instant with model caching
- Model downloads 4x smaller with INT8 quantization
- STT inference 6x faster with Distil-Whisper backend
- Memory footprint reduced by 50% with quantized models

### Fixed
- Server mode now properly caches models between requests
- Quantized model downloads no longer require separate configuration

## [0.4.1] - 2025-04-09

### Fixed
- Missing `pre_speech_pad_ms` field in VAD benchmarks
- Redundant closure in WebSocket TTS handler (clippy warning)

### Changed
- Code formatting improvements

## [0.4.0] - 2025-04-08

### Added
- Gapless streaming playback for TTS
- WebSocket TTS endpoint for real-time synthesis
- Support for streaming audio chunks without gaps

### Changed
- Improved WebSocket audio streaming architecture

## [0.3.0] - 2025-03-15

### Added
- Chatterbox TTS backend for voice cloning
- Pocket TTS backend for pure Rust synthesis
- Piper TTS backend for multilingual synthesis
- CoreML support for Chatterbox on macOS
- Metal GPU acceleration for Pocket on Apple Silicon

### Changed
- Refactored TTS backends to use common trait
- Improved audio playback pipeline

## [0.2.0] - 2025-02-10

### Added
- Sherpa-ONNX STT backend for multilingual support
- Streaming STT with partial results
- WebSocket server for real-time transcription
- HTTP API server with REST endpoints
- Web UI for browser-based voice interaction

### Changed
- Split CLI and server into separate features
- Improved error handling in audio pipeline

## [0.1.0] - 2025-01-15

### Added
- Initial release
- Whisper STT backend (tiny to medium models)
- Kokoro TTS backend with 50+ voices
- Silero VAD v5 for speech detection
- CLI commands: `vox listen`, `vox speak`, `vox chat`
- Python bindings via PyO3
- Model management CLI (`vox models`)
- Voice chat with Ollama integration

### Core Features
- Local-first voice AI (no cloud dependencies)
- Real-time microphone transcription
- Text-to-speech synthesis
- VAD-based utterance detection
- Cross-platform support (macOS, Linux, Windows)
