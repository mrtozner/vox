# Changelog

All notable changes to Vox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-04-12

### Added
- **Speaker Diarization** *(experimental)* — real-time speaker identification in the Listen tab
  - ECAPA-TDNN speaker encoder with 512-dimensional voice embeddings
  - Automatic speaker enrollment and recognition with cosine similarity
  - SQLite-backed speaker database (`~/Library/Application Support/vox/speakers.db`)
  - Per-speaker colored transcript entries (Tableau 10 colorblind-friendly palette)
  - Speaker sidebar with rename and forget controls
  - `diarization` feature flag (auto-enabled by `server`)
- **Live Talk** — barge-in voice chat (`/v1/live-talk` WebSocket, experimental)
  - Full-duplex conversation: speak while the LLM is still responding
  - VAD + STT + LLM + TTS in a single WebSocket connection
  - Interrupt TTS playback by speaking (barge-in)
  - Voice and LLM model selection in the web UI
- **Continuous Voice Chat** — `/v1/converse` WebSocket endpoint
  - Persistent voice chat sessions with VAD-based turn management
- **Capability Registry** — `/v1/capabilities` endpoint
  - Reports available backends, features, and hardware at runtime
- Qwen3-TTS backend with state-of-the-art speech synthesis
  - 0.6B and 1.7B CustomVoice model variants
  - 20 voices across 10 languages
  - Apple Metal / NVIDIA CUDA GPU acceleration
  - Streaming synthesis with chunked audio output (~800ms chunks)
- Streaming TTS via `/v1/speak` WebSocket with gapless playback
- Web UI: 5-tab interface (Listen, Speak, Chat, Live Talk, Dashboard)
- Kokoro TTS: 57 voices across 9 languages with voice selection dropdown

### Fixed
- ONNX Runtime SIGSEGV crash during speaker diarization (borrowed tensor + wrong input names)
- Piper TTS "Invalid speaker id 0" error on single-speaker models
- Speaker colors: replaced DJB2 hash (all greens) with Tableau 10 distinct palette
- Forget button now always visible (0.4 opacity) instead of hidden until hover
- Diarization mel-spectrogram DFT performance (pre-computed twiddle factors)
- Speaker fragmentation (5+ IDs for 2 people): added CMVN normalization, lowered threshold to 0.35, EMA embedding adaptation
- Live Talk echo/STT degradation: VAD gating during TTS playback, VAD reset after turns
- Live Talk audio playback: dedicated AudioContext for TTS (separate from 16kHz mic context)
- Whisper hallucination during silence: `no_speech_thold(0.6)` filtering
- Stale speaker database: embedding version tracking auto-clears speakers on preprocessing changes

### Changed
- **BREAKING**: Renamed TTS HTTP endpoint from `/v1/tts` to `/v1/synthesize`
- Default TTS priority: Qwen3 → Kokoro → Piper (when features enabled)
- Voice chat uses Piper TTS by default for fast CPU-only synthesis
- Speaker diarization uses `std::sync::Mutex` + `spawn_blocking` for thread-safe ONNX inference

### Performance
- M4 Pro: Qwen3 0.6B RTF 0.96 (2.6GB RAM), Qwen3 1.7B RTF 1.14 (4.4GB RAM)
- Diarization: pre-computed twiddle factors eliminate all trig calls from DFT hot loop
- Metal GPU acceleration for near real-time synthesis on Apple Silicon

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
