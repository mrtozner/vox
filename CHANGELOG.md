# Changelog

All notable changes to Vox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-11-23

### Added

- Voice conversations can use a local OpenAI-compatible runtime, such as llama.cpp or MLX LM. Select it with `VOX_LLM_BACKEND=openai-compatible`.
- Live Talk reports the time from the server's turn-end signal to the start of reply playback, including transcription.

### Changed

- The browser transfers completed microphone buffers directly from the audio worklet to the main thread.

### Fixed

- Offline Ollama requests bypass proxies and refuse redirects. `localhost` resolves directly to loopback.
- A forced model refresh keeps the installed file until its replacement passes verification. Offline refusal leaves existing model files intact.
- Voiceprint collection honors consent withdrawals during enrollment and deferred adaptation writes.
- Overlapping Live Talk connections share the same forget state. Playback feedback applies to the correct turn, and open sessions stay available through idle sweeps and capacity eviction.
- Release builds pin Whisper's CPU features on Linux and Windows. Linux and macOS archives include Piper's eSpeak NG runtime data, and the server falls back to Kokoro when Piper's phonemizer cannot start.

## [0.7.0] - 2026-11-22

### Changed
- The Windows release binary is built without Piper (espeak-ng does not compile with MSVC); Live Talk uses Kokoro there
- **BREAKING**: `GET /v1/models` now returns the OpenAI list shape. Vox's own model report moved to `GET /v1/vox/models`
- **BREAKING**: the Cargo package is now `vox-voice` (the `vox` name on crates.io belongs to another project). The library crate is still `vox` and the binary is still `vox`, so `use vox::...` and the command line are unchanged. Depend on it with `vox = { package = "vox-voice", git = "..." }`
- **Release binaries are GPL-3.0.** They are built with the new `dist` feature (`cli`, `server`, `kokoro`, `piper`), and Piper links espeak-ng; Kokoro's phonemizer is derived from espeak-ng too. The source code stays MIT OR Apache-2.0. See `NOTICE`
- Release binaries need glibc 2.39+ on Linux, macOS 13.4+, and an x86-64-v3 CPU (AVX2: Intel Haswell / AMD Zen or newer) on x86_64; these floors come from `ort` 2.0.0-rc.11's ONNX Runtime
- Minimum supported Rust version is 1.88 (required by `ort` 2.0.0-rc.11)
- Release builds use thin LTO, one codegen unit and stripped symbols
- The web UI opens on the Live Talk tab and was redesigned
- The server binds to loopback only and refuses a non-loopback `--host` unless `VOX_ALLOW_REMOTE=1`; cross-origin requests are rejected unless allow-listed (`VOX_ALLOWED_ORIGINS`, `VOX_ALLOWED_HOSTS`)
- Speaker enrollment is opt-in: voiceprints are stored and adapted only after consent is recorded
- The models directory is resolved at run time (`VOX_MODELS_DIR`, then a source checkout's `models/`, then the per-user data directory); nothing from the build machine is baked into the binary
- Model downloads are pinned to a revision and verified by size and SHA-256
- `vox chat` streams from Ollama `/api/chat` and keeps conversation memory
- `vox speak` with an unknown `--backend` lists the backends the binary was actually built with
- The HTTP/WebSocket server is part of the library (`vox::server`, `server` feature) instead of being compiled into the binary by path; the `server` feature now also enables `silero`
- `cpal` is an optional dependency behind the new `mic` feature and Ctrl-C handling is behind the new `signal` feature. Both are on by default. **With `default-features = false`, add `mic` to keep `Vox::builder().build()` and `vox::audio::AudioCapture`, and `signal` to keep Ctrl-C in `Vox::listen()`**
- The library no longer enables `tokio`'s `full` feature set: the core asks for `rt`, `sync`, `macros` and `time`, and each Cargo feature adds what it uses. An application that relied on vox to turn on other tokio features must now enable them itself

### Added
- OpenAI-compatible endpoints: `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`, `GET /v1/models`, `GET /v1/models/{id}`
- Live Talk protocol v2: `protocol_version` and `session_id` in `ready`, `turn_id` on turn-scoped events, `turn_metrics`, `memory`, `playback_done`, `forget_memory`; memory survives a reconnect with the same `session_id`
- Speaker REST API: list, rename, forget, merge, export, and enrollment consent under `/v1/speakers`
- Offline mode: `--offline` (global flag, or `VOX_OFFLINE=1`) refuses an Ollama host that is not loopback and refuses all model downloads. Not covered: the `pocket`, `chatterbox` and `qwen3` backends, which fetch weights through the Hugging Face hub client
- `GET /v1/privacy/ledger`: outbound connections made by the server process (destination, purpose, body bytes)
- `vox tune`: measures the installed STT, TTS and Ollama models on a spoken sample and saves the best as the defaults of `vox chat` (`[tune]` in `config.toml`); `--dry-run` loads, connects and writes nothing
- Live Talk identifies known speakers (read-only lookup; the model gets `[Name] text`). In Live Talk a voice is enrolled only by the explicit `enroll_speaker` command, and only with consent
- Per-person long-term memory, off until switched on per speaker: `GET/PUT/DELETE /v1/speakers/{id}/memory`. Forgetting a speaker deletes it; the speaker export includes it
- Distill on evict: Live Talk turns that leave the context window are distilled into facts by the conversation's Ollama model while idle (`memory_distilled`; `VOX_MEMORY_DISTILL=0` turns it off)
- `VOX_SESSION_TTL_SECS`: lifetime of an idle Live Talk session (default 30 minutes)
- `GET /health/ready` readiness probe; graceful shutdown; request, text-length and concurrent-inference limits
- `vox doctor`: explains which models directory is used and which models are missing or corrupt
- `vox serve --download`, `--language`, `--stt-model`, `--piper-voice`, `--ollama-host`; `vox listen --language`; `vox chat --language`
- Multilingual Whisper models, Whisper large-v3-turbo (including quantized variants) and a Turkish Piper voice in the model registry
- Release archives for Linux aarch64, plus `SHA256SUMS` and build provenance attestations for every archive
- `NOTICE`, `LICENSE-GPL-3.0`, and `docs/raspberry_pi.md`
- Embedding API: `VoxBuilder::build_with_channel` (engine fed from a channel, no audio device), `Vox::listen_until` (caller-supplied stop condition, no signal handler), and `vox::server::{build_router, serve, run_with_config, ServerState}`

### Fixed
- Kokoro and Piper in one binary: `kokoro-tts` is vendored with its phoneme buffer made private. Upstream exports a `ph_list2` symbol that clashes with espeak-ng's, which broke Linux links with lld (default since Rust 1.100) and could make the two engines share a buffer
- Live Talk: a short noise (a cough, a click) while a reply is being transcribed or generated no longer cancels it; speech has to last 250 ms before it counts as a barge-in
- Microphone capture resamples from the device's real sample rate and feeds the VAD whole frames
- Silero VAD receives the context samples the v5 model expects; utterance length is capped
- Whisper honours the configured language instead of always assuming English
- Speakers are no longer wiped at startup; speaker IDs are not reused after a delete
- Kokoro reads numbers aloud and no longer truncates long text; the voices file is the one `kokoro-tts` reads
- Piper selects the speaker per request instead of sharing mutable state across requests
- Qwen3 inference no longer blocks the async runtime
- Converse keeps reading its socket while a turn is in progress
- README: CLI examples, model table, storage paths, Python instructions and the speaker model description now match the code

### Removed
- `Formula/vox.rb`: it pointed at a v0.2.0 source tarball with a placeholder checksum
- `Dockerfile.static`, `scripts/build_static.sh`, `scripts/build_for_raspberry_pi.sh` and `.cargo/config.toml`: the static-musl and cross-compile recipes did not build the current crate (missing vendored manifests, a `quantized` feature that does not exist, invalid `[target.*.env]` tables). Use the release archives or see `docs/raspberry_pi.md`
- No prebuilt binary for Intel macOS: `ort` 2.0.0-rc.11 ships no ONNX Runtime for that target
- Unused `ModelCache`, `StreamingPipeline` and `intelligence` modules

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
