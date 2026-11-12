<p align="center">
  <img src="assets/vox-logo.jpg" alt="Vox" width="200">
  <p align="center"><strong>Local-first voice AI framework. Speech-to-text, text-to-speech, speaker diarization, and voice chat &mdash; all running on your machine.</strong></p>
</p>

<p align="center">
  <a href="https://github.com/mrtozner/vox/actions/workflows/ci.yml"><img src="https://github.com/mrtozner/vox/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/mrtozner/vox/releases/latest"><img src="https://img.shields.io/github/v/release/mrtozner/vox" alt="Latest release"></a>
  <a href="#license"><img src="https://img.shields.io/badge/source-MIT%2FApache--2.0-blue.svg" alt="Source license: MIT or Apache-2.0"></a>
  <a href="#license"><img src="https://img.shields.io/badge/release%20binaries-GPL--3.0-blue.svg" alt="Release binaries: GPL-3.0"></a>
</p>

---

No API keys. No cloud. No data leaving your machine.

```
Mic --> VAD (Silero) --> STT (Whisper/Sherpa) --> Speaker ID --> Your Code --> TTS (Kokoro/Piper/Qwen3) --> Speaker
```

<br>

## Highlights

- **Speech-to-Text** &mdash; Whisper, Distil-Whisper (6x faster), Sherpa-ONNX (streaming + multilingual)
- **Text-to-Speech** &mdash; Kokoro (57 voices, 9 languages), Piper (multilingual), Qwen3, Pocket (pure Rust), Chatterbox (voice cloning)
- **Speaker Diarization** *(experimental)* &mdash; Know *who* is speaking. Real-time speaker identification with voice embeddings and a local speaker database. Voiceprints are stored only after you give consent
- **Live Talk** *(experimental)* &mdash; Barge-in voice chat. Talk to an LLM and interrupt it mid-sentence, just like a real conversation
- **Web Interface** &mdash; 5-tab browser UI: Live Talk, Listen, Speak, Chat, Dashboard
- **HTTP + WebSocket API** &mdash; REST endpoints (including OpenAI-compatible `/v1/audio/*`) and 4 WebSocket channels for real-time streaming
- **Fully Local** &mdash; Everything runs on-device. Works offline after model download
- **Pluggable Backends** &mdash; Swap any component via Rust traits

<br>

## Quick Start

### Prebuilt binary

Download the archive for your platform from the [latest release](https://github.com/mrtozner/vox/releases/latest), unpack it, and put `vox` on your `PATH`. Archives exist for Linux x86_64 and aarch64 (glibc 2.35+), macOS on Apple Silicon, and Windows x86_64. Each release carries a `SHA256SUMS` file and a build provenance attestation:

```bash
sha256sum --check SHA256SUMS --ignore-missing
gh attestation verify vox-v*.tar.gz --repo mrtozner/vox
```

Release binaries include the CLI, the server, Kokoro and Piper. Because Piper links espeak-ng, **the release binaries are GPL-3.0**; see [License](#license).

### From source

Vox is not on crates.io yet (the `vox` name there is an unrelated crate). Install from git; the package is `vox-voice`, the command it installs is `vox`. Needs Rust 1.88+, CMake and a C/C++ compiler (plus `libasound2-dev` and `libclang-dev` on Linux).

```bash
# Same feature set as the release binaries: CLI + server + Kokoro + Piper
cargo install --locked --git https://github.com/mrtozner/vox vox-voice --features dist
```

### First run

```bash
# Start the web server, downloading the default models it needs
vox serve --download
# Open http://localhost:3000

# Transcribe speech from your microphone
vox listen

# Text-to-speech with voice selection
vox speak "Hello from Vox!" --voice af_heart

# Voice chat with Ollama
vox chat --llm llama3.2

# Something not working? This explains which models are missing and why
vox doctor
```

`vox listen`, `vox speak` and `vox chat` offer to download a missing model; pass `-y` to skip the prompt.

<br>

## Web Interface

Start the server and open your browser:

```bash
vox serve --port 3000
```

| Tab | What it does |
|-----|-------------|
| **Live Talk** | The default tab. Experimental barge-in voice chat: full-duplex, so you can interrupt the LLM mid-sentence by speaking. Needs a running [Ollama](https://ollama.com). |
| **Listen** | Real-time mic transcription with optional speaker diarization *(experimental)*. Turn on "Identify speakers" to see who is talking, with color-coded transcript entries and a sidebar for renaming and forgetting speakers. |
| **Speak** | Text-to-speech synthesis. Pick a voice, type text, and hear it. |
| **Chat** | Voice chat with any Ollama LLM. Speak a question, get a spoken answer. |
| **Dashboard** | Server stats, loaded backends, model info, and environment capabilities. |

No separate frontend build. The UI is a single embedded HTML file served by the Rust binary.

The server has no authentication. It binds to `127.0.0.1` and refuses a non-loopback `--host` unless `VOX_ALLOW_REMOTE=1` is set; requests from other origins are rejected unless listed in `VOX_ALLOWED_ORIGINS`. `vox serve --help` lists the flags; the main environment variables are:

| Variable | Meaning |
|---|---|
| `VOX_MODELS_DIR` | Where models are read from and downloaded to |
| `VOX_LANGUAGE`, `VOX_STT_MODEL`, `VOX_PIPER_VOICE` | Defaults for `--language`, `--stt-model`, `--piper-voice` |
| `VOX_OLLAMA_HOST` | Ollama address (default `localhost:11434`) |
| `VOX_ALLOW_REMOTE` | `1` permits a non-loopback bind |
| `VOX_ALLOWED_HOSTS`, `VOX_ALLOWED_ORIGINS` | Comma-separated extra `Host` / `Origin` values to accept |
| `VOX_SPEAKER_DB` | Path of the speaker database |

<br>

## CLI

```bash
vox listen                                 # transcribe from microphone (Whisper)
vox listen --model base.en                 # use a larger Whisper model
vox listen --model small --language tr     # multilingual model, Turkish
vox listen --stt-backend distil-whisper --model small.en   # Distil-Whisper
vox listen --stt-backend sherpa            # Sherpa SenseVoice (multilingual)
vox listen --stt-backend sherpa-streaming  # real-time streaming transcription
vox speak "Hello from Vox!"                # text-to-speech (Kokoro)
vox speak "Hello" --voice af_heart         # Kokoro with voice selection
vox speak "Hallo" --backend piper --voice de   # Piper: en, de, es, fr, tr, zh
vox speak "Hi" --backend chatterbox --voice ref.wav  # voice cloning
vox speak "A long text." --stream          # start playing sentence by sentence
vox chat --llm llama3.2                    # voice chat with Ollama
vox serve --port 3000                      # start web + API server
vox serve --download --language tr         # fetch default models for Turkish first
vox doctor                                 # check models directory and installed models
vox test                                   # audio I/O diagnostics
vox benchmark                              # benchmark STT/TTS performance
vox config                                 # interactive setup wizard
vox models list                            # show available and downloaded models
vox models download kokoro                 # download a specific model
vox models path                            # show models directory
```

What a command can do depends on the features the binary was built with. `vox speak --backend` accepts `kokoro`, `piper` and `chatterbox`; the release binaries have the first two. Qwen3 and Pocket are available to the server and the Rust library, not to `vox speak`. `vox serve` exists only in builds with the `server` feature, `--stt-backend distil-whisper` needs `distil-whisper`, and the Sherpa backends need `sherpa`.

<br>

## API

### REST Endpoints

```bash
# Transcribe audio
curl -X POST http://localhost:3000/v1/transcribe \
  -H "Content-Type: audio/wav" \
  --data-binary @audio.wav

# Synthesize speech
curl -X POST http://localhost:3000/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Vox!", "voice": "af_heart"}'

# List available voices
curl http://localhost:3000/v1/voices

# Models, OpenAI shape ({"object": "list", "data": [...]})
curl http://localhost:3000/v1/models

# Models, Vox's own shape
curl http://localhost:3000/v1/vox/models

# OpenAI-compatible transcription (WAV upload) and speech (WAV or PCM out)
curl http://localhost:3000/v1/audio/transcriptions -F file=@audio.wav -F model=whisper-1
curl http://localhost:3000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "Hello from Vox!", "voice": "af_heart"}' -o speech.wav

# Known speakers (also PATCH/DELETE /v1/speakers/{id}, POST /v1/speakers/merge)
curl http://localhost:3000/v1/speakers

# List Ollama models
curl http://localhost:3000/v1/ollama-models

# Environment capabilities
curl http://localhost:3000/v1/capabilities

# Server stats
curl http://localhost:3000/v1/stats

# Liveness, and readiness (200 once STT, TTS and VAD are loaded, else 503)
curl http://localhost:3000/health
curl http://localhost:3000/health/ready
```

`/v1/models` returns the OpenAI list shape, so OpenAI SDKs can use a Vox server as their `base_url`. Vox's own model report, which `/v1/models` returned up to 0.6.0, is at `/v1/vox/models`.

### WebSocket Endpoints

**`/v1/listen`** &mdash; Real-time speech-to-text

Send PCM f32 LE frames at 16kHz mono. Receive JSON events:

```json
{"type": "speech_start"}
{"type": "partial", "text": "hello", "is_final": false}
{"type": "transcript", "text": "hello world", "duration_ms": 1200, "speaker": "speaker_1"}
{"type": "speech_end"}
```

When speaker identification is on (it needs the `speaker-encoder` model), transcripts include a `speaker` field.

**`/v1/speak`** &mdash; Streaming text-to-speech

Send JSON, receive chunked audio for gapless playback:

```json
{"text": "Hello world", "voice": "af_heart"}
```

**`/v1/converse`** &mdash; Continuous voice chat

Combines VAD + STT + LLM + TTS in a single WebSocket. Speak, get a spoken LLM response, repeat.

**`/v1/live-talk`** &mdash; Barge-in voice chat (experimental)

Full-duplex conversation. You can interrupt the LLM mid-response by speaking. Requires an Ollama instance. The server's `ready` message carries `protocol_version: 2` and a `session_id`; turn-scoped events carry a `turn_id`.

<br>

## Rust Library

The package is `vox-voice`; the library crate is `vox`. Until it is published, depend on it from git:

```toml
[dependencies]
vox = { package = "vox-voice", git = "https://github.com/mrtozner/vox" }
```

```rust
use vox::{Vox, SileroVad, WhisperBackend};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let vox = Vox::builder()
        .vad(SileroVad::new("silero_vad.onnx")?)
        .stt(WhisperBackend::from_model("ggml-tiny.en.bin")?)
        .on_utterance(|result, _ctx| {
            println!("{}", result.text);
        })
        .build()?;

    vox.listen().await?;
    Ok(())
}
```

<br>

## Python Bindings

The bindings (Whisper STT, Silero VAD, Kokoro TTS) are built from this repository with [maturin](https://www.maturin.rs); there is no package on PyPI. Inside a virtualenv:

```bash
pip install maturin
cd python
maturin develop --release
```

```python
from vox_voice import Vox, SileroVad, WhisperStt

vox = Vox(vad=SileroVad(), stt=WhisperStt("tiny.en"))
for result in vox.listen():
    print(result.text)
```

<br>

## Architecture

```
+--------+     +-----+     +-----+     +------------+     +-----+     +---------+
|  Mic   | --> | VAD | --> | STT | --> | Speaker ID | --> |  CB | --> |   TTS   |
| (cpal) |     |     |     |     |     | (optional) |     |     |     |         |
+--------+     +-----+     +-----+     +------------+     +-----+     +---------+
                  |                         |                |
            Silero ONNX            WeSpeaker ResNet34   Your callback
            v5 model                voice embeddings     gets text +
                                    + SQLite DB          VoxContext
```

Audio is captured via `cpal`, resampled to 16kHz mono, and fed frame-by-frame to VAD. On speech end, the utterance goes to STT. If diarization is enabled, the audio is also processed by a speaker encoder that extracts a voice embedding (WeSpeaker ResNet34; the dimension is read from the model), compares it against known speakers, and assigns an identity. Your callback receives the text, speaker label, and a `VoxContext` for optional TTS reply.

<br>

## Models

`vox models list` prints every model Vox can download, with its exact size and whether it is installed. Downloads are checked against a pinned size and SHA-256.

| Component | Model (`vox models download <name>`) | Size | Notes |
|-----------|-------|------|-------|
| **VAD** | `silero-vad` | 2 MB | Speech activity detection |
| **STT** | `whisper-tiny.en` | 74 MB | Fast, English |
| | `whisper-base.en` | 141 MB | Better accuracy, English |
| | `whisper-small.en` | 465 MB | High accuracy, English |
| | `whisper-medium.en` | 1.4 GB | Highest accuracy, English |
| | `whisper-tiny`, `-base`, `-small`, `-medium` | 74 MB &ndash; 1.4 GB | Multilingual (use with `--language`) |
| | `whisper-large-v3-turbo-q5` | 547 MB | Multilingual, quantized Turbo |
| | `whisper-tiny.en-int8`, `whisper-base.en-int8` | 42 MB, 78 MB | Quantized, English |
| | `distil-whisper-small.en`, `-medium.en`, `-large-v3` | 321 MB &ndash; 1.4 GB | Distil-Whisper (`distil-whisper` feature) |
| | `sherpa-sensevoice` (+ `-tokens`) | 228 MB | Multilingual (zh/en/ja/ko/yue), `sherpa` feature |
| | `sherpa-streaming-*` (4 files) | 69 MB | Real-time partial results, `sherpa` feature |
| **TTS** | `kokoro` + `kokoro-voices` | 311 MB + 27 MB | 57 voices, 9 languages, 24 kHz |
| | `kokoro-int8` | 88 MB | Quantized Kokoro |
| | `piper-en-us`, `-de`, `-es`, `-fr`, `-tr`, `-zh` (+ `-config`) | 60 MB per voice | Fast CPU synthesis |
| **Diarization** | `speaker-encoder` | 25 MB | WeSpeaker ResNet34 (VoxCeleb) speaker embeddings |

Qwen3, Pocket and Chatterbox are not in `vox models`: those backends fetch their own weights the first time they are created. For Qwen3, `VOX_QWEN3_MODEL_PATH` points at a local copy.

```bash
vox models download silero-vad
vox models download whisper-base.en
vox models download kokoro
vox models download kokoro-voices
vox models download piper-en-us
vox models download piper-en-us-config
```

`vox serve --download` fetches the default set for your machine and language in one step.

### Model Storage

The models directory is resolved in this order:

1. The `VOX_MODELS_DIR` environment variable.
2. A `models/` directory next to Vox's `Cargo.toml`, when running from a source checkout.
3. The per-user data directory:
   - **macOS**: `~/Library/Application Support/vox/models`
   - **Linux**: `$XDG_DATA_HOME/vox/models`, or `~/.local/share/vox/models`
   - **Windows**: `%LOCALAPPDATA%\vox\models`

Run `vox models path` to see yours, and `vox doctor` to see why it was chosen and what is missing. Use `vox models download <name> --force` to re-download a corrupted file.

The speaker database is `speakers.db` in the same per-user `vox` data directory (override with `VOX_SPEAKER_DB`).

<br>

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `dist` | no | What the release binaries are built with: `cli`, `server`, `kokoro`, `piper`. Produces a GPL-3.0 binary |
| `cli` | no | CLI binary (`vox listen`, `vox speak`, `vox chat`, `vox doctor`, `vox models`) |
| `server` | no | HTTP/WebSocket server with web UI (includes `diarization`) |
| `whisper` | **yes** | Whisper STT via whisper-rs |
| `distil-whisper` | no | Distil-Whisper STT (6x faster) |
| `silero` | **yes** | Silero VAD via ONNX Runtime |
| `sherpa` | no | Sherpa-ONNX STT (SenseVoice, Zipformer, streaming) |
| `kokoro` | no | Kokoro TTS (57 voices, 9 languages) |
| `qwen3` | no | Qwen3 TTS (state-of-the-art, streaming) |
| `qwen3-metal` | no | Qwen3 with Apple Metal GPU |
| `qwen3-cuda` | no | Qwen3 with NVIDIA CUDA GPU |
| `piper` | no | Piper TTS (multilingual, fast CPU). Links espeak-ng (GPL-3.0): see [License](#license) |
| `pocket` | no | Pocket TTS (pure Rust, edge-ready) |
| `pocket-metal` | no | Pocket with Apple Metal GPU |
| `chatterbox` | no | Chatterbox TTS (voice cloning) |
| `chatterbox-coreml` | no | Chatterbox with CoreML (macOS) |
| `diarization` | no | Speaker identification (auto-enabled by `server`) |
| `tts` | no | Audio playback for TTS output |

### Common Feature Combinations

```bash
# The release feature set: CLI + server + Kokoro + Piper (GPL-3.0 binary)
cargo install --locked --git https://github.com/mrtozner/vox vox-voice --features dist

# Minimal: just transcription
cargo install --locked --git https://github.com/mrtozner/vox vox-voice --features cli

# Server without Piper (no espeak-ng in the binary)
cargo install --locked --git https://github.com/mrtozner/vox vox-voice --features cli,server,kokoro

# Server with Qwen3 on Apple Silicon
cargo install --locked --git https://github.com/mrtozner/vox vox-voice --features cli,server,qwen3-metal

# Server with Sherpa streaming STT + Kokoro TTS
cargo install --locked --git https://github.com/mrtozner/vox vox-voice --features cli,server,sherpa,kokoro
```

<br>

## Performance

Measured on Apple M1 MacBook Pro:

| Metric | Value |
|--------|-------|
| VAD frame latency | ~1 ms per 32 ms frame |
| Whisper STT (3 s utterance) | ~200 ms |
| Streaming STT (per chunk) | <1 ms (0.03x real-time) |
| End-to-end (speech end to text) | ~250 ms |
| Kokoro TTS ("Hello world") | ~300 ms |
| Piper TTS ("Hello world") | ~200 ms |
| Qwen3 0.6B TTS (streaming, M4 Pro) | 0.96 RTF |
| Qwen3 1.7B TTS (streaming, M4 Pro) | 1.14 RTF |
| Memory (idle pipeline) | ~150 MB |
| Memory (Kokoro loaded) | ~400 MB |
| Memory (Qwen3 0.6B loaded) | ~2.6 GB |

<br>

## Raspberry Pi

The `aarch64-unknown-linux-gnu` release archive is built for 64-bit ARM Linux with glibc 2.35+, which covers a Raspberry Pi 4 or 5 on Raspberry Pi OS Bookworm (64-bit). See [docs/raspberry_pi.md](docs/raspberry_pi.md) for installation, model choices for small machines, and building on the Pi itself.

<br>

## Examples

```bash
cargo run --example simple_listen --features whisper,silero
cargo run --example voice_assistant --features whisper,silero,kokoro
cargo run --example diarization_demo --features whisper,silero,diarization
cargo run --example live_diarization --features whisper,silero,diarization
cargo run --example tts_speak --features kokoro
cargo run --example piper_speak --features piper
cargo run --example pocket_speak --features pocket
cargo run --example chatterbox_speak --features chatterbox
cargo run --example test_streaming --features qwen3
```

Download the models an example needs first (`vox models download ...`).

<br>

## Platform Support

| Platform | Release binary | Status |
|----------|----------------|--------|
| macOS (Apple Silicon) | yes | CI tested |
| Linux x86_64 (glibc 2.35+) | yes | CI tested |
| Linux aarch64 (glibc 2.35+, Raspberry Pi 4/5) | yes | Built and smoke-tested by the release workflow |
| Windows x86_64 | yes | CI tested |
| macOS (Intel) | no | Not built or tested. The ONNX Runtime bindings Vox uses (`ort` 2.0.0-rc.11) ship no prebuilt library for Intel macOS, so a build needs your own ONNX Runtime. |

<br>

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes and add tests
4. Run `cargo test` and `cargo clippy`
5. Submit a pull request

<br>

## License

**Source code:** MIT OR Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)).

**Release binaries:** GPL-3.0 ([LICENSE-GPL-3.0](LICENSE-GPL-3.0)). The prebuilt `vox` binaries include Piper, which links [espeak-ng](https://github.com/espeak-ng/espeak-ng) (GPL-3.0-or-later), so each binary as a whole is distributed under GPL-3.0. The same applies to any binary you build with the `piper` or `dist` feature and pass on to someone else. A build without `piper` contains no espeak-ng.

Models are downloaded separately and carry their publishers' licenses. [NOTICE](NOTICE) has the details.
