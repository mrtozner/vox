<p align="center">
  <img src="assets/vox-logo.jpg" alt="Vox" width="200">
  <p align="center"><strong>The offline voice runtime in Rust. Talk to a local LLM, interrupt it mid-sentence, and let it remember who it is talking to &mdash; all on your machine.</strong></p>
</p>

<p align="center">
  <a href="https://github.com/mrtozner/vox/actions/workflows/ci.yml"><img src="https://github.com/mrtozner/vox/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/mrtozner/vox/releases/latest"><img src="https://img.shields.io/github/v/release/mrtozner/vox" alt="Latest release"></a>
  <a href="#license"><img src="https://img.shields.io/badge/source-MIT%2FApache--2.0-blue.svg" alt="Source license: MIT or Apache-2.0"></a>
  <a href="#license"><img src="https://img.shields.io/badge/release%20binaries-GPL--3.0-blue.svg" alt="Release binaries: GPL-3.0"></a>
</p>

---

Vox is a voice AI runtime that ships as one binary and as a Rust library: voice activity detection, speech-to-text, speaker identification, text-to-speech, and full-duplex voice chat with a local LLM through [Ollama](https://ollama.com). It recognizes the people it has been introduced to and, if they opt in, remembers what they said across sessions. No API keys, no cloud APIs, and an `--offline` switch that refuses a remote LLM host and every download.

```
Mic --> VAD (Silero) --> STT (Whisper/Sherpa) --> Speaker ID --> LLM (Ollama) or your code --> TTS (Kokoro/Piper/Qwen3) --> Speaker
                                                      |
                                              per-person memory (SQLite, local)
```

<br>

## Why Vox

Most voice-agent frameworks are Python orchestration layers built around cloud speech and LLM APIs. Vox takes the other route:

- **One binary, no Python.** `vox serve` is the whole stack: speech models run in-process, the web UI is embedded, there is no separate frontend or worker to deploy. Two things live outside the binary: the model files, which are downloaded once, and the LLM, which is an Ollama server you run.
- **Offline you can check.** In the release feature set Vox makes two kinds of outbound connection: requests to Ollama and model downloads. `--offline` refuses an Ollama host that is not loopback and refuses every download. `GET /v1/privacy/ledger` shows the connections the server process made, with destination, purpose and body bytes. (The optional `pocket`, `chatterbox` and `qwen3` backends fetch their own weights and are outside this control.)
- **It knows who is speaking.** Speaker identification runs inside the conversation loop. No voiceprint is stored before you record consent, and in Live Talk a voice is enrolled only by an explicit command. Long-term memory is opt-in per person and deleted with the speaker.
- **Embeddable.** The engine and the HTTP/WebSocket server are library APIs (`vox::server`, `VoxBuilder::build_with_channel`), so you can feed it your own audio and mount its router in your own application.
- **OpenAI-shaped where it helps.** `/v1/audio/transcriptions`, `/v1/audio/speech` and `/v1/models` follow the OpenAI shape, so OpenAI SDK clients can use a Vox server for transcription and speech.

What Vox is not: it has no telephony or WebRTC transport, no cloud-provider plugins, and no built-in authentication (it binds to loopback by default). If you need a hosted, multi-provider agent platform, a framework like Pipecat or LiveKit Agents is the better fit. Live Talk and speaker identification are still marked experimental.

<br>

## Highlights

- **Live Talk** *(experimental)* &mdash; Full-duplex voice chat with barge-in in every phase of a turn. While a reply is being transcribed or generated, a cough or a click does not cancel it: speech has to last 250 ms. The conversation records only the sentences you actually heard, sessions survive a reconnect, and every turn reports latency metrics
- **Speaker-aware conversations** *(experimental)* &mdash; Known speakers are identified while they talk and named to the model. Voiceprints are stored only after you give consent. In Live Talk, talking never enrolls or changes a voiceprint; enrollment is an explicit command
- **Per-person long-term memory** &mdash; Opt-in for each speaker, stored in the local SQLite speaker database, recalled at the start of their next session. Turns that fall out of the context window are distilled into facts by the conversation's Ollama model. Forgetting a speaker deletes their memory
- **Offline mode and privacy ledger** &mdash; `--offline` (or `VOX_OFFLINE=1`) blocks remote Ollama hosts and model downloads; `/v1/privacy/ledger` shows the server's outbound connections
- **Speech-to-Text** &mdash; Whisper (including multilingual and large-v3-turbo models), Distil-Whisper, Sherpa-ONNX (streaming + multilingual)
- **Text-to-Speech** &mdash; Kokoro (57 voices, 9 languages), Piper (multilingual), Qwen3, Pocket (pure Rust), Chatterbox (voice cloning)
- **`vox tune`** &mdash; Measures the STT, TTS and Ollama models installed on your machine with a spoken sample and saves the best combination as the defaults of `vox chat`
- **HTTP + WebSocket API** &mdash; OpenAI-compatible `/v1/audio/*` and `/v1/models`, a speaker and memory REST API, and 4 WebSocket channels for real-time streaming
- **Web Interface** &mdash; 5-tab browser UI served by the binary: Listen, Speak, Chat, Live Talk, Dashboard. It opens on Live Talk
- **Verified downloads and releases** &mdash; Model downloads are pinned to a revision and checked by size and SHA-256; release archives carry `SHA256SUMS` and build provenance attestations
- **Pluggable Backends** &mdash; Swap any component via Rust traits

<br>

## Quick Start

### Prebuilt binary

Download the archive for your platform from the [latest release](https://github.com/mrtozner/vox/releases/latest), unpack it, and put `vox` on your `PATH`. Archives exist for Linux x86_64 and aarch64 (glibc 2.39+: Ubuntu 24.04, Debian 13, Raspberry Pi OS Trixie), macOS 13.4+ on Apple Silicon, and Windows x86_64 (without Piper: espeak-ng does not build with MSVC, so Windows uses Kokoro for every voice). x86_64 builds need an AVX2-capable CPU (x86-64-v3: Intel Haswell / AMD Zen or newer). Each release carries a `SHA256SUMS` file and a build provenance attestation:

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
# Open http://localhost:3000 (Chat and Live Talk need a running Ollama)

# Once the models are there: no downloads, loopback Ollama only
vox serve --offline

# Transcribe speech from your microphone
vox listen

# Text-to-speech with voice selection
vox speak "Hello from Vox!" --voice af_heart

# Voice chat with Ollama
vox chat --llm llama3.2

# Measure the installed models and save the best ones as vox chat defaults
vox tune

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
| `VOX_OFFLINE` | `1` is the same as `--offline`: no remote Ollama, no model downloads |
| `VOX_MEMORY_DISTILL` | `0` stops distilling trimmed Live Talk turns into long-term facts (on by default) |
| `VOX_SESSION_TTL_SECS` | How long an idle Live Talk session is kept (default 1800) |

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
vox serve --offline                        # loopback Ollama only, no downloads
vox tune                                   # measure installed models, save the best for vox chat
vox tune --dry-run                         # recommend only: loads, connects and writes nothing
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

# Long-term memory of one speaker (GET reads, PUT {"enabled": true} switches it on, DELETE forgets)
curl http://localhost:3000/v1/speakers/speaker_1/memory

# Outbound connections this server process made (destination, purpose, body bytes)
curl http://localhost:3000/v1/privacy/ledger

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

`/v1/models` returns the OpenAI list shape, so OpenAI SDKs can use a Vox server as their `base_url` for transcription and speech. Vox's own model report, which `/v1/models` returned up to 0.6.0, is at `/v1/vox/models`. Chat completions are not provided.

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

Full-duplex conversation. You can interrupt the LLM mid-response by speaking. Requires an Ollama instance. The server's `ready` message carries `protocol_version: 2` and a `session_id`; turn-scoped events carry a `turn_id`. With `diarization` on and the `speaker-encoder` model installed, known speakers are named in `transcript`; `enroll_speaker` stores the voice of the newest turn (consent required), and `memory_distilled` reports facts saved to long-term memory.

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

### Your own audio source

`build()` opens the default microphone. To run the same pipeline on audio you already have (a file, a socket, a call), feed it through a channel instead. This path needs neither the `mic` nor the `signal` feature, so an embedding application can drop the audio driver and decide for itself when to stop:

```toml
[dependencies]
vox = { package = "vox-voice", git = "https://github.com/mrtozner/vox", default-features = false, features = ["whisper", "silero"] }
```

```rust
use vox::{AudioChunk, SileroVad, Vox, WhisperBackend};

let (vox, audio) = Vox::builder()
    .vad(SileroVad::new("silero_vad.onnx")?)
    .stt(WhisperBackend::from_model("ggml-tiny.en.bin")?)
    .on_utterance(|result, _ctx| println!("{}", result.text))
    .build_with_channel(48_000, 32)?; // source sample rate, channel capacity

// Chunks of any length and channel count; drop `audio` to end the stream.
audio.send(AudioChunk { samples, sample_rate: 48_000, channels: 2 }).await?;

// Runs until the sender is dropped or the future you pass resolves.
vox.listen_until(shutdown.cancelled()).await?;
```

### Embedding the server

With the `server` feature the HTTP/WebSocket server is part of the library as `vox::server`: `run_with_config` is what `vox serve` calls, `serve` runs a `ServerState` you built on a listener you bound, and `build_router` returns the `axum::Router` to merge into your own application.

```rust
use std::{sync::Arc, time::Duration};
use vox::server::{self, ServerOptions, ServerState};

let state = Arc::new(ServerState {
    stt: Some(my_stt), // any Arc<dyn vox::SttBackend>
    ..ServerState::with_options("127.0.0.1", 3000, &ServerOptions::default())
});
let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
server::serve(listener, state, Duration::from_secs(5), my_shutdown_future).await?;
```

The server has no authentication; read the notes on `vox::server::check_bind_host` before binding anything other than loopback.

<br>

## Python Bindings

The bindings (Whisper STT, Silero VAD, Kokoro TTS) are built from this repository with [maturin](https://www.maturin.rs); there is no package on PyPI. A built wheel includes Kokoro, whose phonemizer is derived from espeak-ng, so treat the wheel as GPL-3.0 if you redistribute it. Inside a virtualenv:

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
| `cli` | no | CLI binary (`vox listen`, `vox speak`, `vox chat`, `vox tune`, `vox doctor`, `vox models`) |
| `server` | no | HTTP/WebSocket server with web UI, as `vox::server` (includes `diarization` and `silero`) |
| `mic` | **yes** | Microphone capture via cpal (`Vox::builder().build()`, `AudioCapture`). Off: feed the engine with `build_with_source` / `build_with_channel` |
| `signal` | **yes** | `Vox::listen()` stops on Ctrl-C. Off: use `Vox::listen_until()` |
| `whisper` | **yes** | Whisper STT via whisper-rs |
| `distil-whisper` | no | Distil-Whisper STT (6x faster) |
| `silero` | **yes** | Silero VAD via ONNX Runtime |
| `sherpa` | no | Sherpa-ONNX STT (SenseVoice, Zipformer, streaming) |
| `kokoro` | no | Kokoro TTS (57 voices, 9 languages). Bundles espeak-ng-derived phonemizer data (GPL-3.0): see [License](#license) |
| `qwen3` | no | Qwen3 TTS (state-of-the-art, streaming) |
| `qwen3-metal` | no | Qwen3 with Apple Metal GPU |
| `qwen3-cuda` | no | Qwen3 with NVIDIA CUDA GPU |
| `piper` | no | Piper TTS (multilingual, fast CPU). Links espeak-ng (GPL-3.0): see [License](#license) |
| `pocket` | no | Pocket TTS (pure Rust, edge-ready) |
| `pocket-metal` | no | Pocket with Apple Metal GPU |
| `pocket-mkl` | no | Pocket with Intel MKL |
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

# Server without Piper
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

The `aarch64-unknown-linux-gnu` release archive is built for 64-bit ARM Linux with glibc 2.39+, which covers a Raspberry Pi 4 or 5 on Raspberry Pi OS Trixie (64-bit). See [docs/raspberry_pi.md](docs/raspberry_pi.md) for installation, model choices for small machines, and building on the Pi itself.

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
| macOS 13.4+ (Apple Silicon) | yes | CI tested |
| Linux x86_64 (glibc 2.39+, x86-64-v3 CPU) | yes | CI tested |
| Linux aarch64 (glibc 2.39+, Raspberry Pi 4/5) | yes | Built and smoke-tested by the release workflow |
| Windows x86_64 (x86-64-v3 CPU) | yes, without Piper | CI tested (no Piper) |
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

**Release binaries:** GPL-3.0 ([LICENSE-GPL-3.0](LICENSE-GPL-3.0)). The prebuilt `vox` binaries include Piper, which links [espeak-ng](https://github.com/espeak-ng/espeak-ng) (GPL-3.0-or-later), so each binary as a whole is distributed under GPL-3.0. The same applies to any binary you build with the `piper`, `kokoro` or `dist` feature and pass on to someone else. Kokoro's phonemizer is derived from espeak-ng as well. A build without `piper` and `kokoro` contains no espeak-ng material.

Models are downloaded separately and carry their publishers' licenses. [NOTICE](NOTICE) has the details.
