<p align="center">
  <h1 align="center">Vox</h1>
  <p align="center"><strong>Local-first voice AI framework. Speech-to-text, text-to-speech, and voice chat.</strong></p>
</p>

<p align="center">
  <a href="https://github.com/mrtozner/vox/actions/workflows/ci.yml"><img src="https://github.com/mrtozner/vox/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/vox"><img src="https://img.shields.io/crates/v/vox.svg" alt="crates.io"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg" alt="License"></a>
</p>

---

Speech-to-text, text-to-speech, and voice chat running locally. No API keys, no cloud, no data leaving your machine.

```
Mic --> VAD (Silero) --> STT (Whisper) --> Your Code --> TTS (Kokoro) --> Speaker
```

<br>

## Quick Start

```bash
# Install
cargo install --git https://github.com/mrtozner/vox --features cli

# Transcribe speech from your microphone
vox listen

# Text-to-speech (requires kokoro feature)
cargo install --git https://github.com/mrtozner/vox --features cli,kokoro
vox speak "Hello from Vox!"

# Voice chat with Ollama
vox chat --llm llama3.2
```

Models auto-download on first run. Pass `-y` to skip prompts.

<br>

## What It Does

- **Speech-to-Text** &mdash; Microphone transcription with Whisper (tiny to medium, English or multilingual)
- **Text-to-Speech** &mdash; Natural synthesis with Kokoro (50+ voices), Pocket (pure Rust, edge-ready), or Chatterbox (voice cloning)
- **Voice Chat** &mdash; Talk to any Ollama LLM and hear responses
- **Web Interface** &mdash; Browser UI for demos and testing (`vox serve`)
- **Python &amp; Rust APIs** &mdash; Use from Python via pip or Rust via cargo
- **HTTP/WebSocket Server** &mdash; Integrate into any stack with REST or streaming WebSocket API
- **Fully Local** &mdash; No API keys, no cloud, no data leaves your machine
- **Pluggable Backends** &mdash; Swap VAD, STT, or TTS engines via traits

<br>

## Usage

### CLI

```bash
vox listen                              # transcribe from microphone
vox listen --model base.en              # use a larger Whisper model
vox speak "Hello from Vox!"             # text-to-speech (needs kokoro feature)
vox speak "Hello" --voice am_adam       # pick a voice
vox chat --llm llama3.2                 # voice chat with Ollama
vox models list                         # show downloaded models
vox models download whisper-base.en     # download a specific model
```

### Web UI

```bash
cargo install --git https://github.com/mrtozner/vox --features cli,server,kokoro
vox serve --port 3000
```

Opens a browser interface at `http://localhost:3000` with real-time mic transcription, TTS synthesis, voice chat with Ollama, and a status dashboard. No separate frontend build.

### HTTP API

Use the same server's REST endpoints directly:

```bash
# Transcribe audio
curl -X POST http://localhost:3000/v1/transcribe \
  -H "Content-Type: audio/wav" \
  --data-binary @audio.wav

# Synthesize speech
curl -X POST http://localhost:3000/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Vox!"}'
```

WebSocket streaming at `ws://localhost:3000/v1/listen` &mdash; send PCM f32 LE frames at 16kHz mono, receive JSON:

```json
{"type": "speech_start"}
{"type": "transcript", "text": "hello world", "duration_ms": 1200, "processing_time_ms": 180}
```

### Rust Library

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

### Python Library

```bash
pip install vox-voice
```

```python
from vox_voice import Vox, SileroVad, WhisperStt

vox = Vox(vad=SileroVad(), stt=WhisperStt("tiny.en"))
for result in vox.listen():
    print(result.text)
```

Built with PyO3 and maturin. Same pipeline, Pythonic API. Build from source in the `python/` directory.

<br>

## Architecture

```
+--------+     +-----+     +-----+     +-----------+     +-----+
|  Mic   | --> | VAD | --> | STT | --> | Callback  | --> | TTS |
| (cpal) |     |     |     |     |     | (your fn) |     |     |
+--------+     +-----+     +-----+     +-----------+     +-----+
                  |                          |
            Silero ONNX               VoxContext gives
            v5 model                  access to speak()
```

Audio captured via `cpal`, resampled to 16kHz mono, fed frame-by-frame to VAD. On speech end, the utterance goes to STT. Your callback gets the text and a `VoxContext` for optional TTS reply.

<br>

## Models

| Component | Model | Size | Notes |
|-----------|-------|------|-------|
| **VAD** | Silero VAD v5 | 2MB | Speech detection |
| **STT** | Whisper tiny.en | 75MB | Fast, English |
| | Whisper base.en | 142MB | Better accuracy |
| | Whisper small.en | 466MB | High accuracy |
| | Whisper medium.en | 1.5GB | Highest accuracy |
| **TTS** | Kokoro | 310MB | 50+ voices |
| | Pocket | 82MB | Pure Rust, edge/embedded |
| | Chatterbox | 350MB | Voice cloning |

```bash
vox models download silero-vad          # 2MB
vox models download whisper-tiny.en     # 75MB
vox models download kokoro              # 310MB
vox models download kokoro-voices       # 27MB
```

<br>

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `cli` | no | CLI binary (`vox listen`, `vox speak`, `vox chat`, `vox serve`) |
| `server` | no | HTTP/WebSocket API server |
| `whisper` | yes | Whisper STT via whisper-rs |
| `silero` | yes | Silero VAD via ONNX Runtime |
| `kokoro` | no | Kokoro TTS (50+ voices) |
| `pocket` | no | Pocket TTS (pure Rust) |
| `pocket-metal` | no | Pocket TTS with Apple Metal GPU |
| `chatterbox` | no | Chatterbox TTS (voice cloning) |
| `chatterbox-coreml` | no | Chatterbox with CoreML (macOS) |
| `tts` | no | Audio playback for TTS output |

<br>

## Platform Support

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon) | Tested |
| macOS (Intel) | Tested |
| Linux (x86_64) | CI tested |
| Windows (x86_64) | CI tested |

<br>

## Performance

Measured on Apple M1 MacBook Pro with Whisper `tiny.en`:

| Metric | Value |
|--------|-------|
| VAD frame latency | ~1ms per 32ms frame |
| STT latency (3s utterance) | ~200ms |
| End-to-end (speech end to text) | ~250ms |
| Memory (idle pipeline) | ~150MB |

<br>

## Examples

```bash
cargo run --example simple_listen --features whisper,silero       # mic to text
cargo run --example vad_only --features silero                    # speech detection only
cargo run --example voice_assistant --features whisper,silero,kokoro  # LLM voice assistant
cargo run --example tts_speak --features kokoro                   # text to speech
```

<br>

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes and add tests
4. Run `cargo test` and `cargo clippy`
5. Submit a pull request

For larger features, open an issue first to discuss the approach.

<br>

## License

MIT OR Apache-2.0
