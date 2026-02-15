<p align="center">
  <h1 align="center">Vox</h1>
  <p align="center"><strong>Local voice AI framework. Ollama for voice.</strong></p>
</p>

<p align="center">
  <a href="https://github.com/mrtozner/vox/actions/workflows/ci.yml"><img src="https://github.com/mrtozner/vox/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/vox"><img src="https://img.shields.io/crates/v/vox.svg" alt="crates.io"></a>
  <a href="https://pypi.org/project/vox-voice/"><img src="https://img.shields.io/pypi/v/vox-voice.svg" alt="PyPI"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <img src="demo.gif" alt="Vox demo" width="600">
  <br>
  <sub><a href="demo.tape">View demo source</a> &mdash; recorded with <a href="https://github.com/charmbracelet/vhs">VHS</a></sub>
</p>

---

Speech-to-text, text-to-speech, and voice chat &mdash; all running locally on your hardware. No API keys, no cloud, no data leaving your machine.

```
Mic --> VAD (Silero) --> STT (Whisper) --> Your Code --> TTS (Kokoro) --> Speaker
```

<br>

|  | Cloud APIs | Vox |
|--|-----------|-----|
| **Privacy** | Data sent to third-party servers | 100% local, nothing leaves your device |
| **Cost** | Per-request pricing | Free and open source |
| **Latency** | Network round-trip | Sub-250ms end-to-end |
| **Offline** | Requires internet | Works anywhere |

<br>

## Quick Start

### Homebrew (macOS)

```bash
brew tap mrtozner/tap
brew install vox
```

### Cargo

```bash
cargo install vox --features cli
```

### Python

```bash
pip install vox-voice
```

Then get started:

```bash
vox listen                              # real-time mic transcription
vox speak "Hello from Vox!"             # text-to-speech
vox chat --llm llama3.2                 # voice chat with an LLM
```

Models auto-download on first run. Add `--yes` to skip the confirmation prompt.

<br>

## Features

**Speech-to-Text** &mdash; Real-time transcription from your microphone using Whisper (tiny to large, English or multilingual).

**Text-to-Speech** &mdash; Near-studio quality synthesis with Kokoro (82M), Pocket (pure Rust), or Chatterbox (voice cloning).

**Voice Chat** &mdash; Full voice conversations with any Ollama model. Speak, get a response, hear it spoken back.

**WebSocket Streaming** &mdash; Real-time transcription over WebSocket for web and mobile apps.

**Use It Your Way** &mdash; CLI, Python library, HTTP API, or embed as a Rust crate.

**Pluggable Backends** &mdash; Swap VAD, STT, or TTS engines by implementing a trait. Bring your own models.

<br>

## Usage

### CLI

```bash
# Transcribe speech from your microphone
vox listen

# Use a larger model for better accuracy
vox listen --model base.en

# Text-to-speech
cargo install vox --features cli,kokoro
vox speak "Hello from Vox!" --voice af_heart

# Voice chat with Ollama
vox chat --llm llama3.2

# Manage models
vox models list
vox models download whisper-base.en
```

### Python

```python
from vox_voice import Vox, SileroVad, WhisperStt

vox = Vox(vad=SileroVad(), stt=WhisperStt("tiny.en"))
for result in vox.listen():
    print(result.text)
```

```python
from vox_voice import KokoroTts

tts = KokoroTts()
audio = tts.synthesize("Hello from Vox!")
audio.save("output.wav")
```

### HTTP API

```bash
vox serve --port 3000
```

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

**WebSocket** &mdash; connect to `ws://localhost:3000/v1/listen`, send raw PCM f32 LE frames at 16kHz mono, receive JSON events:

```json
{"type": "speech_start"}
{"type": "transcript", "text": "hello world", "duration_ms": 1200, "processing_time_ms": 180}
{"type": "speech_end"}
```

### Rust Library

```toml
[dependencies]
vox = "0.2"
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
| **TTS** | Kokoro | 88MB (int8) | Near-studio quality, 50+ voices |
| | Pocket | 82MB | Pure Rust, edge/embedded |
| | Chatterbox | 350MB | Voice cloning |

```bash
vox models download silero-vad          # ~2MB
vox models download whisper-tiny.en     # ~75MB
vox models download kokoro              # ~88MB
```

<br>

## Backends

| Component | Default | Alternatives |
|-----------|---------|-------------|
| VAD | Silero VAD v5 | Implement `VadBackend` trait |
| STT | Whisper (via whisper.cpp) | Implement `SttBackend` trait |
| TTS | Kokoro (82M) | Pocket, Chatterbox, or implement `TtsBackend` trait |

<details>
<summary>Custom backend example</summary>

```rust
use async_trait::async_trait;
use vox::{VadBackend, VadEvent, AudioChunk, VoxError};

struct MyVad { /* ... */ }

#[async_trait]
impl VadBackend for MyVad {
    async fn process_frame(&mut self, frame: &AudioChunk) -> Result<Vec<VadEvent>, VoxError> {
        Ok(vec![VadEvent::Silence])
    }

    fn reset(&mut self) { /* ... */ }
    fn frame_size(&self) -> usize { 512 }
    fn sample_rate(&self) -> u32 { 16000 }
}
```

The same pattern applies to `SttBackend` and `TtsBackend`.

</details>

<br>

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `cli` | no | CLI binary (`vox listen`, `vox speak`, `vox chat`, `vox serve`) |
| `server` | no | HTTP/WebSocket API server |
| `whisper` | yes | Whisper STT via whisper-rs |
| `silero` | yes | Silero VAD via ONNX Runtime |
| `kokoro` | no | Kokoro TTS (82M, near-studio quality) |
| `pocket` | no | Pocket TTS (pure Rust, edge/embedded) |
| `pocket-metal` | no | Pocket TTS with Apple Metal GPU |
| `chatterbox` | no | Chatterbox Turbo TTS (350M, voice cloning) |
| `chatterbox-coreml` | no | Chatterbox with CoreML (macOS) |
| `tts` | no | Audio playback for TTS output |

<br>

## Platform Support

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon) | Tested |
| Linux (x86_64) | Tested |
| Raspberry Pi 5 | Supported |
| NVIDIA Jetson | Supported |

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
cargo run --example simple_listen                  # mic to text
cargo run --example vad_only                       # speech detection only
cargo run --example voice_assistant                # LLM voice assistant
cargo run --example tts_speak --features kokoro    # text to speech
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
