<p align="center">
  <img src="assets/vox-logo.jpg" alt="Vox" width="200">
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
Mic --> VAD (Silero) --> STT (Whisper/Sherpa/Streaming) --> Your Code --> TTS (Kokoro/Piper/Chatterbox) --> Speaker
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

- **Speech-to-Text** &mdash; Whisper (tiny to medium), Sherpa-ONNX (SenseVoice, Zipformer, Paraformer), or streaming Sherpa for real-time partial transcription
- **Text-to-Speech** &mdash; Natural synthesis with Kokoro (50+ voices), Piper (multilingual), Pocket (pure Rust, edge-ready), or Chatterbox (voice cloning)
- **Voice Chat** &mdash; Talk to any Ollama LLM and hear responses
- **Web Interface** &mdash; Browser UI for demos and testing (`vox serve`)
- **Python Bindings** &mdash; Same pipeline from Python via PyO3
- **HTTP/WebSocket Server** &mdash; Integrate into any stack with REST or streaming WebSocket API
- **Fully Local** &mdash; No API keys, no cloud, no data leaves your machine
- **Pluggable Backends** &mdash; Swap VAD, STT, or TTS engines via traits

<br>

## Usage

### CLI

```bash
vox listen                              # transcribe from microphone (Whisper)
vox listen --model base.en              # use a larger Whisper model
vox listen --stt-backend sherpa         # use Sherpa SenseVoice (multilingual)
vox listen --stt-backend sherpa-streaming  # real-time streaming transcription
vox listen --stt-backend distil-whisper # use Distil-Whisper (6x faster)
vox speak "Hello from Vox!"             # text-to-speech (needs kokoro feature)
vox speak "Hello" --voice am_adam       # pick a voice
vox speak "Hallo" --backend piper --voice de  # multilingual TTS with Piper
vox speak "Hi" --backend pocket          # pure Rust TTS (edge-ready, no external deps)
vox speak "Hi" --backend chatterbox --voice ref.wav  # voice cloning
vox chat --llm llama3.2                 # voice chat with Ollama
vox test                                # run audio I/O diagnostics
vox benchmark                           # benchmark STT/TTS performance
vox config                              # interactive setup wizard
vox models list                         # show downloaded models
vox models download whisper-base.en     # download a specific model
vox models download kokoro --force      # force re-download if corrupted
vox models path                         # show where models are stored
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

WebSocket streaming at `ws://localhost:3000/v1/listen` &mdash; send PCM f32 LE frames at 16kHz mono, receive JSON events in real time:

```json
{"type": "speech_start"}
{"type": "partial", "text": "hello", "is_final": false, "stability": 0.5, "duration_ms": 600, "processing_time_ms": 2}
{"type": "partial", "text": "hello world", "is_final": false, "stability": 0.5, "duration_ms": 1000, "processing_time_ms": 3}
{"type": "transcript", "text": "hello world", "duration_ms": 1200, "processing_time_ms": 180}
{"type": "speech_end"}
```

When a streaming STT backend is available (sherpa-streaming model downloaded), partial results arrive incrementally as you speak. Without it, partials are omitted and you get the final transcript on speech end.

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
cd python
pip install maturin
maturin develop --features whisper,silero,kokoro
```

```python
from vox_voice import Vox, SileroVad, WhisperStt

vox = Vox(vad=SileroVad(), stt=WhisperStt("tiny.en"))
for result in vox.listen():
    print(result.text)
```

Built with PyO3. Same pipeline, Pythonic API.

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
| | Distil-Whisper tiny.en | 75MB | 6x faster than Whisper |
| | Distil-Whisper base.en | 142MB | 6x faster, high accuracy |
| | Distil-Whisper tiny.en-int8 | 19MB | Quantized, 4x smaller |
| | Distil-Whisper base.en-int8 | 35MB | Quantized, 4x smaller |
| | Sherpa SenseVoice | 230MB | Multilingual (zh/en/ja/ko/yue) |
| | Sherpa Streaming Zipformer | 27MB | Real-time partial results |
| **TTS** | Kokoro | 310MB | 50+ voices |
| | Kokoro INT8 | 77MB | Quantized, 4x smaller |
| | Piper | 63MB/voice | Multilingual (en/de/es/fr/zh) |
| | Pocket | 82MB | Pure Rust, edge/embedded |
| | Pocket INT8 | 20MB | Quantized, edge-optimized |
| | Chatterbox | 350MB | Voice cloning |

```bash
vox models download silero-vad          # 2MB
vox models download whisper-tiny.en     # 75MB
vox models download kokoro              # 310MB
vox models download kokoro-voices       # 27MB
vox models download piper-en-us         # 63MB (+ piper-en-us-config)
```

### Model Management and Troubleshooting

**Models Directory Location:**

Models are stored in platform-specific directories:
- **macOS**: `~/Library/Application Support/vox/models`
- **Linux**: `~/.local/share/vox/models`
- **Windows**: `{FOLDERPATH}/vox/models`

To find your models directory:
```bash
vox models path
```

**Corrupted or Partial Downloads:**

If a download is interrupted (e.g., disk full, network failure), you may see ONNX Runtime errors like:
```
External initializer offset out of bounds
```

To fix:
1. **Automatic cleanup**: Run `vox models list` to auto-clean partial downloads (`.part` files)
2. **Force re-download**: Use `vox models download <model-name> --force` to replace corrupted files
3. **Manual cleanup**: Delete corrupted files from the models directory (use `vox models path` to locate)

Example recovery workflow:
```bash
# Check for partial downloads and clean them up
vox models list

# Force re-download a corrupted model
vox models download kokoro --force

# Verify the download
vox models list
```

**TTS Backend: Pocket**

Pocket is a pure Rust TTS backend that requires no external dependencies and is optimized for edge/embedded devices:

```bash
# Install with pocket support
cargo install --git https://github.com/mrtozner/vox --features cli,pocket

# Use pocket TTS
vox speak "Hello" --backend pocket

# With Apple Metal GPU acceleration (macOS only)
cargo install --git https://github.com/mrtozner/vox --features cli,pocket-metal
vox speak "Hello" --backend pocket
```

Pocket is ideal when you need:
- Minimal dependencies
- Fast cold start
- Edge/embedded deployment
- Apple Silicon GPU acceleration (with `pocket-metal`)

<br>

## Performance Optimizations

**Distil-Whisper Backend**

Distil-Whisper is a distilled version of Whisper that runs 6x faster with minimal accuracy loss. Perfect for real-time applications and resource-constrained devices:

```bash
# Install with distil-whisper support
cargo install --git https://github.com/mrtozner/vox --features cli,distil-whisper

# Use distil-whisper for transcription
vox listen --stt-backend distil-whisper
vox listen --stt-backend distil-whisper --model base.en
```

Distil-Whisper is recommended when you need:
- Real-time transcription on CPU
- Lower latency (<100ms for short utterances)
- Raspberry Pi deployment
- Battery-powered devices

**INT8 Quantization**

Models can be quantized to INT8 for 4x smaller file sizes and faster inference with minimal quality loss:

```bash
# Quantized models are automatically downloaded when available
vox models download whisper-base.en-int8    # 35MB vs 142MB
vox models download kokoro-int8             # 77MB vs 310MB

# Use quantized model
vox listen --model base.en-int8
vox speak "Hello" --quantize int8
```

Benefits:
- 4x smaller download and disk usage
- 2-3x faster inference on CPU
- Lower memory footprint
- Ideal for Raspberry Pi and edge devices

**Model Caching**

In server mode, models are cached after first load, eliminating cold start latency:

```bash
vox serve --port 3000 --cache-models
```

First request loads the model (~2s), subsequent requests are instant. Cache persists until server restart.

<br>

## Raspberry Pi Deployment

Vox runs on Raspberry Pi 4 and newer. Recommended configuration for real-time performance:

**Recommended Models:**
- STT: Distil-Whisper tiny.en-int8 (6x faster, 4x smaller)
- VAD: Silero VAD v5 (lightweight, runs at 50x real-time)
- TTS: Pocket (pure Rust, no dependencies) or Piper with INT8

**Performance Benchmarks (Raspberry Pi 4, 4GB RAM):**

| Configuration | RTF (Real-Time Factor) | Memory | Notes |
|--------------|----------------------|--------|-------|
| Whisper base.en | 3.3x (unusable) | 450MB | Too slow for real-time |
| Whisper base.en-int8 | 1.8x | 220MB | Still too slow |
| Distil-Whisper base.en-int8 | 0.8x | 180MB | Real-time capable |
| Distil-Whisper tiny.en-int8 | 0.3x | 120MB | 3x faster than real-time |

RTF < 1.0 means faster than real-time (e.g., 0.3x = processes 3 seconds of audio in 1 second).

**Installation on Raspberry Pi:**

```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install vox with recommended features
cargo install --git https://github.com/mrtozner/vox \
  --features cli,distil-whisper,pocket

# Download quantized models
vox models download silero-vad
vox models download distil-whisper-tiny.en-int8
vox models download pocket-int8

# Run voice assistant
vox listen --stt-backend distil-whisper --model tiny.en-int8
vox speak "Hello" --backend pocket --quantize int8
```

**Memory Usage Guide:**
- Base system + Vox: ~150MB
- Distil-Whisper tiny.en-int8: +120MB
- Silero VAD: +30MB
- Pocket TTS: +50MB
- **Total**: ~350MB (fits comfortably on 1GB+ Pi models)

For Raspberry Pi 3 or Zero 2, use Sherpa streaming STT instead of Whisper.

<br>

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `cli` | no | CLI binary (`vox listen`, `vox speak`, `vox chat`, `vox serve`) |
| `server` | no | HTTP/WebSocket API server |
| `whisper` | yes | Whisper STT via whisper-rs |
| `distil-whisper` | no | Distil-Whisper STT (6x faster, quantization support) |
| `silero` | yes | Silero VAD via ONNX Runtime |
| `sherpa` | no | Sherpa-ONNX STT (SenseVoice, Zipformer, streaming) |
| `piper` | no | Piper TTS (multilingual) |
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

Measured on Apple M1 MacBook Pro:

| Metric | Value |
|--------|-------|
| VAD frame latency | ~1ms per 32ms frame |
| Whisper STT (3s utterance) | ~200ms |
| Streaming STT (per chunk) | <1ms (0.03x real-time) |
| End-to-end (speech end to text) | ~250ms |
| Piper TTS ("Hello world") | ~200ms |
| Chatterbox TTS ("Hello world") | ~2s |
| Memory (idle pipeline) | ~150MB |

<br>

## Examples

```bash
cargo run --example simple_listen --features whisper,silero       # mic to text
cargo run --example vad_only --features silero                    # speech detection only
cargo run --example voice_assistant --features whisper,silero,kokoro  # voice assistant
cargo run --example tts_speak --features kokoro                   # kokoro TTS
cargo run --example piper_speak --features piper                  # piper TTS
cargo run --example chatterbox_speak --features chatterbox        # voice cloning
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
