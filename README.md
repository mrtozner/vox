# Vox

**The open-source voice AI framework for Rust.**

Run Whisper + LLM + TTS on your Raspberry Pi. One crate. No cloud. No Python.

```text
Audio In --> VAD (Silero) --> STT (Whisper) --> Your Code --> TTS --> Audio Out
```

## Why Vox?

- **Local & private** -- your voice never leaves your device
- **Fast** -- Rust performance, no Python runtime overhead
- **Simple** -- 10 lines to transcribe speech from your microphone
- **Modular** -- swap VAD, STT, TTS backends via traits
- **Cross-platform** -- macOS, Linux, Raspberry Pi

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
vox = "0.1"
```

Download models:

```bash
# Whisper tiny.en (~75MB)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin

# Silero VAD (~2MB)
wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
```

Listen and transcribe:

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

## Architecture

```text
+--------+     +-----+     +-----+     +-----------+     +-----+
|  Mic   | --> | VAD | --> | STT | --> | Callback  | --> | TTS |
| (cpal) |     |     |     |     |     | (your fn) |     |     |
+--------+     +-----+     +-----+     +-----------+     +-----+
                  |                          |
            Silero ONNX               VoxContext gives
            v5 model                  access to speak()
```

Audio is captured from the default microphone via `cpal`, resampled to 16kHz mono,
and fed frame-by-frame to the VAD backend. When the VAD detects a complete utterance
(speech followed by silence), it hands the audio to the STT backend for transcription.
The result is passed to your callback along with a `VoxContext` that can optionally
speak back via TTS.

## Backends

| Component | Default | Alternatives |
|-----------|---------|-------------|
| VAD | Silero VAD v5 | Implement `VadBackend` trait |
| STT | Whisper (via whisper.cpp) | Implement `SttBackend` trait |
| TTS | Coming soon (Kokoro, Piper) | Implement `TtsBackend` trait |

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `whisper` | yes | Whisper STT via whisper-rs |
| `silero` | yes | Silero VAD via ONNX Runtime |
| `kokoro` | no | Kokoro TTS (coming soon) |
| `piper` | no | Piper TTS (coming soon) |
| `tts` | no | Audio playback for TTS output |

## Hardware Support

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon) | Tested |
| Linux (x86_64) | Tested |
| Raspberry Pi 5 | Supported |
| NVIDIA Jetson | Supported |

## Models

### Whisper

Vox uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) GGML models.
Pick a size based on your hardware:

| Model | Size | RAM | Speed | Quality |
|-------|------|-----|-------|---------|
| `ggml-tiny.en.bin` | 75MB | ~125MB | Fastest | Good for English |
| `ggml-base.en.bin` | 142MB | ~210MB | Fast | Better accuracy |
| `ggml-small.en.bin` | 466MB | ~600MB | Medium | High accuracy |
| `ggml-medium.en.bin` | 1.5GB | ~1.7GB | Slow | Highest accuracy |

Download from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main):

```bash
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
```

Or load by model size from a directory:

```rust
use vox::{WhisperBackend, WhisperModel};

let stt = WhisperBackend::from_dir("./models", WhisperModel::TinyEn)?;
```

### Silero VAD

Download the ONNX model (~2MB):

```bash
wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
```

Tune detection sensitivity:

```rust
use vox::{SileroVad, VadConfig};

let vad = SileroVad::with_config("silero_vad.onnx", VadConfig {
    speech_threshold: 0.5,    // probability threshold
    silence_duration_ms: 500, // silence before end-of-speech
    min_speech_ms: 250,       // ignore very short sounds
})?;
```

## Custom Backends

Implement the backend traits to plug in your own engines:

```rust
use async_trait::async_trait;
use vox::{VadBackend, VadEvent, AudioChunk, VoxError};

struct MyVad { /* ... */ }

#[async_trait]
impl VadBackend for MyVad {
    async fn process_frame(&mut self, frame: &AudioChunk) -> Result<Vec<VadEvent>, VoxError> {
        // your VAD logic
        Ok(vec![VadEvent::Silence])
    }

    fn reset(&mut self) { /* ... */ }
    fn frame_size(&self) -> usize { 512 }
    fn sample_rate(&self) -> u32 { 16000 }
}
```

The same pattern applies to `SttBackend` and `TtsBackend`.

## Performance

Benchmarks measured on Apple M1 MacBook Pro, Whisper `tiny.en` model:

| Metric | Value |
|--------|-------|
| VAD frame latency (Silero) | ~1ms per 32ms frame |
| STT latency (tiny.en, 3s utterance) | ~200ms |
| End-to-end (speech end -> text) | ~250ms |
| Memory (idle pipeline) | ~150MB |

_Benchmarks are approximate and vary by hardware. Formal benchmarks coming soon._

## Examples

```bash
# Simple mic-to-text
cargo run --example simple_listen

# VAD-only (speech detection without transcription)
cargo run --example vad_only

# Voice assistant with LLM placeholder
cargo run --example voice_assistant
```

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes and add tests
4. Run `cargo test` and `cargo clippy`
5. Submit a pull request

Please keep PRs focused on a single change. For larger features, open an issue first to discuss the approach.

## License

MIT OR Apache-2.0
