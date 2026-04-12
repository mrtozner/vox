<p align="center">
  <img src="assets/vox-logo.jpg" alt="Vox" width="200">
  <p align="center"><strong>Local-first voice AI framework. Speech-to-text, text-to-speech, speaker diarization, and voice chat &mdash; all running on your machine.</strong></p>
</p>

<p align="center">
  <a href="https://github.com/mrtozner/vox/actions/workflows/ci.yml"><img src="https://github.com/mrtozner/vox/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/vox"><img src="https://img.shields.io/crates/v/vox.svg" alt="crates.io"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg" alt="License"></a>
</p>

---

No API keys. No cloud. No data leaving your machine.

```
Mic --> VAD (Silero) --> STT (Whisper/Sherpa) --> Speaker ID --> Your Code --> TTS (Kokoro/Piper/Qwen3) --> Speaker
```

<br>

## Highlights

- **Speech-to-Text** &mdash; Whisper, Distil-Whisper (6x faster), Sherpa-ONNX (streaming + multilingual)
- **Text-to-Speech** &mdash; Kokoro (57 voices, 9 languages), Piper (multilingual), Qwen3 (state-of-the-art), Pocket (pure Rust), Chatterbox (voice cloning)
- **Speaker Diarization** *(experimental)* &mdash; Know *who* is speaking. Real-time speaker identification with voice embeddings, auto-enrollment, and a persistent speaker database
- **Live Talk** *(experimental)* &mdash; Barge-in voice chat. Talk to an LLM and interrupt it mid-sentence, just like a real conversation
- **Web Interface** &mdash; 5-tab browser UI: Listen, Speak, Chat, Live Talk, Dashboard
- **HTTP + WebSocket API** &mdash; REST endpoints and 4 WebSocket channels for real-time streaming
- **Fully Local** &mdash; Everything runs on-device. Works offline after model download
- **Pluggable Backends** &mdash; Swap any component via Rust traits

<br>

## Quick Start

```bash
# Install with CLI + Kokoro TTS
cargo install --git https://github.com/mrtozner/vox --features cli,kokoro

# Transcribe speech from your microphone
vox listen

# Text-to-speech with voice selection
vox speak "Hello from Vox!" --voice af_heart

# Voice chat with Ollama
vox chat --llm llama3.2

# Start the web server
cargo install --git https://github.com/mrtozner/vox --features cli,server,kokoro,piper
vox serve --port 3000
# Open http://localhost:3000
```

Models auto-download on first run. Pass `-y` to skip prompts.

<br>

## Web Interface

Start the server and open your browser:

```bash
vox serve --port 3000
```

| Tab | What it does |
|-----|-------------|
| **Listen** | Real-time mic transcription with optional speaker diarization *(experimental)*. Toggle "Identify speakers" to see who is talking, with color-coded transcript entries and a speaker sidebar for renaming/removing speakers. |
| **Speak** | Text-to-speech synthesis. Pick a voice from the dropdown (57 Kokoro voices, Piper voices, etc.), type text, and hear it. |
| **Chat** | Voice chat with any Ollama LLM. Speak a question, get a spoken answer. |
| **Live Talk** | Experimental barge-in voice chat. Full-duplex &mdash; interrupt the LLM mid-sentence by speaking. Select voice and LLM model from dropdowns. |
| **Dashboard** | Server stats, loaded backends, model cache info, and environment capabilities. |

No separate frontend build. The UI is a single embedded HTML file served by the Rust binary.

<br>

## CLI

```bash
vox listen                                 # transcribe from microphone (Whisper)
vox listen --model base.en                 # use a larger Whisper model
vox listen --stt-backend sherpa            # Sherpa SenseVoice (multilingual)
vox listen --stt-backend sherpa-streaming  # real-time streaming transcription
vox listen --stt-backend distil-whisper    # Distil-Whisper (6x faster)
vox speak "Hello from Vox!"               # text-to-speech
vox speak "Hello" --voice af_heart         # Kokoro with voice selection
vox speak "Hello" --backend qwen3          # Qwen3 state-of-the-art TTS
vox speak "Hallo" --backend piper          # Piper multilingual TTS
vox speak "Hi" --backend pocket            # pure Rust TTS (edge-ready)
vox speak "Hi" --backend chatterbox --voice ref.wav  # voice cloning
vox chat --llm llama3.2                    # voice chat with Ollama
vox serve --port 3000                      # start web + API server
vox test                                   # audio I/O diagnostics
vox benchmark                              # benchmark STT/TTS performance
vox config                                 # interactive setup wizard
vox models list                            # show downloaded models
vox models download kokoro                 # download a specific model
vox models path                            # show models directory
```

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

# List Ollama models
curl http://localhost:3000/v1/ollama-models

# Environment capabilities
curl http://localhost:3000/v1/capabilities

# Server stats
curl http://localhost:3000/v1/stats

# Health check
curl http://localhost:3000/health
```

### WebSocket Endpoints

**`/v1/listen`** &mdash; Real-time speech-to-text

Send PCM f32 LE frames at 16kHz mono. Receive JSON events:

```json
{"type": "speech_start"}
{"type": "partial", "text": "hello", "is_final": false}
{"type": "transcript", "text": "hello world", "duration_ms": 1200, "speaker": "speaker_1"}
{"type": "speech_end"}
```

When diarization is enabled (server feature), transcripts include a `speaker` field identifying the speaker.

**`/v1/speak`** &mdash; Streaming text-to-speech

Send JSON, receive chunked audio for gapless playback:

```json
{"text": "Hello world", "voice": "af_heart"}
```

**`/v1/converse`** &mdash; Continuous voice chat

Combines VAD + STT + LLM + TTS in a single WebSocket. Speak, get a spoken LLM response, repeat.

**`/v1/live-talk`** &mdash; Barge-in voice chat (experimental)

Full-duplex conversation. You can interrupt the LLM mid-response by speaking. Requires an Ollama instance.

<br>

## Rust Library

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

```bash
cd python && pip install maturin
maturin develop --features whisper,silero,kokoro
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
            Silero ONNX              ECAPA-TDNN         Your callback
            v5 model                voice embeddings     gets text +
                                    + SQLite DB          VoxContext
```

Audio is captured via `cpal`, resampled to 16kHz mono, and fed frame-by-frame to VAD. On speech end, the utterance goes to STT. If diarization is enabled, the audio is also processed by a speaker encoder that extracts a 512-dimensional voice embedding, compares it against known speakers, and assigns an identity. Your callback receives the text, speaker label, and a `VoxContext` for optional TTS reply.

<br>

## Models

| Component | Model | Size | Notes |
|-----------|-------|------|-------|
| **VAD** | Silero VAD v5 | 2 MB | Speech activity detection |
| **STT** | Whisper tiny.en | 75 MB | Fast, English |
| | Whisper base.en | 142 MB | Better accuracy |
| | Whisper small.en | 466 MB | High accuracy |
| | Whisper medium.en | 1.5 GB | Highest accuracy |
| | Distil-Whisper tiny.en | 75 MB | 6x faster than Whisper |
| | Sherpa SenseVoice | 230 MB | Multilingual (zh/en/ja/ko/yue) |
| | Sherpa Streaming Zipformer | 27 MB | Real-time partial results |
| **TTS** | Kokoro | 310 MB | 57 voices, 9 languages, 24 kHz |
| | Kokoro INT8 | 77 MB | Quantized, 4x smaller |
| | Qwen3 0.6B | 1.2 GB | 20 voices, 10 languages, streaming |
| | Qwen3 1.7B | 3.4 GB | Higher quality, streaming |
| | Piper | 63 MB/voice | Multilingual, fast CPU synthesis |
| | Pocket | 82 MB | Pure Rust, edge/embedded |
| | Chatterbox | 350 MB | Voice cloning from reference audio |
| **Diarization** | ECAPA-TDNN | ~30 MB | 512-dim speaker embeddings |

```bash
vox models download silero-vad           # 2 MB
vox models download whisper-tiny.en      # 75 MB
vox models download kokoro               # 310 MB
vox models download kokoro-voices        # 27 MB
vox models download piper-en-us          # 63 MB
vox models download qwen3-0.6b           # 1.2 GB
```

### Model Storage

Models are stored in platform-specific directories:
- **macOS**: `~/Library/Application Support/vox/models`
- **Linux**: `~/.local/share/vox/models`
- **Windows**: `{FOLDERPATH}/vox/models`

Run `vox models path` to find yours. Use `vox models download <name> --force` to re-download corrupted files.

<br>

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `cli` | no | CLI binary (`vox listen`, `vox speak`, `vox chat`, `vox serve`) |
| `server` | no | HTTP/WebSocket server with web UI (includes `diarization`) |
| `whisper` | **yes** | Whisper STT via whisper-rs |
| `distil-whisper` | no | Distil-Whisper STT (6x faster) |
| `silero` | **yes** | Silero VAD via ONNX Runtime |
| `sherpa` | no | Sherpa-ONNX STT (SenseVoice, Zipformer, streaming) |
| `kokoro` | no | Kokoro TTS (57 voices, 9 languages) |
| `qwen3` | no | Qwen3 TTS (state-of-the-art, streaming) |
| `qwen3-metal` | no | Qwen3 with Apple Metal GPU |
| `qwen3-cuda` | no | Qwen3 with NVIDIA CUDA GPU |
| `piper` | no | Piper TTS (multilingual, fast CPU) |
| `pocket` | no | Pocket TTS (pure Rust, edge-ready) |
| `pocket-metal` | no | Pocket with Apple Metal GPU |
| `chatterbox` | no | Chatterbox TTS (voice cloning) |
| `chatterbox-coreml` | no | Chatterbox with CoreML (macOS) |
| `diarization` | no | Speaker identification (auto-enabled by `server`) |
| `tts` | no | Audio playback for TTS output |

### Common Feature Combinations

```bash
# Minimal: just transcription
cargo install --git https://github.com/mrtozner/vox --features cli

# Full server with Kokoro + Piper voices
cargo install --git https://github.com/mrtozner/vox --features cli,server,kokoro,piper

# Server with Qwen3 on Apple Silicon
cargo install --git https://github.com/mrtozner/vox --features cli,server,qwen3-metal

# Server with Sherpa streaming STT + Kokoro TTS
cargo install --git https://github.com/mrtozner/vox --features cli,server,sherpa,kokoro

# Edge deployment (smallest footprint)
cargo install --git https://github.com/mrtozner/vox --features cli,distil-whisper,pocket
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

Vox runs on Raspberry Pi 4+. Recommended setup:

```bash
cargo install --git https://github.com/mrtozner/vox --features cli,distil-whisper,pocket
vox models download silero-vad
vox models download distil-whisper-tiny.en-int8
vox listen --stt-backend distil-whisper --model tiny.en-int8
```

| Config | RTF | Memory |
|--------|-----|--------|
| Distil-Whisper tiny.en-int8 | 0.3x | 120 MB |
| Distil-Whisper base.en-int8 | 0.8x | 180 MB |
| Pocket TTS INT8 | real-time | ~50 MB |

Total footprint: ~350 MB on a 1 GB Pi.

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

<br>

## Platform Support

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon) | Tested |
| macOS (Intel) | Tested |
| Linux (x86_64) | CI tested |
| Windows (x86_64) | CI tested |

<br>

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes and add tests
4. Run `cargo test` and `cargo clippy`
5. Submit a pull request

<br>

## License

MIT OR Apache-2.0
