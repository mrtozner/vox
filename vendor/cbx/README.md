# cbx

A single-binary CLI for local text-to-speech.

This project wraps [Resemble AI's Chatterbox ONNX models](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX) in a standalone executable. The neural network inference is handled by ONNX Runtime; the Rust code provides a CLI interface, model management, and voice caching.

Based on [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox), a state-of-the-art open-source TTS model.

---

## Why cbx?

cbx offers a zero-dependency way to use Chatterbox:

- **Single binary** - Download one file, run it. No Python, no virtual environments, no pip.
- **Built-in model management** - Commands to download, list, and clean up model files.
- **Voice profile caching** - Encode a reference voice once, reuse it without re-processing.
- **Cross-platform** - Same CLI on macOS, Linux, and Windows.

The [official Chatterbox repository](https://github.com/resemble-ai/chatterbox) has more features (multilingual, GPU acceleration, fine-tuning). Use cbx when you want a simple, portable tool for basic TTS.

---

## Audio Samples

Listen to what Chatterbox can produce:

- [Official Chatterbox Turbo demos](https://resemble-ai.github.io/chatterbox_turbo_demopage/) - samples from Resemble AI
- [Original Chatterbox demos](https://resemble-ai.github.io/chatterbox_demopage/) - includes emotion exaggeration examples

cbx uses the same underlying model, so output quality is identical.

<!--
Samples generated with cbx (upload your own to GitHub releases):

**"Hello, and welcome to cbx."**
<audio controls src="https://github.com/srv1n/cbx/releases/download/samples/hello_world.wav"></audio>

**"The quick brown fox jumps over the lazy dog."**
<audio controls src="https://github.com/srv1n/cbx/releases/download/samples/sample_narration.wav"></audio>
-->

---

## Installation

### Quick Install (Recommended)

**macOS / Linux:**

```bash
curl -fsSL https://raw.githubusercontent.com/srv1n/cbx/main/packaging/scripts/install.sh | bash
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/srv1n/cbx/main/packaging/scripts/install.ps1 | iex
```

The installer downloads the appropriate binary for your platform from GitHub Releases. Models are downloaded separately on first use.

### Other Options

- **Homebrew:** See [Homebrew installation](#homebrew)
- **Manual:** Download from [GitHub Releases](https://github.com/srv1n/cbx/releases) and place on your PATH
- **From source:** `cargo build --release`

---

## Quick Start

### 1. Generate speech

```bash
cbx speak --text "Hello from cbx." --voice-wav ./your-voice.wav --out-wav ./output.wav
```

This will:
- Download the required model files (first run only, ~1-2 GB depending on variant)
- Encode your reference voice
- Generate speech and save to `output.wav`

### 2. (Optional) Use the bundled default voice

If you don't have a reference WAV file, install the pre-packaged default voice:

**macOS / Linux:**

```bash
curl -fsSL https://raw.githubusercontent.com/srv1n/cbx/main/packaging/scripts/install_default_voice.sh | bash
```

**Windows:**

```powershell
irm https://raw.githubusercontent.com/srv1n/cbx/main/packaging/scripts/install_default_voice.ps1 | iex
```

Then generate speech without specifying a voice:

```bash
cbx speak --text "Hello from cbx." --out-wav ./output.wav
```

### 3. (Optional) Cache a voice profile for faster runs

Voice encoding takes a few seconds. Cache it once to skip this step on future runs:

```bash
cbx voice add --name myvoice --voice-wav ./your-voice.wav
cbx speak --voice myvoice --text "Much faster now." --out-wav ./output.wav
```

---

## Commands

| Command | Description |
|---------|-------------|
| `cbx speak` | Generate speech from text |
| `cbx download` | Pre-download model files |
| `cbx sizes` | Show download sizes for each model variant |
| `cbx models` | List cached models |
| `cbx voice add` | Create a cached voice profile |
| `cbx voice list` | List cached voice profiles |
| `cbx voice remove` | Delete a voice profile |
| `cbx clean` | Remove cached models |

Run `cbx --help` or `cbx <command> --help` for details.

---

## Model Variants

Chatterbox is published in several ONNX variants. Use `--dtype` to select:

| Variant | Notes |
|---------|-------|
| `fp16` (default) | Good balance of size and speed |
| `fp32` | Largest, most compatible |
| `quantized`, `q4`, `q4f16` | Smaller downloads, speed varies by platform |

Check available sizes without downloading:

```bash
cbx sizes
```

Download a specific variant:

```bash
cbx download --dtype fp16
```

---

## Voice Profiles

cbx supports two ways to use reference voices:

**Direct path (slower):** Pass `--voice-wav` each time. The voice is re-encoded on every run.

```bash
cbx speak --voice-wav ./voice.wav --text "Hello" --out-wav ./out.wav
```

**Cached profile (faster):** Encode once, reuse many times.

```bash
cbx voice add --name myvoice --voice-wav ./voice.wav
cbx speak --voice myvoice --text "Hello" --out-wav ./out.wav
```

Voice profiles are tied to the model variant (`--dtype`). If you switch variants, create a new profile for that variant.

### Voice file requirements

- Format: WAV
- Channels: Mono or stereo (converted internally)
- Duration: 5-20 seconds of clear speech works well
- Quality: Clean recording without background noise

---

## Performance

### CPU is usually fastest

cbx includes platform-specific acceleration (CoreML on macOS, DirectML on Windows, CUDA on Linux), but for this particular model, **CPU execution is often faster** due to graph partitioning overhead in the neural accelerators.

The default is `--ep cpu`, which should work well on most systems.

### Thread tuning

cbx automatically uses all available CPU cores. Override with:

```bash
cbx --intra-threads 4 speak --text "Hello" --out-wav out.wav
```

On Apple Silicon, 4 threads often outperforms higher counts due to contention. Experiment to find what works best for your hardware.

### Benchmarks (M1 MacBook Pro)

| Configuration | Average Time |
|---------------|--------------|
| CPU, 4 threads | 22.7s |
| CPU, 2 threads | 29.0s |
| CPU, 8 threads | 33.1s |
| CoreML, 4 threads | 49.7s |

Text: "The quick brown fox jumps over the lazy dog. This is a benchmark run for cbx."

---

## Cache Management

### View cache status

```bash
cbx models        # Show cached model files
cbx models --long # Show detailed info including commit hashes
cbx downloads     # Show all downloaded variants
cbx voice list    # Show cached voice profiles
```

### Clean up

Preview what would be deleted:

```bash
cbx clean --dry-run
```

Delete specific variants:

```bash
cbx clean --dtype fp32 --yes
```

Delete everything (models and voices):

```bash
cbx clean --all --voices --yes
```

---

## Uninstall

**macOS / Linux:**

```bash
curl -fsSL https://raw.githubusercontent.com/srv1n/cbx/main/packaging/scripts/uninstall.sh | bash
```

**Windows:**

```powershell
irm https://raw.githubusercontent.com/srv1n/cbx/main/packaging/scripts/uninstall.ps1 | iex
```

This removes the binary. To also remove cached models and voices:

```bash
cbx clean --all --voices --yes
```

---

## Homebrew

A Homebrew formula template is available at `packaging/homebrew/cbx.rb.template`. Once configured in a tap:

```bash
brew tap srv1n/tap
brew install cbx
```

---

## Building from Source

```bash
cargo build --release
./target/release/cbx --help
```

Enable platform-specific acceleration:

```bash
cargo build --release --features coreml    # macOS
cargo build --release --features directml  # Windows
cargo build --release --features cuda      # Linux with NVIDIA GPU
```

---

## Acknowledgments

This project would not exist without:

- [Resemble AI](https://www.resemble.ai/) for creating and open-sourcing [Chatterbox](https://github.com/resemble-ai/chatterbox)
- The [Chatterbox Turbo ONNX export](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX) on Hugging Face
- The [ort](https://crates.io/crates/ort) crate for Rust ONNX Runtime bindings

---

## License

Dual-licensed under Apache License 2.0 or MIT, at your option.

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT).
